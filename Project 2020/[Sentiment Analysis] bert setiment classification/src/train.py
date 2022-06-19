import config
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import dataset
from model import BERTBasedUncased
import Trainer
import warnings
warnings.filterwarnings('ignore')

def run():
    df = pd.read_csv(config.TRAINING_FILE).fillna('none')
    df.sentiment = df.sentiment.apply(
        lambda x : 1 if x == 'positive' else 0
    )

    df_train, df_val = train_test_split(
        df,
        test_size=0.1,
        random_state=0,
        stratify=df.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    print('Dataset is Done')

    train_dataset = dataset.BERTDataset(
        review=df_train.review.values,
        target=df_train.sentiment.values,
    )
    val_dataset = dataset.BERTDataset(
        review=df_val.review.values,
        target=df_val.sentiment.values
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4
    )

    print('Dataloader is Done') 
    
    model = BERTBasedUncased().to(config.DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_parameters = [
        {'params' : [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.001},
        {'params' : [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    best_acc = 0
    for epoch in range(config.EPOCHS):
        Trainer.train_fn(train_dataloader, model, optimizer, scheduler, config)
        outputs, targets = Trainer.eval_fn(val_dataloader, model, config)
        outputs = np.array(outputs) >= 0.5
        acc = accuracy_score(targets, outputs)
        print(f'Accuracy : {acc:.3f}')
        if acc > best_acc :
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_acc = acc

if __name__ == '__main__':
    run()