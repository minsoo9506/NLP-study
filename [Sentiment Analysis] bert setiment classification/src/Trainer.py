from tqdm import tqdm
import torch
import torch.nn as nn

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, scheduler, config):

    model.train()

    for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = data['ids'].to(config.DEVICE, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(config.DEVICE, dtype=torch.long)
        mask = data['mask'].to(config.DEVICE, dtype=torch.long)
        targets = data['targets'].to(config.DEVICE, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs, targets)
        loss.backward()

        if (batch_idx+1) % config.ACCUMULATION == 0:
            optimizer.step()
            scheduler.step()
        
        # if batch_idx == 100:
        #     break

def eval_fn(data_loader, model, config):

    model.eval()
    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = data['ids'].to(config.DEVICE, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(config.DEVICE, dtype=torch.long)
            mask = data['mask'].to(config.DEVICE, dtype=torch.long)
            targets = data['targets'].to(config.DEVICE, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            # if batch_idx == 100:
            #     break

    return final_outputs, final_targets