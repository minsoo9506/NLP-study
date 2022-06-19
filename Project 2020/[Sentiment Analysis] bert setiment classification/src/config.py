import transformers
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'src/model_save'
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 1
ACCUMULATION = 2
TRAINING_FILE = '../IMDB.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)