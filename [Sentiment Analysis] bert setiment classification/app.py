from src import config
import torch

from flask import Flask, jsonify, request
from src.model import BERTBasedUncased

import time

app = Flask(__name__)

model = None

def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
    review,
    None,
    add_special_tokens=True,
    max_length=max_len,
    pad_to_max_length=True,
    )   

    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(config.DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(config.DEVICE, dtype=torch.long)
    mask = mask.to(config.DEVICE, dtype=torch.long)

    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    print(outputs)
    return outputs[0][0]

@app.route('/predict')
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        "sentence": str(sentence),
        "taked time": str(time.time() - start_time),
    }
    return jsonify(response)

if __name__ == '__main__':
    model = BERTBasedUncased()
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(config.DEVICE)
    model.eval()
    app.run(host='0.0.0.0', port='8080')