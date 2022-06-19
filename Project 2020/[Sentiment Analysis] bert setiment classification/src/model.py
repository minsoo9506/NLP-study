import transformers
import torch.nn as nn

class BERTBasedUncased(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, out2 = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        output = self.bert_drop(out2)
        output = self.out(output)
        return output