import torch
from torch import nn
from transformers import AutoModel
from config import config
import torch.nn.functional as F


class PubMedBERT_TextCNN(torch.nn.Module):
    def __init__(self,  num_filters = 256, kernel_sizes=[3, 4, 5], embedding_dim = 768):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(config['checkpoint'], output_hidden_states = True)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, embedding_dim), padding = (k-2 ,0)) for k in kernel_sizes])
        self.dropout = nn.Dropout(config['dropout'])
        self.linear = nn.Linear(in_features=768, out_features=1)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids, token_type_ids, attention_mask).pooler_output #.last_hidden_state.unsqueeze(1)
        #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        #x = torch.cat(x, 1)
        x = self.dropout(x)  
        x = self.linear(x)
        return x
    
    
class PubMedBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(config['checkpoint'], output_hidden_states = True)
        
        self.dropout = nn.Dropout(config['dropout'])
        
        self.linear = nn.Linear(in_features=768*4, out_features=1)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids, token_type_ids, attention_mask).hidden_states
        x = torch.stack(x)
        x = torch.cat((x[-1], x[-2], x[-3], x[-4]),-1)
        x = x[:, 0]
        x = self.linear(x)
        return x