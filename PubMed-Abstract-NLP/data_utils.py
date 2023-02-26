import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from transformers import AutoTokenizer
import numpy as np


df = pd.read_excel('23.01.05 - MasterTable.xlsx')


def get_data(config):
    columns = ['title', 'abstract_text', config['label']]
    label_df = df[columns]
    label_df.dropna(inplace=True)
    label_df.reset_index(drop=True, inplace=True)
    label_df['text'] = label_df['title'] + " " + label_df['abstract_text']
    label_df = label_df[['text', config['label']]]
    label_df.columns = ['text', 'labels']
    label_df = label_df.astype({'labels': 'int'})
    
    texts = label_df['text'].values
    labels = label_df['labels'].values
    
    return texts, labels



class PubMedText(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    
def make_dataloader(config):
    texts, labels = get_data(config)
    
    train_text, valid_text, train_labels, valid_labels = train_test_split(texts, labels, test_size=0.10, random_state=0)
    
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_labels])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), )
    
    tokenizer = AutoTokenizer.from_pretrained(config['checkpoint'])
    
    train_encodings = tokenizer(list(train_text), truncation=True, padding=True, max_length = 512)
    valid_encodings = tokenizer(list(valid_text), truncation=True, padding=True, max_length = 512)
    
    train_dataset = PubMedText(train_encodings, train_labels)
    valid_dataset = PubMedText(valid_encodings, valid_labels)
    
    if config['weighted_random_sampler'] == True:
    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
        
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, valid_loader