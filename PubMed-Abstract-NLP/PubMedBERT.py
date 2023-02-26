## This script comes from PubMedBERT.ipynb

from datasets import load_dataset
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorForTokenClassification, AdamW, get_scheduler, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb
import random
from sklearn.model_selection import train_test_split
import time
import torchmetrics
from sklearn.model_selection import KFold
from datetime import date
from torch import nn
import pprint

#wandb.login()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
today = date.today()

config = {
    'checkpoint': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'label': 'penetrance',
    'batch_size': 16,
    'num_epochs': 2,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'weighted_random_sampler': False,
    'k_folds': 10
     }
'''
sweep_config = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'accuracy'},
    'parameters': 
    {
        'checkpoint': {'values': ['microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext']},
        'architecture': {'values': ['PubMedBERT']},
        'label': {'values': ['penetrance']},
        'batch_size': {'values': [16]},
        'num_epochs': {'values': [2]},
        'learning_rate': {'values': [5e-5]},
        'weight_decay': {'values': [0.01]},
        'weighted_random_sampler': {'values': [True]}
     }
}
'''

#sweep_id = wandb.sweep(sweep_config, project="kevin_project")

#pprint.pprint(sweep_config)

accuracy = torchmetrics.Accuracy(task = 'binary').to(device)
precision = torchmetrics.Precision(task = 'binary').to(device)
recall = torchmetrics.Recall(task = 'binary').to(device)
f1score = torchmetrics.F1Score(task = 'binary').to(device)
auroc = torchmetrics.AUROC(task='binary').to(device)

df = pd.read_excel('23.01.05 - MasterTable.xlsx')

columns = ['title', 'abstract_text', 'penetrance', 'incidence', 'metaanalysis', 'polymorphism', 'Germline', 'Somatic']

df = df[columns]

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
    
    train_text, valid_test_text, train_labels, valid_test_labels = train_test_split(texts, labels, test_size=0.10, random_state=0)
    valid_text, test_text, valid_labels, test_labels = train_test_split(valid_test_text, valid_test_labels, test_size=0.5, random_state=0)
    
    pd.DataFrame({'texts': test_text, f"{config['label']}": test_labels}).to_csv(f"{config['label']}_testset.csv", index = False)
    
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


def make(config):
    
    train_loader, valid_loader = make_dataloader(config)
    
    model = AutoModelForSequenceClassification.from_pretrained(config['checkpoint']).to(device)
    
    optim = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    return model, train_loader, valid_loader, optim

def validate(model, valid_loader):
    
    valid_loss = 0
    
    model.eval()
    
    for batch_idx, batch in enumerate(valid_loader):
        
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']
        
        valid_loss += loss
        
        preds = torch.argmax(logits, 1)
        
        accuracy.update(preds, labels)
        precision.update(preds, labels)
        recall.update(preds, labels)
        f1score.update(preds, labels)
        auroc.update(preds, labels)
        
    acc = accuracy.compute().cpu().numpy()
    prec = precision.compute().cpu().numpy()
    rec = recall.compute().cpu().numpy()
    f1 = f1score.compute().cpu().numpy()
    auc = auroc.compute().cpu().numpy()
    

    accuracy.reset()
    precision.reset()
    recall.reset()
    f1score.reset()
    auroc.reset()
    
    return valid_loss / batch_idx, acc, prec, rec, f1, auc

def train_epoch(model, optim, train_loader, progress_bar):
    
    epoch_loss = 0
    
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):

        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']
        
        epoch_loss += loss

        preds = torch.argmax(logits, 1)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        progress_bar.update(1)
        
    
    return epoch_loss / batch_idx


def train(model, optim, train_loader, valid_loader, config, progress_bar):
    
    #wandb.watch(model, log="all", log_freq=10)
    
    for epoch in range(config['num_epochs']):
        
        train_loss = train_epoch(model, optim, train_loader, progress_bar)
        
        valid_loss, accuracy, precision, recall, f1, auc = validate(model, valid_loader)
        
        #wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1})
        
        print(f"""Epoch {epoch+1}: 
                  Train Loss = {train_loss},
                  Valid Loss = {valid_loss}
                  Accuracy = {accuracy},
                  Precision = {precision},
                  Recall = {recall},
                  F1 = {f1},
                  AUC = {auc}
                """
            )

def main(config):
    
    #run = wandb.init(project="kevin_project", config=config)
        
    #config = wandb.config

    model, train_loader, valid_loader, optim = make(config)

    num_training_steps = config['num_epochs'] * len(train_loader)

    progress_bar = tqdm(range(num_training_steps))

    train(model, optim, train_loader, valid_loader, config, progress_bar)

#wandb.agent(sweep_id, function=main, count = 2)

results = main(config)