from data_utils import make_dataloader
from model_utils import PubMedBERT
import torch
from torch import nn
from config import config
from tqdm.auto import tqdm
from train_utils import train_epoch, valid_epoch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make(config):
    
    train_loader, valid_loader = make_dataloader(config)
    
    model = PubMedBERT().to(device)
    
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    
    optim = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    return model, train_loader, valid_loader, optim, loss_fn


def main(config=config):
    
    results = {'train_loss': [],
               'valid_loss': [],
               'accuracy' : [],
               'precision': [],
               'recall': [],
               'f1': [],
               'auc': []
               }

    model, train_loader, valid_loader, optim, loss_fn = make(config)

    num_training_steps = config['num_epochs'] * len(train_loader)

    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(config['num_epochs']):

        train_loss = train_epoch(model, optim, loss_fn, train_loader, config, progress_bar)
        
        results['train_loss'].append(train_loss.cpu().detach().numpy())
    
        valid_loss, accuracy, precision, recall, f1, auc = valid_epoch(model, loss_fn, valid_loader, config)

        results['valid_loss'].append(valid_loss.cpu().detach().numpy())
        results['accuracy'].append(accuracy)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        results['auc'].append(auc)
        
        
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
    
    return results

main(config)