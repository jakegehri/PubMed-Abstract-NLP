import torch
import torchmetrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'

accuracy = torchmetrics.Accuracy(task = 'binary', num_classes=2).to(device)
precision = torchmetrics.Precision(task = 'binary', num_classes=2).to(device)
recall = torchmetrics.Recall(task = 'binary', num_classes=2).to(device)
f1score = torchmetrics.F1Score(task = 'binary', num_classes=2).to(device)
auroc = torchmetrics.AUROC(task='binary').to(device)





def train_epoch(model, optim, loss_fn, train_loader, config, progress_bar):
    
    model.train()
        
    train_loss = 0

    for batch_idx, batch in enumerate(train_loader):

        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).float()

        outputs = model(input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask)
        logits = outputs.squeeze(1)
        
        loss = loss_fn(logits, labels)
        
        train_loss += loss

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        progress_bar.update(1)

    return train_loss / batch_idx



def valid_epoch(model, loss_fn, valid_loader, config):
    
    valid_loss = 0
    
    model.eval()
    
    for batch_idx, batch in enumerate(valid_loader):
        
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).float()

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask)
        logits = outputs.squeeze(1)
        
        loss = loss_fn(logits, labels)
        
        valid_loss += loss
        
        probs = logits.sigmoid()
    
        preds = (probs > config['threshold']).float()
        
        accuracy.update(labels, preds)
        precision.update(labels, preds)
        recall.update(labels, preds)
        f1score.update(labels, preds)
        confusion_matrix.update(labels, preds)
        auroc.update(labels, preds)
        
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