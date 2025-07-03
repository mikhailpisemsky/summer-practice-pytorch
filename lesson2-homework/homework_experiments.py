import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from models.utils import accuracy, log_epoch
from homework_datasets import DataFromCSV
from sklearn.model_selection import train_test_split
from models.logreg_torch import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, r2_score

# Задание 3: Эксперименты и анализ
# 3.1 Исследование гиперпараметров
# Проведите эксперименты с различными:
# - Скоростями обучения (learning rate)
# - Размерами батчей
# - Оптимизаторами (SGD, Adam, RMSprop)
experiments = {
    "lr": [0.1, 0.01, 0.001],
    "batch_size": [16, 32, 64],
    "optimizer": [optim.SGD, optim.Adam, optim.RMSprop]
}

def compute_metrics(y_true, y_pred_logits, zero_division=0):
    y_pred_labels = y_pred_logits.argmax(dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    precision = precision_score(y_true, y_pred_labels, average='macro', zero_division=zero_division)
    recall = recall_score(y_true, y_pred_labels, average='macro', zero_division=zero_division)
    f1 = f1_score(y_true, y_pred_labels, average='macro', zero_division=zero_division)
    try:
        y_true_one_hot = np.eye(np.max(y_true)+1)[y_true]
        roc_auc = roc_auc_score(
            y_true_one_hot,
            y_pred_logits.softmax(dim=1).cpu().numpy(),
            multi_class='ovr',
            average='macro'
        )
    except ValueError:
        roc_auc = np.nan
    return precision, recall, f1, roc_auc

# Данные для регрессии: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

# Создаём датасет и даталоадер
database_breast = DataFromCSV("data/breast-cancer-binary-classification.csv", target="diagnosis")
# Разделяем данные на тестовую и тренеровачную выборки
X_train, X_test, y_train, y_test = train_test_split(database_breast.X, database_breast.y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
for exp_lr in experiments['lr']:
    for exp_batch_size in experiments['batch_size']:
        for exp_optimizer in experiments['optimizer']:
            print(f"Эксперемент с гиперпараметрами: lr={exp_lr}, batch_size={exp_batch_size}, optimizer={exp_optimizer}")
            train_dataloader = DataLoader(train_dataset, batch_size=exp_batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=exp_batch_size, shuffle=True)
            # Создаём модель, функцию потерь и оптимизатор
            model = LogisticRegression(in_features=database_breast.X.shape[1])
            criterion = nn.BCEWithLogitsLoss()
            optimizer = exp_optimizer(model.parameters(), lr=exp_lr)
            best_params = {'avg_loss': 0, 'avg_acc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'roc_auc': 0}

            # Обучаем модель
            epochs = 100
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss, total_acc = 0, 0
                for i, (batch_X, batch_y) in enumerate(train_dataloader):
                    batch_y = batch_y.view(-1)
                    optimizer.zero_grad()
                    logits = model(batch_X).squeeze(1)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                    # Вычисляем accuracy
                    y_pred = torch.sigmoid(logits)
                    acc = accuracy(y_pred, batch_y)
                        
                    total_loss += loss.item()
                    total_acc += acc
                avg_loss = total_loss / (i + 1)
                avg_acc = total_acc / (i + 1)

                model.eval()
                all_preds, all_targets = [], []               

                with torch.no_grad():
                    for test_X, test_y in test_dataloader:
                        logits = model(test_X)
                        all_preds.append(torch.sigmoid(logits))
                        all_targets.append(test_y)

                all_preds = torch.cat(all_preds)
                all_targets = torch.cat(all_targets).long().squeeze()

                precision, recall, f1, roc_auc = compute_metrics(all_targets, all_preds)
                if(avg_loss > best_params['avg_loss']): best_params['avg_loss']=avg_loss
                if(avg_acc > best_params['avg_acc']): best_params['avg_acc']=avg_acc
                if(precision > best_params['precision']): best_params['precision']=precision
                if(recall > best_params['recall']): best_params['recall']=recall
                if(f1 > best_params['f1']): best_params['f1']=f1
                if(roc_auc > best_params['roc_auc']): best_params['roc_auc']=roc_auc
                if(epoch % 10 == 0):
                    log_epoch(epoch, avg_loss, acc=avg_acc, precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)
                if(epoch % 100 == 0):
                    print(f"Эксперемент с гиперпараметрами: lr={exp_lr}, batch_size={exp_batch_size}, optimizer={exp_optimizer}")
                    log_epoch(epoch, best_params['avg_loss'], acc=best_params['avg_acc'], precision=best_params['precision'], recall=best_params['recall'], f1=best_params['f1'], roc_auc=best_params['roc_auc'])
