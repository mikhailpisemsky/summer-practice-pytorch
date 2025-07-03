import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from models.linreg_torch import LinearRegression
from models.logreg_torch import LogisticRegression
from models.utils import make_regression_data, make_classification_data, accuracy, log_epoch, RegressionDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, r2_score

# Задание 1: Модификация существующих моделей
# 1.1 Расширение линейной регрессии
# Модифицируйте существующую линейную регрессию:
# - Добавьте L1 и L2 регуляризацию
def regularization(model, p, u):
    l1, l2 = 0, 0
    for param in model.parameters():
        l1 += torch.sum(torch.abs(param))
        l2 += torch.sum(param**2)
    return u * l1 + p * l2
# - Добавьте early stopping
# Для early stopping определим тренеровачный набор данных и будем отслеживать loss на каждой эпохе.
# Генерируем данные
X, y = make_regression_data(n=200)
# Разделяем данные на тестовую и тренеровачную выборки
X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
# Создаём датасет и даталоадер
train_dataset = RegressionDataset(X_train, y_train)
test_dataset = RegressionDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# Создаём модель, функцию потерь и оптимизатор
model = LinearRegression(in_features=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

p, u = 1e-4, 1e-4
patience = 10
best_test_loss = np.inf
epochs_no_improve = 0
# Обучаем модель
epochs = 100
for epoch in range(1, epochs + 1):
    train_total_loss = 0
    for i, (batch_X, batch_y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        reg_loss = regularization(model, p, u)
        total_loss = loss + reg_loss
        
        total_loss.backward()
        optimizer.step()

        train_total_loss += total_loss.item()
    avg_train_loss = train_total_loss / (i + 1)
    model.eval()
    with torch.no_grad():
        test_total_loss = 0
        for j, (test_batch_X, test_batch_y) in enumerate(test_dataloader):
            test_pred = model(test_batch_X)
            test_loss = criterion(test_pred, test_batch_y)
            test_total_loss += test_loss.item()
        avg_test_loss = test_total_loss / (j + 1)
    log_epoch(epoch, avg_train_loss, val_loss=avg_test_loss)
    # early stopping
    if avg_test_loss < best_test_loss - 1e-4:
        best_test_loss = avg_test_loss
        epochs_no_improve = 0
        # Сохраняем модель
        torch.save(model.state_dict(), 'best_linreg_torch.pth')
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping on {epoch} epoch")
        break
# Сохраняем модель
torch.save(model.state_dict(), 'linreg_torch.pth')
# Загружаем модель
new_model = LinearRegression(in_features=1)
new_model.load_state_dict(torch.load('linreg_torch.pth'))
new_model.eval()
# 1.2 Расширение логистической регрессии
# Модифицируйте существующую логистическую регрессию:
# - Добавьте поддержку многоклассовой классификации
# - Реализуйте метрики: precision, recall, F1-score, ROC-AUC
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

# - Добавьте визуализацию confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm_da=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_da, annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок')
    plt.xlabel('Прогноз')
    plt.ylabel('Действительность')
    plt.show()

# Генерируем данные
X, y = make_classification_data(n=200)
# Разделяем данные на тестовую и тренеровачную выборки
X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
# Создаём датасет и даталоадер
train_dataset = RegressionDataset(X_train, y_train)
test_dataset = RegressionDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# Создаём модель, функцию потерь и оптимизатор
model = LogisticRegression(in_features=2)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# Обучаем модель
epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, total_acc = 0, 0
    for i, (batch_X, batch_y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        logits = model(batch_X)
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
        
    if epoch % 10 == 0:
        log_epoch(epoch, avg_loss, acc=avg_acc)

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
    log_epoch(epoch, avg_loss, acc=avg_acc, precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)

    if epoch == epochs:
        y_pred_labels = (all_preds > 0.5).int().cpu().numpy()
        plot_confusion_matrix(all_targets.cpu().numpy(), y_pred_labels)

# Сохраняем модель
torch.save(model.state_dict(), 'logreg_torch.pth')
    
# Загружаем модель
new_model = LogisticRegression(in_features=2)
new_model.load_state_dict(torch.load('logreg_torch.pth'))
new_model.eval() 
