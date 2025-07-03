import pandas as pd
import torch
import torch.nn as nn
from models.utils import log_epoch, accuracy
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from models.linreg_torch import LinearRegression
from models.logreg_torch import LogisticRegression

# Задание 2: Работа с датасетами
# 2.1 Кастомный Dataset класс
# Создайте кастомный класс датасета для работы с CSV файлами:
class DataFromCSV(Dataset):
    def __init__(self, path, target, categorical_features=None, binary_features=None):
        # - Загрузка данных из файла
        self.data = pd.read_csv(path)
        self.target = target
        self.categorical_features = categorical_features if categorical_features else []
        self.binary_features = binary_features if binary_features else []
        X = self.data.drop(target, axis=1)
        if self.data[target].dtype == 'object':
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            target_encoded = encoder.fit_transform(self.data[[target]])
            self.y = torch.tensor(target_encoded, dtype=torch.float32).squeeze(1)
        else:
            self.y = torch.tensor(self.data[target].values, dtype=torch.float32).unsqueeze(1)
        numeriс_features = [feature for feature in X if feature not in self.categorical_features + self.binary_features]
        self.numeriс_features = numeriс_features if numeriс_features else []
        # - Предобработка (нормализация, кодирование категорий)
        #Кодирование категорий
        if self.categorical_features:
            catcategorical_data = X[self.categorical_features]
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            catcategorical_encoded = encoder.fit_transform(catcategorical_data)
            self.catcategorical_tensor = torch.tensor(catcategorical_encoded, dtype=torch.float32)
        else:
            self.catcategorical_tensor = None

        #Масштабирование числовых переменных        
        if self.numeriс_features:
            numeriс_data = X[self.numeriс_features]
            scaler = StandardScaler()
            numeric_scaled = scaler.fit_transform(numeriс_data)
            self.numeriс_tensor = torch.tensor(numeric_scaled, dtype=torch.float32)
            
        # - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.
        # Работа с бинарными данными
        if self.binary_features:
            binary_data = X[self.binary_features].astype(float)
            self.binary_tensor = torch.tensor(binary_data.values, dtype=torch.float32)
        else:
            self.binary_tensor = None
        
        tensor = [features for features in [self.catcategorical_tensor, self.numeriс_tensor, self.binary_tensor] if features is not None]
        if not tensor:
            raise ValueError("Ошибка сбора данных")
        self.X = torch.cat(tensor, dim=1)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
        
# 2.2 Эксперименты с различными датасетами
# Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию

# Данные для регрессии: https://www.kaggle.com/datasets/abrambeyer/openintro-possum

# Создаём датасет и даталоадер
database_possum = DataFromCSV("data/possum-liner-regression.csv", target="age", categorical_features=["Pop", "sex"])
dataloader = DataLoader(database_possum, batch_size=32, shuffle=True)
print(f'Размер датасета: {len(database_possum)}')
print(f'Количество батчей: {len(dataloader)}')
# Создаём модель, функцию потерь и оптимизатор
model = LinearRegression(in_features=database_possum.X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Обучаем модель
epochs = 100
for epoch in range(1, epochs + 1):
    total_loss = 0
    for i, (batch_X, batch_y) in enumerate(dataloader):
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()            
        total_loss += loss.item()
    avg_loss = total_loss / (i + 1)
    if epoch % 10 == 0:
        log_epoch(epoch, avg_loss)
   
   
# Сохраняем модель
torch.save(model.state_dict(), 'possum_linreg_torch.pth')
    
# Загружаем модель
new_model = LinearRegression(in_features=database_possum.X.shape[1])
new_model.load_state_dict(torch.load('possum_linreg_torch.pth'))
new_model.eval()

# Данные для классификации https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

# Создаём датасет и даталоадер
database_breast = DataFromCSV("data/breast-cancer-binary-classification.csv", target="diagnosis")
dataloader = DataLoader(database_breast, batch_size=32, shuffle=True)
print(f'Размер датасета: {len(database_breast)}')
print(f'Количество батчей: {len(dataloader)}')
# Создаём модель, функцию потерь и оптимизатор
model = LogisticRegression(in_features=database_breast.X.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Обучаем модель
epochs = 100
for epoch in range(1, epochs + 1):
    total_loss = 0
    total_acc = 0

    for i, (batch_X, batch_y) in enumerate(dataloader):
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
    if epoch % 10 == 0:
        log_epoch(epoch, avg_loss, acc=avg_acc)
    
# Сохраняем модель
torch.save(model.state_dict(), 'breast_logreg_torch.pth')
    
# Загружаем модель
new_model = LogisticRegression(in_features=database_breast.X.shape[1])
new_model.load_state_dict(torch.load('breast_logreg_torch.pth'))
new_model.eval()
