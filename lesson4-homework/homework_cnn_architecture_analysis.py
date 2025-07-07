# Задание 2: Анализ архитектур CNN (30 баллов)

import torch
import time
from utils.training_utils import train_model
from cnn_models import CNNWithKernelSize
from utils.datasets_utils import get_cifar_loaders
from utils.comparsion_utils import count_parameters
from utils.visualization_utils import plot_training_history, plot_confusion_matrix, plot_first_layer_activations

# 2.1 Влияние размера ядра свертки (15 баллов)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_cifar_loaders(batch_size=64)

# Исследуйте влияние размера ядра свертки:
# - 3x3 ядра
cnn_model_3x3 = CNNWithKernelSize(kernel_size=3).to(device)
print(f"CNN model with kernel_size=3x3 parameters: {count_parameters(cnn_model_3x3)}")

# - 5x5 ядра
cnn_model_5x5 = CNNWithKernelSize(kernel_size=5).to(device)
print(f"CNN model with kernel_size=5x5 parameters: {count_parameters(cnn_model_5x5)}")

# - 7x7 ядра
cnn_model_7x7 = CNNWithKernelSize(kernel_size=7).to(device)
print(f"CNN model with kernel_size=7x7 parameters: {count_parameters(cnn_model_7x7)}")

# - Комбинация разных размеров (1x1 + 3x3)
cnn_model_1x13x3 = CNNWithKernelSize(kernel_size=3, combination=True).to(device)
print(f"CNN model with kernel_size=(1x1+3x3) parameters: {count_parameters(cnn_model_1x13x3)}")

# Для каждого варианта:
# - Поддерживайте одинаковое количество параметров
# - Сравните точность и время обучения
# - Проанализируйте рецептивные поля
# - Визуализируйте активации первого слоя

print("Training CNN model with kernel_size=3x3...")

time_start = time.time()
model_3x3_history = train_model(cnn_model_3x3, train_loader, test_loader, epochs=15, device=str(device))
model_3x3_time = time.time() - time_start
plot_training_history(model_3x3_history)

print(f"CNN model with kernel_size=3x3 losses: train: {model_3x3_history['train_losses']}, test: {model_3x3_history['test_losses']}.")
print(f"CNN model with kernel_size=3x3 accuracy: train: {model_3x3_history['train_accs']}, test: {model_3x3_history['test_accs']}.")
print(f"CNN model with kernel_size=3x3 train time: {model_3x3_time}")

# Сохраняем модель
torch.save(cnn_model_3x3.state_dict(), 'cnn_model_3x3.pth')

print("Training CNN model with kernel_size=5x5...")

time_start = time.time()
model_5x5_history = train_model(cnn_model_5x5, train_loader, test_loader, epochs=15, device=str(device))
model_5x5_time = time.time() - time_start
plot_training_history(model_5x5_history)

print(f"CNN model with kernel_size=5x5 losses: train: {model_5x5_history['train_losses']}, test: {model_5x5_history['test_losses']}.")
print(f"CNN model with kernel_size=5x5 accuracy: train: {model_5x5_history['train_accs']}, test: {model_5x5_history['test_accs']}.")
print(f"CNN model with kernel_size=5x5 train time: {model_5x5_time}")

# Сохраняем модель
torch.save(cnn_model_5x5.state_dict(), 'cnn_model_5x5.pth')

print("Training CNN model with kernel_size=7x7...")

time_start = time.time()
model_7x7_history = train_model(cnn_model_7x7, train_loader, test_loader, epochs=15, device=str(device))
model_7x7_time = time.time() - time_start
plot_training_history(model_7x7_history)

print(f"CNN model with kernel_size=7x7 losses: train: {model_7x7_history['train_losses']}, test: {model_7x7_history['test_losses']}.")
print(f"CNN model with kernel_size=7x7 accuracy: train: {model_7x7_history['train_accs']}, test: {model_7x7_history['test_accs']}.")
print(f"CNN model with kernel_size=7x7 train time: {model_7x7_time}")

# Сохраняем модель
torch.save(cnn_model_7x7.state_dict(), 'cnn_model_7x7.pth')

print("Training CNN model with kernel_size=(1x1+3x3)...")

time_start = time.time()
model_1x13x3_history = train_model(cnn_model_1x13x3, train_loader, test_loader, epochs=15, device=str(device))
model_1x13x3_time = time.time() - time_start
plot_training_history(model_1x13x3_history)

print(f"CNN model with kernel_size=(1x1+3x3) losses: train: {model_1x13x3_history['train_losses']}, test: {model_1x13x3_history['test_losses']}.")
print(f"CNN model with kernel_size=(1x1+3x3) accuracy: train: {model_1x13x3_history['train_accs']}, test: {model_1x13x3_history['test_accs']}.")
print(f"CNN model with kernel_size=(1x1+3x3) train time: {model_1x13x3_time}")

# Сохраняем модель
torch.save(cnn_model_1x13x3.state_dict(), 'cnn_model_1x13x3.pth')

# Визуализация Confusion Matrix
labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

print(f"Confusion Matrix for CNN model with kernel_size=3x3 on CIFAR-10")
plot_confusion_matrix(cnn_model_3x3, test_loader, labels)

print(f"Confusion Matrix for CNN model with kernel_size=5x5 on CIFAR-10")
plot_confusion_matrix(cnn_model_5x5, test_loader, labels)

print(f"Confusion Matrix for CNN model with kernel_size=7x7 on CIFAR-10")
plot_confusion_matrix(cnn_model_7x7, test_loader, labels)

print(f"Confusion Matrix for CNN model with kernel_size=(1x1+3x3) on CIFAR-10")
plot_confusion_matrix(cnn_model_1x13x3, test_loader, labels)

# Визуализируйте активации первого слоя
print(f"Активации первого слоя для CNN model with kernel_size=3x3 on CIFAR-10")
plot_first_layer_activations(cnn_model_3x3, test_loader, num_images=1)

print(f"Активации первого слоя для CNN model with kernel_size=5x5 on CIFAR-10")
plot_first_layer_activations(cnn_model_5x5, test_loader, num_images=1)

print(f"Активации первого слоя для CNN model with kernel_size=7x7 on CIFAR-10")
plot_first_layer_activations(cnn_model_7x7, test_loader, num_images=1)

print(f"Активации первого слоя для CNN model with kernel_size=(1x1+3x3) on CIFAR-10")
plot_first_layer_activations(cnn_model_1x13x3, test_loader, num_images=1)

# 2.2 Влияние глубины CNN (15 баллов)
# Исследуйте влияние глубины CNN:
# - Неглубокая CNN (2 conv слоя)
# - Средняя CNN (4 conv слоя)
# - Глубокая CNN (6+ conv слоев)
# - CNN с Residual связями

# Для каждого варианта:
# - Сравните точность и время обучения
# - Проанализируйте vanishing/exploding gradients
# - Исследуйте эффективность Residual связей
# - Визуализируйте feature maps
