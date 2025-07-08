# Задание 3: Кастомные слои и эксперименты (30 баллов)

import torch
import time
from utils.training_utils import train_model
from utils.datasets_utils import get_cifar_loaders
from utils.comparsion_utils import count_parameters
from utils.visualization_utils import plot_training_history
from models.cnn_models import CNNWithResidual
from models.custom_layers import CNNWithCustomConv, CNNWithAttention, CNNWithCustomActivation, CNNWithCustomPooling, CNNWithBottleneckResidual, CNNWithWideResidual

# 3.1 Реализация кастомных слоев (15 баллов)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_cifar_loaders(batch_size=64)

# Реализуйте кастомные слои:
# - Кастомный сверточный слой с дополнительной логикой

cnn_custom_layers = CNNWithCustomConv().to(device)
print(f"CNN with custom layers parameters: {count_parameters(cnn_custom_layers)}")

# - Attention механизм для CNN

cnn_with_attention = CNNWithAttention().to(device)
print(f"CNN with attention parameters: {count_parameters(cnn_with_attention)}")

# - Кастомная функция активации

cnn_custom_activation = CNNWithCustomActivation().to(device)
print(f"CNN with custom activation parameters: {count_parameters(cnn_with_attention)}")

# - Кастомный pooling слой

cnn_custom_pooling = CNNWithCustomPooling().to(device)
print(f"CNN with custom pooling parameters: {count_parameters(cnn_custom_pooling)}")
 
# Для каждого слоя:
# - Реализуйте forward и backward проходы
# - Добавьте параметры если необходимо
# - Протестируйте на простых примерах
# - Сравните с стандартными аналогами

print("Training CNN with custom layers...")

time_start = time.time()
cnn_custom_layers_history = train_model(cnn_custom_layers, train_loader, test_loader, epochs=15, device=str(device))
cnn_custom_layers_time = time.time() - time_start
plot_training_history(cnn_custom_layers_history)

print(f"CNN with custom layers losses: train: {cnn_custom_layers_history['train_losses']}, test: {cnn_custom_layers_history['test_losses']}.")
print(f"CNN with custom layers accuracy: train: {cnn_custom_layers_history['train_accs']}, test: {cnn_custom_layers_history['test_accs']}.")
print(f"CNN with custom layers train time: {cnn_custom_layers_time}")

# Сохраняем модель
torch.save(cnn_custom_layers.state_dict(), 'cnn_custom_layers.pth')

print("Training CNN with attention...")

time_start = time.time()
cnn_with_attention_history = train_model(cnn_with_attention, train_loader, test_loader, epochs=15, device=str(device))
cnn_with_attention_time = time.time() - time_start
plot_training_history(cnn_with_attention_history)

print(f"CNN with attention losses: train: {cnn_with_attention_history['train_losses']}, test: {cnn_with_attention_history['test_losses']}.")
print(f"CNN with attention accuracy: train: {cnn_with_attention_history['train_accs']}, test: {cnn_with_attention_history['test_accs']}.")
print(f"CNN with attention train time: {cnn_with_attention_time}")

# Сохраняем модель
torch.save(cnn_with_attention.state_dict(), 'cnn_with_attention.pth')

print("Training CNN with custom activation...")

time_start = time.time()
cnn_custom_activation_history = train_model(cnn_custom_activation, train_loader, test_loader, epochs=15, device=str(device))
cnn_custom_activation_time = time.time() - time_start
plot_training_history(cnn_custom_activation_history)

print(f"CNN with custom activation losses: train: {cnn_custom_activation_history['train_losses']}, test: {cnn_custom_activation_history['test_losses']}.")
print(f"CNN with custom activation accuracy: train: {cnn_custom_activation_history['train_accs']}, test: {cnn_custom_activation_history['test_accs']}.")
print(f"CNN with custom activation train time: {cnn_custom_activation_time}")

# Сохраняем модель
torch.save(cnn_custom_activation.state_dict(), 'cnn_custom_activation.pth')

print("Training CNN with custom pooling...")

time_start = time.time()
cnn_custom_pooling_history = train_model(cnn_custom_pooling, train_loader, test_loader, epochs=15, device=str(device))
cnn_custom_pooling_time = time.time() - time_start
plot_training_history(cnn_custom_pooling_history)

print(f"CNN with custom pooling losses: train: {cnn_custom_pooling_history['train_losses']}, test: {cnn_custom_pooling_history['test_losses']}.")
print(f"CNN with custom pooling accuracy: train: {cnn_custom_pooling_history['train_accs']}, test: {cnn_custom_pooling_history['test_accs']}.")
print(f"CNN with custom pooling train time: {cnn_custom_pooling_time}")

# Сохраняем модель
torch.save(cnn_custom_pooling.state_dict(), 'cnn_custom_pooling.pth')

# 3.2 Эксперименты с Residual блоками (15 баллов)
# Исследуйте различные варианты Residual блоков:

# - Базовый Residual блок
cnn_base_residual = CNNWithResidual(input_channels=3, num_classes=10).to(device)
print(f"CNN with base Residual parameters: {count_parameters(cnn_base_residual)}")

# - Bottleneck Residual блок
cnn_bottleneck_residual = CNNWithBottleneckResidual().to(device)
print(f"CNN with Bottleneck Residual parameters: {count_parameters(cnn_bottleneck_residual)}")

# - Wide Residual блок
cnn_wide_residual = CNNWithWideResidual(depth=16, width_factor=4, dropout_rate=0.2).to(device)
print(f"CNN with Wide Residual parameters: {count_parameters(cnn_wide_residual)}")

# Для каждого варианта:
# - Реализуйте блок с нуля
# - Сравните производительность
# - Проанализируйте количество параметров
# - Исследуйте стабильность обучения

print("Training CNN with base Residual...")

time_start = time.time()
cnn_base_residual_history = train_model(cnn_base_residual, train_loader, test_loader, epochs=15, device=str(device))
cnn_base_residual_time = time.time() - time_start
plot_training_history(cnn_base_residual_history)

print(f"CNN with base Residual losses: train: {cnn_base_residual_history['train_losses']}, test: {cnn_base_residual_history['test_losses']}.")
print(f"CNN with base Residual accuracy: train: {cnn_base_residual_history['train_accs']}, test: {cnn_base_residual_history['test_accs']}.")
print(f"CNN with base Residual train time: {cnn_base_residual_time}")

# Сохраняем модель
torch.save(cnn_base_residual.state_dict(), 'cnn_base_residual.pth')

print("Training CNN with Bottleneck Residual...")

time_start = time.time()
cnn_bottleneck_residual_history = train_model(cnn_bottleneck_residual, train_loader, test_loader, epochs=10, device=str(device))
cnn_bottleneck_residual_time = time.time() - time_start
plot_training_history(cnn_bottleneck_residual_history)

print(f"CNN with Bottleneck Residual losses: train: {cnn_bottleneck_residual_history['train_losses']}, test: {cnn_bottleneck_residual_history['test_losses']}.")
print(f"CNN with Bottleneck Residual accuracy: train: {cnn_bottleneck_residual_history['train_accs']}, test: {cnn_bottleneck_residual_history['test_accs']}.")
print(f"CNN with Bottleneck Residual train time: {cnn_bottleneck_residual_time}")

# Сохраняем модель
torch.save(cnn_bottleneck_residual.state_dict(), 'cnn_bottleneck_residual.pth')

print("Training CNN with Wide Residual...")

time_start = time.time()
cnn_wide_residual_history = train_model(cnn_wide_residual, train_loader, test_loader, epochs=10, device=str(device))
cnn_wide_residual_time = time.time() - time_start
plot_training_history(cnn_wide_residual_history)

print(f"CNN with Wide Residual losses: train: {cnn_wide_residual_history['train_losses']}, test: {cnn_wide_residual_history['test_losses']}.")
print(f"CNN with Wide Residual accuracy: train: {cnn_wide_residual_history['train_accs']}, test: {cnn_wide_residual_history['test_accs']}.")
print(f"CNN with Wide Residual train time: {cnn_wide_residual_time}")

# Сохраняем модель
torch.save(cnn_wide_residual.state_dict(), 'cnn_wide_residual.pth')
