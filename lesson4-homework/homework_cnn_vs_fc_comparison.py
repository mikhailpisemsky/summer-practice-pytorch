# Задание 1: Сравнение CNN и полносвязных сетей (40 баллов)

import time

# 1.1 Сравнение на MNIST (20 баллов)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_mnist_loaders(batch_size=64)
# Сравните производительность на MNIST:
# - Полносвязная сеть (3-4 слоя)
fcn_model = FullyConnectedModel(input_size=28*28).to(device)
print(f"FCN model parameters: {count_parameters(fcn_model)}")
# - Простая CNN (2-3 conv слоя)
simple_cnn = SimpleCNN(input_channels=1, num_classes=10).to(device)
print(f"Simple CNN parameters: {count_parameters(simple_cnn)}")
# - CNN с Residual Block
residual_cnn = CNNWithResidual(input_channels=1, num_classes=10).to(device)
print(f"Residual CNN parameters: {count_parameters(residual_cnn)}")
# Для каждого варианта:
# - Обучите модель с одинаковыми гиперпараметрами
# - Сравните точность на train и test множествах
# - Измерьте время обучения и инференса
# - Визуализируйте кривые обучения
# - Проанализируйте количество параметров

print("Training FCN...")
time_start = time.time()
fcn_history = train_model(fcn_model, train_loader, test_loader, epochs=15, device=str(device))
fcn_time = time.time() - time_start
plot_training_history(fcn_history)
print(f"FCN model losses: train: {fcn_history['train_losses']}, test: {fcn_history['test_losses']}.")
print(f"FCN model accuracy: train: {fcn_history['train_accs']}, test: {fcn_history['test_accs']}.")
print(f"FCN model train time: {fcn_time}")
# Сохраняем модель
torch.save(fcn_model.state_dict(), 'fcn_model.pth')

print("Training Simple CNN...")
time_start = time.time()
simple_history = train_model(simple_cnn, train_loader, test_loader, epochs=15, device=str(device))
simple_cnn_time = time.time() - time_start
plot_training_history(simple_history)
print(f"Simplie CNN model losses: train: {simple_history['train_losses']}, test: {simple_history['test_losses']}.")
print(f"Simplie CNN model accuracy: train: {simple_history['train_accs']}, test: {simple_history['test_accs']}.")
print(f"Simple CNN model train time: {simple_cnn_time}")
# Сохраняем модель
torch.save(simple_cnn.state_dict(), 'simple_cnn.pth')

print("Training Residual CNN...")
time_start = time.time()
residual_history = train_model(residual_cnn, train_loader, test_loader, epochs=15, device=str(device))
residual_cnn_time = time.time() - time_start
plot_training_history(residual_history)
print(f"Residual CNN model losses: train: {residual_history['train_losses']}, test: {residual_history['test_losses']}.")
print(f"Residual CNN model accuracy: train: {residual_history['train_accs']}, test: {residual_history['test_accs']}.")
print(f"Residual CNN model train time: {residual_cnn_time}")
# Сохраняем модель
torch.save(residual_cnn.state_dict(), 'residual_cnn.pth')

compare_models_on_train(fcn_history, 'FC Network', simple_history, 'Simple CNN')
compare_models_on_test(fcn_history, 'FC Network', simple_history, 'Simple CNN')

compare_models_on_train(fcn_history, 'FC Network', residual_history, 'Residual CNN')
compare_models_on_test(fcn_history, 'FC Network', residual_history, 'Residual CNN')

# 1.2 Сравнение на CIFAR-10 (20 баллов)
# Сравните производительность на CIFAR-10:
# - Полносвязная сеть (глубокая)
# - CNN с Residual блоками
# - CNN с регуляризацией и Residual блоками
# Для каждого варианта:
# - Обучите модель с одинаковыми гиперпараметрами
# - Сравните точность и время обучения
# - Проанализируйте переобучение
# - Визуализируйте confusion matrix
# - Исследуйте градиенты (gradient flow)
