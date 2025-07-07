import torch
import matplotlib.pyplot as plt

def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model

def compare_models_on_train(fc_history, title1, cnn_history, title2):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(fc_history['train_accs'], label=title1, marker='o')
    ax1.plot(cnn_history['train_accs'], label=title2, marker='s')
    ax1.set_title('Train Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(fc_history['train_losses'], label=title1, marker='o')
    ax2.plot(cnn_history['train_losses'], label=title2, marker='s')
    ax2.set_title('Train Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show() 

def compare_models_on_test(fc_history, title1, cnn_history, title2):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(fc_history['test_accs'], label=title1, marker='o')
    ax1.plot(cnn_history['test_accs'], label=title2, marker='s')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(fc_history['test_losses'], label=title1, marker='o')
    ax2.plot(cnn_history['test_losses'], label=title2, marker='s')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show() 
