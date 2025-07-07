import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_training_history(history):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, test_loader, class_names):
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            if isinstance(model, FullyConnectedModel):
                xb = xb.view(xb.size(0), -1)
            outputs = model(xb)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Прогноз')
    plt.ylabel('Действительность')
    plt.title('Confusion Matrix')
    plt.show()

def plot_first_layer_activations(model, test_loader, num_images=3):
    model.eval()
    device = next(model.parameters()).device
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images.to(device)

    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    if first_conv is None:
        raise ValueError("Не найден сверточный слой для визуализации.")

    with torch.no_grad():
        activations = first_conv(images[:num_images])
    activations = activations.cpu()

    for idx in range(num_images):
        plt.figure(figsize=(12, 2))
        for i in range(min(8, activations.shape[1])):  
            plt.subplot(1, 8, i+1)
            plt.imshow(activations[idx, i].numpy(), cmap='viridis')
            plt.axis('off')
        plt.suptitle(f'Активации для изображения {idx + 1}')
        plt.show()
