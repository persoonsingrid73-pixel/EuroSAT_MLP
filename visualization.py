import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_history(history, save_dir='results'):
    """绘制训练 & 验证 Loss 和 Accuracy 曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    filename = os.path.join(save_dir, 'loss_acc.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ 训练曲线已保存至 {filename}")

def plot_confusion_matrix_heatmap(y_true, y_pred, classes, save_dir='results'):
    """绘制归一化的混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    filename = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ 混淆矩阵已保存至 {filename}")

def visualize_weights(W1, save_dir='results', num_neurons=16):
    """
    将第一层权重 (12288, hidden_dim) 重塑为图像并可视化前 num_neurons 个神经元。
    假设数据原始为 HWC 顺序 (64, 64, 3)，权重每列形状也一样。
    """
    input_dim, hidden_dim = W1.shape
    h = w = int(np.sqrt(input_dim / 3))  # 64
    num_features = min(num_neurons, hidden_dim)
    
    # 创建一个 4x4 的子图展示
    rows = int(np.ceil(np.sqrt(num_features)))
    cols = int(np.ceil(num_features / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    axes = axes.flatten()
    
    for i in range(num_features):
        # 取出第 i 个神经元的权重，形状 (12288,)
        weight = W1[:, i]
        # 重塑为 (64, 64, 3)
        img = weight.reshape(h, w, 3)
        # 归一化到 [0,1] 以便显示
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Neuron {i+1}')
    
    # 隐藏多余的子图
    for j in range(num_features, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('First Layer Weight Visualization (Reshaped to 64x64x3)', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(save_dir, 'weight_vis.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ 权重可视化已保存至 {filename}")