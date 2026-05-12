import os
import numpy as np
import pickle
from utils.data_loader import load_eurosat
from models.mlp import ThreeLayerMLP

def create_mini_batches(X, y, batch_size, augment=True):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, X.shape[0])
        excerpt = indices[start_idx:end_idx]
        X_batch = X[excerpt].copy()
        y_batch = y[excerpt].copy()
        
        # 🚀 提速方案：使用向量化掩码极速翻转
        if augment:
            flip_mask = np.random.rand(len(X_batch)) > 0.5
            if np.any(flip_mask):
                imgs = X_batch[flip_mask].reshape(-1, 64, 64, 3)
                X_batch[flip_mask] = imgs[:, :, ::-1, :].reshape(-1, 12288)
        yield X_batch, y_batch

def clip_gradients(model, max_norm=5.0):
    """全局梯度裁剪，防止梯度爆炸"""
    total_norm = 0.0
    for layer in model.weight_layers:
        total_norm += np.sum(layer.dW ** 2)
        total_norm += np.sum(layer.db ** 2)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for layer in model.weight_layers:
            layer.dW *= scale
            layer.db *= scale

def train_model(model, dataset, epochs=200, batch_size=128, lr=0.01, lr_decay=0.5, step_size=40, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    X_train, y_train = dataset['train']
    X_val, y_val = dataset['val']
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    patience = 0
    
    print(f"🚀 开始训练... Epochs: {epochs}, Batch: {batch_size}, 初始LR: {lr}, 激活: {model.activation_name}")
    
    for epoch in range(epochs):
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in create_mini_batches(X_train, y_train, batch_size, augment=True):
            logits = model.forward(X_batch, training=True)
            loss = model.compute_loss(logits, y_batch)
            train_losses.append(loss)
            
            preds = np.argmax(logits, axis=1)
            train_correct += np.sum(preds == y_batch)
            train_total += len(y_batch)
            
            model.backward()
            clip_gradients(model, max_norm=5.0)   # 梯度裁剪
            model.step(lr)
            
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_correct / train_total
        
        val_logits = model.forward(X_val, training=False)
        epoch_val_loss = model.compute_loss(val_logits, y_val)
        val_preds = np.argmax(val_logits, axis=1)
        epoch_val_acc = np.mean(val_preds == y_val)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:03d}/{epochs} | LR: {lr:.5f} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience = 0
            checkpoint = {
                'weights': {
                    'W1': model.fc1.W.copy(), 'b1': model.fc1.b.copy(),
                    'W2': model.fc2.W.copy(), 'b2': model.fc2.b.copy(),
                    'W3': model.fc3.W.copy(), 'b3': model.fc3.b.copy()
                },
                'config': {
                    'hidden_dim': model.fc1.W.shape[1],
                    'activation': model.activation_name,
                    'dropout_p': model.dropout_p,
                    'l2_reg': model.l2_reg
                }
            }
            save_path = os.path.join(save_dir, 'best_model.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  🌟 [最优模型已保存] 验证集准确率提升至: {best_val_acc:.4f}")
        else:
            patience += 1
            
        if patience >= 30:
            print(f"  🛑 连续30轮验证集无提升，触发早停。")
            break
            
        if (epoch + 1) % step_size == 0:
            lr *= lr_decay
            print(f"  📉 [学习率阶梯衰减] 学习率调整为: {lr:.6f}")
    
    with open(os.path.join(save_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    return history

if __name__ == "__main__":
    dataset = load_eurosat()
    # 作为基准单次训练的入口
    model = ThreeLayerMLP(input_dim=12288, hidden_dim=512, num_classes=10, 
                          l2_reg=0.01, activation='tanh', dropout_p=0.5)
    train_model(model, dataset, epochs=200, batch_size=128, lr=0.05, lr_decay=0.5, step_size=40)