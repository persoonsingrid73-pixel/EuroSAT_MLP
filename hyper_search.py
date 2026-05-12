import os
import pickle
import numpy as np
import shutil
import itertools
from utils.data_loader import load_eurosat
from models.mlp import ThreeLayerMLP
from train import create_mini_batches, clip_gradients

# ===================== 快速搜索配置（可调整） =====================
FAST_EPOCHS_GRID = 20        # 阶段三每个组合的轮数
KILL_CHECK = 10               # 前 KILL_CHECK 轮验证准确率从未超过阈值就终止
KILL_THRESHOLD = 0.55         # 终止阈值
PATIENCE = 5                  # 早停耐心
BATCH_SIZE = 256              # 批大小（内存不足可改回128）
GRID_STEP_SIZE = 10           # 阶段三学习率衰减步长

# 阶段三搜索空间（缩小范围）
REMAINING_SPACE = {
    'hidden_dim': [512, 1024],
    'dropout_p': [0.2, 0.3, 0.4],
    'activation': ['tanh']
}
# 总计 2 × 3 × 1 = 6 组

# 已确定的最优学习率和 L2
BEST_LR = 0.01
BEST_L2 = 0.001

# =========================================================

def train_with_early_kill(model, dataset, epochs=50, batch_size=128, lr=0.01,
                          lr_decay=0.5, step_size=20, save_dir=None,
                          kill_threshold=0.55, kill_check_epoch=20,
                          patience=15):
    """
    训练并返回 history, best_val_acc, best_train_loss, best_val_loss。
    如果前 kill_check_epoch 轮验证准确率从未超过 kill_threshold，
    直接终止并返回 0.0。
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    X_train, y_train = dataset['train']
    X_val, y_val = dataset['val']

    # 仅对 relu 做偏移（目前只搜 tanh，但保留逻辑）
    if model.activation_name == 'relu':
        X_train = X_train + 0.5
        X_val = X_val + 0.5

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_train_loss = None
    best_val_loss = None
    patience_counter = 0

    for epoch in range(epochs):
        # 训练一个 epoch
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
            clip_gradients(model, max_norm=5.0)
            model.step(lr)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_correct / train_total

        # 验证
        val_logits = model.forward(X_val, training=False)
        epoch_val_loss = model.compute_loss(val_logits, y_val)
        val_preds = np.argmax(val_logits, axis=1)
        epoch_val_acc = np.mean(val_preds == y_val)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # 每个 epoch 打印进度
        print(f"     Epoch {epoch+1:03d}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}",
              flush=True)

        # 早期淘汰检查
        if epoch == kill_check_epoch - 1:
            if max(history['val_acc']) <= kill_threshold:
                print(f"     ⚠️ 前 {kill_check_epoch} 轮最高验证准确率 "
                      f"{max(history['val_acc']):.4f} ≤ {kill_threshold}，提前终止")
                return history, 0.0, None, None

        # 更新最佳
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_train_loss = epoch_train_loss
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # 保存最优模型
            if save_dir:
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
                with open(os.path.join(save_dir, 'best_model.pkl'), 'wb') as f:
                    pickle.dump(checkpoint, f)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"     🛑 连续 {patience} 轮验证准确率未提升，早停")
                break

        # 学习率衰减
        if (epoch + 1) % step_size == 0:
            lr *= lr_decay
            print(f"     📉 学习率衰减至: {lr:.6f}")

    return history, best_val_acc, best_train_loss, best_val_loss


def grid_search_remaining(dataset, best_lr, best_l2, search_space, epochs, batch_size,
                          kill_threshold, kill_check, patience):
    print("\n🔍 阶段三：网格搜索 hidden_dim × dropout × activation")
    hidden_dims = search_space['hidden_dim']
    dropout_ps = search_space['dropout_p']
    activations = search_space['activation']
    combinations = list(itertools.product(hidden_dims, dropout_ps, activations))
    total = len(combinations)
    print(f"总组合数: {total}")

    results = []
    best_overall_acc = 0.0

    for idx, (hd, dp, act) in enumerate(combinations, 1):
        print(f"\n{'='*50}")
        print(f"网格 {idx}/{total} | Hidden: {hd}, Dropout: {dp}, Act: {act}")
        model = ThreeLayerMLP(input_dim=12288, hidden_dim=hd, num_classes=10,
                              l2_reg=best_l2, activation=act, dropout_p=dp)
        save_dir = f'checkpoints/grid_{idx:02d}'
        _, val_acc, train_loss, val_loss = train_with_early_kill(
            model, dataset, epochs=epochs, batch_size=batch_size,
            lr=best_lr, lr_decay=0.5, step_size=GRID_STEP_SIZE,
            save_dir=save_dir, kill_threshold=kill_threshold,
            kill_check_epoch=kill_check, patience=patience
        )
        if val_acc > 0:
            print(f"   ✅ 最佳验证准确率: {val_acc:.4f} "
                  f"(训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f})")
        else:
            print(f"   ❌ 淘汰")

        results.append({
            'hidden_dim': hd,
            'dropout_p': dp,
            'activation': act,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            src = os.path.join(save_dir, 'best_model.pkl')
            if os.path.exists(src):
                shutil.copy(src, 'checkpoints/best_overall.pkl')
                print(f"   🏆 新高！模型已保存至 best_overall.pkl")

    # 按验证准确率降序，验证损失升序排列
    results.sort(key=lambda x: (
        -x['val_acc'],
        x['val_loss'] if x['val_loss'] is not None else float('inf')
    ))

    # 打印 Top-10
    print("\n" + "="*80)
    print("🏁 网格搜索 Top-10 结果：")
    header = f"{'排名':<4} {'Hidden':<7} {'Drop':<6} {'Act':<6} {'Val Acc':<8} {'Train Loss':<10} {'Val Loss':<10}"
    print(header)
    for i, r in enumerate(results[:10], 1):
        tl = f"{r['train_loss']:.4f}" if r['train_loss'] is not None else "N/A"
        vl = f"{r['val_loss']:.4f}" if r['val_loss'] is not None else "N/A"
        print(f"{i:<4} {r['hidden_dim']:<7} {r['dropout_p']:<6} {r['activation']:<6} "
              f"{r['val_acc']:<8.4f} {tl:<10} {vl:<10}")

    return results


if __name__ == "__main__":
    # 加载数据
    dataset = load_eurosat()

    print("🚀 使用预设最优参数：lr=0.01, l2=0.001")
    print("📋 阶段三网格搜索：")
    print(f"   hidden_dim: {REMAINING_SPACE['hidden_dim']}")
    print(f"   dropout_p:  {REMAINING_SPACE['dropout_p']}")
    print(f"   activation: {REMAINING_SPACE['activation']}")

    # 阶段三：网格搜索剩余参数
    final_results = grid_search_remaining(
        dataset, BEST_LR, BEST_L2, REMAINING_SPACE,
        epochs=FAST_EPOCHS_GRID, batch_size=BATCH_SIZE,
        kill_threshold=KILL_THRESHOLD, kill_check=KILL_CHECK,
        patience=PATIENCE
    )

    # 保存完整结果
    output = {
        'best_lr': BEST_LR,
        'best_l2': BEST_L2,
        'results': final_results
    }
    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/hyper_v2_fast_results.pkl', 'wb') as f:
        pickle.dump(output, f)

    print("\n🎉 快速超参数搜索完成！最优模型已保存于 checkpoints/best_overall.pkl")