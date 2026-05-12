import os
import pickle
import numpy as np
from utils.data_loader import load_eurosat
from models.mlp import ThreeLayerMLP
from train import train_model

DROPOUT_LIST = [0.2, 0.3, 0.4]
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.01
LR_DECAY = 0.5
STEP_SIZE = 40
SAVE_ROOT = 'checkpoints/final_relu_zscore_offset_drop'

def main():
    dataset = load_eurosat(normalize='zscore')

    # ReLU 偏移
    shift = 0.5
    X_train, y_train = dataset['train']
    X_val, y_val = dataset['val']
    dataset['train'] = (X_train + shift, y_train)
    dataset['val']   = (X_val + shift, y_val)
    print(f"🔧 ReLU 输入偏移 +{shift} 已应用\n")

    best_acc = 0.0
    best_dropout = None
    best_save_dir = None

    for dp in DROPOUT_LIST:
        name = f'drop{dp}'
        save_dir = f'{SAVE_ROOT}_{name}'
        print(f"{'='*60}")
        print(f"🚀 训练 ReLU + dropout={dp}")
        print(f"   数据: Z-score + 0.5, 隐藏层:512, lr={LR}, 每{STEP_SIZE}轮衰减")
        print(f"{'='*60}")

        model = ThreeLayerMLP(
            input_dim=12288,
            hidden_dim=512,
            num_classes=10,
            l2_reg=0.001,
            activation='relu',
            dropout_p=dp
        )

        history = train_model(
            model, dataset,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            lr_decay=LR_DECAY,
            step_size=STEP_SIZE,
            save_dir=save_dir
        )

        with open(os.path.join(save_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)

        val_acc = max(history['val_acc'])
        print(f"✅ dropout={dp} 最佳验证准确率: {val_acc:.4f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            best_dropout = dp
            best_save_dir = save_dir

    print("🎉 全部训练完成！")
    print(f"🏆 最佳 dropout: {best_dropout} (验证准确率: {best_acc:.4f})")
    print(f"   模型路径: {best_save_dir}/best_model.pkl")
    print("   ⚠️ 测试时请用 load_eurosat(normalize='zscore') 并对测试数据 +0.5")

if __name__ == "__main__":
    main()