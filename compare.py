import os
import pickle
import numpy as np
from utils.data_loader import load_eurosat
from models.mlp import ThreeLayerMLP
from train import train_model

def main():
    print("🚀 启动终极竞技场：ReLU vs Tanh 交叉 Dropout 对比实验")
    
    # 1. 加载基础的 Z-Score 数据
    dataset_base = load_eurosat(normalize='zscore')
    X_train_base, y_train_base = dataset_base['train']
    X_val_base, y_val_base = dataset_base['val']

    # 2. 定义竞技场规则
    ACTIVATIONS = ['tanh', 'relu']
    DROPOUTS = [0.2, 0.3, 0.4]
    EPOCHS = 200
    BATCH_SIZE = 128
    LR = 0.01          # 统一使用 0.01 作为公平的初始学习率
    STEP_SIZE = 40
    
    best_overall_acc = 0.0
    best_combo = None
    best_save_dir = None
    results_summary = []

    # 3. 开始 2x3 网格交叉对决
    for act in ACTIVATIONS:
        for dp in DROPOUTS:
            name = f'{act}_drop{dp}'
            save_dir = f'checkpoints/compare_{name}'

            print(f"\n{'='*65}")
            print(f"🔬 [实验组] 激活函数 = {act.upper():<5} | Dropout = {dp}")
            print(f"{'='*65}")

            # 🌟 核心：动态数据分配法则
            if act == 'relu':
                print("🔧 [数据喂养] ReLU: 应用 +0.5 偏移 (拯救负数特征)")
                curr_dataset = {
                    'train': (X_train_base + 0.5, y_train_base),
                    'val': (X_val_base + 0.5, y_val_base)
                }
            else:
                print("⚖️ [数据喂养] Tanh: 保持纯 Z-Score (完美零中心化)")
                curr_dataset = {
                    'train': (X_train_base, y_train_base),
                    'val': (X_val_base, y_val_base)
                }

            # 实例化当前组合的模型
            model = ThreeLayerMLP(
                input_dim=12288,
                hidden_dim=512,
                num_classes=10,
                l2_reg=0.001,
                activation=act,
                dropout_p=dp
            )

            # 开启训练
            history = train_model(
                model, curr_dataset,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                lr_decay=0.5,
                step_size=STEP_SIZE,
                save_dir=save_dir
            )

            # 记录战绩
            val_acc = max(history['val_acc'])
            results_summary.append({
                'act': act.upper(),
                'dp': dp,
                'acc': val_acc
            })

            # 追踪全场 MVP
            if val_acc > best_overall_acc:
                best_overall_acc = val_acc
                best_combo = (act.upper(), dp)
                best_save_dir = save_dir

    # 4. 打印战绩排行榜 (直接用于实验报告)
    print("\n\n" + "🏆"*25)
    print(f"{' 终极对比实验排行榜 ':^45}")
    print("🏆"*25)
    print(f"{'排名':<5} | {'激活函数':<8} | {'Dropout':<8} | {'验证集最高准确率'}")
    print("-" * 50)
    
    # 按准确率降序排列
    results_summary.sort(key=lambda x: x['acc'], reverse=True)
    
    for rank, res in enumerate(results_summary, 1):
        crown = "👑" if rank == 1 else "  "
        print(f" {rank:<3} {crown}| {res['act']:<8} | p = {res['dp']:<4} | {res['acc']*100:>6.2f}%")
        
    print("\n🎉 竞技场对决结束！")
    print(f"🌟 全场 MVP: 【{best_combo[0]} + Dropout {best_combo[1]}】")
    print(f"💾 最优模型已封存在: {best_save_dir}/best_model.pkl")
    print("👉 下一步：修改 test.py 的 weights_path 指向上述路径，一键生成测试图表！")

if __name__ == '__main__':
    main()