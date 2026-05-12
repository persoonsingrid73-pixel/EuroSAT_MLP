import os
import pickle
import numpy as np
from models.mlp import ThreeLayerMLP
from utils.data_loader import load_eurosat
from utils.visualization import plot_history, plot_confusion_matrix_heatmap, visualize_weights

def test_model(weights_path='checkpoints/compare_relu_drop0.3/best_model.pkl'):
    # 尝试加载全局最优，没有则降级加载基础训练的最优
    if not os.path.exists(weights_path):
        weights_path = 'checkpoints/best_model.pkl'
    if not os.path.exists(weights_path):
        print(f"❌ 找不到权重文件 {weights_path}，请先运行训练脚本！")
        return

    print("🚀 加载数据集...")
    dataset = load_eurosat(normalize='zscore') 
    X_test, y_test = dataset['test']
    classes = dataset['classes']

    with open(weights_path, 'rb') as f:
        checkpoint = pickle.load(f)

    config = checkpoint['config']
    weights = checkpoint['weights']
    print(f"📦 加载的模型配置: {config}")

    model = ThreeLayerMLP(
        input_dim=12288,
        hidden_dim=config['hidden_dim'],
        num_classes=10,
        l2_reg=config['l2_reg'],
        activation=config['activation'],
        dropout_p=config['dropout_p']
    )
    
    model.fc1.W, model.fc1.b = weights['W1'], weights['b1']
    model.fc2.W, model.fc2.b = weights['W2'], weights['b2']
    model.fc3.W, model.fc3.b = weights['W3'], weights['b3']
    print("✅ 最优模型权重加载成功！")

    # 🌟 核心保护：同步 ReLU 数据偏移
    if config.get('activation') == 'relu':
        print("🔧 检测到 ReLU 模型，对测试集应用 +0.5 偏移以对齐训练分布")
        X_test = X_test + 0.5

    print("🧠 推理中...")
    test_logits = model.forward(X_test, training=False)
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = np.mean(test_preds == y_test)

    print("\n" + "="*50)
    print(f"🏆 独立测试集分类准确率 (Test Accuracy): {test_acc * 100:.2f}%")
    print("="*50)

    print("🎨 生成图表素材中...")
    os.makedirs('results', exist_ok=True)

    plot_confusion_matrix_heatmap(y_test, test_preds, classes, save_dir='results')
    visualize_weights(model.fc1.W, save_dir='results', num_neurons=16)

    # 寻找并绘制训练历史曲线
    history_path = os.path.join(os.path.dirname(weights_path), 'history.pkl')
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        plot_history(history, save_dir='results')
    else:
        print("⚠️ 未找到训练历史文件，跳过曲线绘制。")

    # 简易错例分析提取
    error_indices = np.where(test_preds != y_test)[0]
    if len(error_indices) > 0:
        print(f"\n📸 测试集中共有 {len(error_indices)} 个错误样本 (抽取前 5 个用于报告分析):")
        for idx in error_indices[:5]:
            true_class = classes[y_test[idx]]
            pred_class = classes[test_preds[idx]]
            print(f"   --> 样本索引 {idx:04d}: 真实类别 [{true_class}], 被错误预测为 [{pred_class}]")

    print("\n🎉 所有测试评估完毕！请前往 'results' 文件夹获取你的实验报告截图。")

if __name__ == "__main__":
    test_model()