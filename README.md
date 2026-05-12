# EuroSAT_MLP
运行环境与依赖说明：
核心代码不依赖任何自动微分框架，仅需以下基础 Python 库即可运行：
numpy, pillow, matplotlib, seaborn, scikit-learn

运行指南：

模型训练： 在终端运行 python final_relu_all_v2.py，程序将自动加载数据并开始迭代，训练过程中的权重会自动落盘。

独立测试与出图： 运行 python test.py，程序将加载最优权重进行推理，并在 results 目录下生成 Loss/Acc 曲线、混淆矩阵以及特征可视化图表。
