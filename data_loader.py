import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_eurosat(data_dir="data/EuroSAT_RGB", test_size=0.2, val_size=0.1, random_state=42, normalize='zscore'):
    X = []
    y = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    print("开始加载并预处理图像数据（64x64）...")
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if not img_name.endswith('.jpg'):
                continue
            img_path = os.path.join(cls_dir, img_name)
            img = Image.open(img_path).resize((64, 64))
            img_array = np.array(img, dtype=np.float32) / 255.0
            X.append(img_array.flatten())
            y.append(class_to_idx[cls_name])

    X = np.array(X)
    y = np.array(y)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio_adjusted = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_state, stratify=y_temp
    )

    if normalize == 'zscore':
        print("进行 Z-Score 标准化 (使用训练集统计量)...")
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    elif normalize == 'minmax':
        print("使用 [0,1] 像素值，不做 Z-Score。")
    else:
        raise ValueError("normalize 只支持 'zscore' 或 'minmax'")

    print(f"训练集数值范围: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"验证集数值范围: [{X_val.min():.3f}, {X_val.max():.3f}]")
    print(f"测试集数值范围: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print("标准化完成！")

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'classes': classes
    }