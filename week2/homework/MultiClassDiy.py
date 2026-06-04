# coding:utf8

# 解决 OpenMP 库冲突问题
"""
手写 numpy 版本完整实现
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，不显示窗口
import matplotlib.pyplot as plt

# 复用数据生成函数
def build_sample():
    """生成单个训练样本：5维向量，标签是最大值的索引"""
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

def build_dataset(num_samples):
    """批量生成数据集"""
    dataset = []
    for _ in range(num_samples):
        x, y = build_sample()
        dataset.append((x, y))
    return dataset

def softmax(z):
    """Softmax 函数，防止数值溢出"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class DiyMultiClassModel:
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=5):
        # 随机初始化权重
        np.random.seed(42)  # 固定随机种子便于调试
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        """前向传播，返回 logits，保存中间结果用于反向传播"""
        # 确保 x 是 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # 第一层
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        # 第二层
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

def evaluate_diy(model, test_data):
    """评估手动模型的准确率"""
    correct = 0
    for x, y in test_data:
        logits = model.forward(x)
        predict = np.argmax(logits)
        if predict == y:
            correct += 1
    return correct / len(test_data)

def cross_entropy_loss(logits, y_true):
    """
    计算交叉熵损失
    logits: (batch, 5)
    y_true: (batch,) 类别索引
    返回: loss, softmax_output, y_one_hot
    """
    m = len(y_true)
    # Softmax
    a = softmax(logits)
    # One-hot 编码
    y_one_hot = np.zeros_like(a)
    y_one_hot[np.arange(m), y_true] = 1
    # 交叉熵
    log_likelihood = -np.log(a[np.arange(m), y_true] + 1e-10)  # 防止 log(0)
    loss = np.sum(log_likelihood) / m
    return loss, a, y_one_hot

def backward(model, x, a2, y_one_hot):
    """
    手动反向传播计算梯度
    返回 dW1, db1, dW2, db2
    """
    m = len(x)
    a1 = model.a1

    # 输出层梯度
    dz2 = a2 - y_one_hot
    dW2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0) / m

    # 隐藏层梯度
    dz1 = (dz2 @ model.W2.T) * (model.z1 > 0)  # ReLU 导数
    dW1 = (x.T @ dz1) / m
    db1 = np.sum(dz1, axis=0) / m

    return dW1, db1, dW2, db2

def update_weights(model, dW1, db1, dW2, db2, lr):
    """使用梯度下降更新权重"""
    model.W1 -= lr * dW1
    model.b1 -= lr * db1
    model.W2 -= lr * dW2
    model.b2 -= lr * db2

def main():
    # 配置
    train_size = 5000
    test_size = 500
    batch_size = 20
    epochs = 50
    lr = 0.1

    # 生成数据
    print("正在生成数据集...")
    train_data = build_dataset(train_size)
    test_data = build_dataset(test_size)

    # 初始化模型
    model = DiyMultiClassModel()

    # 训练记录
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # 训练循环
    print("开始训练...")
    for epoch in range(epochs):
        total_loss = 0
        # 按 batch 训练
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            x_batch = np.array([x for x, y in batch])
            y_batch = np.array([y for x, y in batch])

            # 前向传播
            logits = model.forward(x_batch)
            loss, a2, y_one_hot = cross_entropy_loss(logits, y_batch)

            # 反向传播
            dW1, db1, dW2, db2 = backward(model, x_batch, a2, y_one_hot)

            # 更新权重
            update_weights(model, dW1, db1, dW2, db2, lr)

            total_loss += loss

        # 记录指标
        avg_loss = total_loss / (len(train_data) / batch_size)
        train_acc = evaluate_diy(model, train_data)
        test_acc = evaluate_diy(model, test_data)

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型参数
    np.savez("multiclass_diy.npz", W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    print("模型参数已保存到 multiclass_diy.npz")

    # 可视化
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss (DIY)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train")
    plt.plot(test_accuracies, label="Test")
    plt.title("Accuracy (DIY)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("multiclass_diy_training.png")
    print("训练曲线已保存到 multiclass_diy_training.png")

    # 测试加载模型
    print("\n测试模型加载和预测...")
    loaded_data = np.load("multiclass_diy.npz")
    loaded_model = DiyMultiClassModel()
    loaded_model.W1 = loaded_data["W1"]
    loaded_model.b1 = loaded_data["b1"]
    loaded_model.W2 = loaded_data["W2"]
    loaded_model.b2 = loaded_data["b2"]

    # 测试几个样本
    for _ in range(5):
        x, y = build_sample()
        logits = loaded_model.forward(x)
        pred = np.argmax(logits)
        print(f"x: {np.round(x, 3)}, true: {y}, pred: {pred}, correct: {y == pred}")

if __name__ == "__main__":
    main()
