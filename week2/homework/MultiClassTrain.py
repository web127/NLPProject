# coding:utf8

# 解决 OpenMP 库冲突问题

"""
基于pytorch框架编写模型训练
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，不显示窗口
import matplotlib.pyplot as plt

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

class MultiClassModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def evaluate(model, test_data):
    """评估模型准确率"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_data:
            output = model(torch.FloatTensor(x))
            predict = torch.argmax(output).item()
            if predict == y:
                correct += 1
    return correct / len(test_data)

def main():
    # 配置
    train_size = 5000
    test_size = 500
    batch_size = 20
    epochs = 20
    lr = 0.01

    # 生成数据
    print("正在生成数据集...")
    train_data = build_dataset(train_size)
    test_data = build_dataset(test_size)

    # 初始化模型
    model = MultiClassModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练记录
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # 训练循环
    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # 按 batch 训练
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            x_batch = torch.FloatTensor([x for x, y in batch])
            y_batch = torch.LongTensor([y for x, y in batch])

            # 前向传播
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 记录指标
        avg_loss = total_loss / (len(train_data) / batch_size)
        train_acc = evaluate(model, train_data)
        test_acc = evaluate(model, test_data)

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "multiclass_model.bin")
    print("模型已保存到 multiclass_model.bin")

    # 可视化
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train")
    plt.plot(test_accuracies, label="Test")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("multiclass_training.png")
    print("训练曲线已保存到 multiclass_training.png")

    # 测试加载模型
    print("\n测试模型加载和预测...")
    loaded_model = MultiClassModel()
    loaded_model.load_state_dict(torch.load("multiclass_model.bin"))
    loaded_model.eval()

    # 测试几个样本
    for _ in range(20):
        x, y = build_sample()
        output = loaded_model(torch.FloatTensor(x))
        pred = torch.argmax(output).item()
        print(f"x: {np.round(x, 3)}, true: {y}, pred: {pred}, correct: {y == pred}")

if __name__ == "__main__":
    main()
