# 多分类任务训练实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现两个版本的多分类训练任务：PyTorch 版本和手写 numpy 版本

**Architecture:** 先实现 PyTorch 版本（参考 TorchDemo.py），再实现手写版本（参考 GradientDescent.py），两个版本共享相同的数据生成逻辑

**Tech Stack:** Python, PyTorch, numpy, matplotlib

---

## 文件结构

| 文件 | 责任 |
|------|------|
| `week2/MultiClassTrain.py` | PyTorch 版本完整实现 |
| `week2/MultiClassDiy.py` | 手写 numpy 版本完整实现 |

---

## 任务列表

### Task 1: 实现 PyTorch 版本 - 数据生成模块

**Files:**
- Create: `week2/MultiClassTrain.py`

- [ ] **Step 1: 创建文件并添加数据生成函数**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
```

- [ ] **Step 2: 验证数据生成**

```python
# 在文件末尾添加测试代码
if __name__ == "__main__":
    # 测试数据生成
    x, y = build_sample()
    print(f"测试样本 - x: {x}, y: {y}, argmax: {np.argmax(x)}")
    assert y == np.argmax(x), "数据生成错误"
    print("数据生成测试通过")
```

- [ ] **Step 3: 运行验证**

Run: `cd week2 && python3 MultiClassTrain.py`
Expected: 打印测试样本并显示"数据生成测试通过"

---

### Task 2: 实现 PyTorch 版本 - 模型和评估函数

**Files:**
- Modify: `week2/MultiClassTrain.py`

- [ ] **Step 1: 添加模型类**

在数据生成函数后添加：

```python
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
```

- [ ] **Step 2: 更新测试代码**

替换 `if __name__ == "__main__":` 部分：

```python
if __name__ == "__main__":
    # 测试数据生成
    x, y = build_sample()
    print(f"测试样本 - x: {x}, y: {y}, argmax: {np.argmax(x)}")
    assert y == np.argmax(x), "数据生成错误"
    print("数据生成测试通过")

    # 测试模型创建
    model = MultiClassModel()
    print(f"模型结构: {model}")

    # 测试前向传播
    test_x = torch.FloatTensor(x)
    output = model(test_x)
    print(f"模型输出: {output}")
    print(f"预测类别: {torch.argmax(output).item()}")
    print("模型测试通过")
```

- [ ] **Step 3: 运行验证**

Run: `cd week2 && python3 MultiClassTrain.py`
Expected: 模型创建成功，前向传播正常

---

### Task 3: 实现 PyTorch 版本 - 完整训练循环

**Files:**
- Modify: `week2/MultiClassTrain.py`

- [ ] **Step 1: 添加训练函数**

在 evaluate 函数后添加：

```python
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
    plt.show()

    # 测试加载模型
    print("\n测试模型加载和预测...")
    loaded_model = MultiClassModel()
    loaded_model.load_state_dict(torch.load("multiclass_model.bin"))
    loaded_model.eval()

    # 测试几个样本
    for _ in range(5):
        x, y = build_sample()
        output = loaded_model(torch.FloatTensor(x))
        pred = torch.argmax(output).item()
        print(f"x: {np.round(x, 3)}, true: {y}, pred: {pred}, correct: {y == pred}")
```

- [ ] **Step 2: 更新入口**

替换 `if __name__ == "__main__":` 为：

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行完整训练**

Run: `cd week2 && python3 MultiClassTrain.py`
Expected: 训练完成，测试准确率接近 100%，保存模型和训练曲线

---

### Task 4: 实现手写版本 - 基础组件

**Files:**
- Create: `week2/MultiClassDiy.py`

- [ ] **Step 1: 创建文件并添加数据生成和基础函数**

```python
import numpy as np
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
```

- [ ] **Step 2: 添加测试验证**

```python
if __name__ == "__main__":
    # 测试 softmax
    test_z = np.array([[1, 2, 3, 4, 5]])
    sm = softmax(test_z)
    print(f"Softmax 测试: {sm}")
    print(f"Sum: {np.sum(sm)}")
    assert np.allclose(np.sum(sm), 1.0), "Softmax 求和应该为1"
    print("Softmax 测试通过")
```

- [ ] **Step 3: 运行验证**

Run: `cd week2 && python3 MultiClassDiy.py`
Expected: Softmax 测试通过

---

### Task 5: 实现手写版本 - 模型类

**Files:**
- Modify: `week2/MultiClassDiy.py`

- [ ] **Step 1: 添加 DiyMultiClassModel 类**

在 softmax 函数后添加：

```python
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
```

- [ ] **Step 2: 添加评估函数**

```python
def evaluate_diy(model, test_data):
    """评估手动模型的准确率"""
    correct = 0
    for x, y in test_data:
        logits = model.forward(x)
        predict = np.argmax(logits)
        if predict == y:
            correct += 1
    return correct / len(test_data)
```

- [ ] **Step 3: 更新测试代码**

```python
if __name__ == "__main__":
    # 测试 softmax
    test_z = np.array([[1, 2, 3, 4, 5]])
    sm = softmax(test_z)
    print(f"Softmax 测试: {sm}")
    print(f"Sum: {np.sum(sm)}")
    assert np.allclose(np.sum(sm), 1.0), "Softmax 求和应该为1"
    print("Softmax 测试通过")

    # 测试模型创建
    model = DiyMultiClassModel()
    x, y = build_sample()
    logits = model.forward(x)
    print(f"\n模型前向传播测试:")
    print(f"输入 x: {x}")
    print(f"输出 logits: {logits}")
    print(f"预测: {np.argmax(logits)}")
    print("模型测试通过")
```

- [ ] **Step 4: 运行验证**

Run: `cd week2 && python3 MultiClassDiy.py`
Expected: 所有测试通过

---

### Task 6: 实现手写版本 - 损失和反向传播

**Files:**
- Modify: `week2/MultiClassDiy.py`

- [ ] **Step 1: 添加损失和反向传播函数**

在 evaluate_diy 函数后添加：

```python
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
```

- [ ] **Step 2: 更新测试代码**

添加到 `if __name__ == "__main__":` 末尾：

```python
    # 测试损失和反向传播
    print(f"\n测试损失和反向传播:")
    # 构造一个 batch
    x_batch = np.array([x, x])
    y_batch = np.array([y, y])
    logits = model.forward(x_batch)
    loss, a2, y_one_hot = cross_entropy_loss(logits, y_batch)
    print(f"Loss: {loss}")
    dW1, db1, dW2, db2 = backward(model, x_batch, a2, y_one_hot)
    print(f"梯度计算完成，dW1 shape: {dW1.shape}")
    print("反向传播测试通过")
```

- [ ] **Step 3: 运行验证**

Run: `cd week2 && python3 MultiClassDiy.py`
Expected: 所有测试通过

---

### Task 7: 实现手写版本 - 完整训练循环

**Files:**
- Modify: `week2/MultiClassDiy.py`

- [ ] **Step 1: 添加完整训练函数**

在 update_weights 后添加：

```python
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
    plt.show()

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
```

- [ ] **Step 2: 更新入口**

替换 `if __name__ == "__main__":` 为：

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行完整训练**

Run: `cd week2 && python3 MultiClassDiy.py`
Expected: 训练完成，测试准确率接近 100%

---

## 验收检查

- [ ] 两个文件都创建完成：`MultiClassTrain.py` 和 `MultiClassDiy.py`
- [ ] PyTorch 版本训练成功，测试准确率 > 95%
- [ ] 手写版本训练成功，测试准确率 > 95%
- [ ] 生成了模型文件和训练曲线图
