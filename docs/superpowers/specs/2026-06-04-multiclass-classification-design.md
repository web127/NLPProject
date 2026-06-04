# 多分类任务训练设计文档

**日期**: 2026-06-04
**项目**: NLPProject week2 作业

## 概述

完成一个多分类任务的训练，任务规律：一个随机向量，哪一维数字最大就属于第几类。

## 任务规格

- **输入**: 5 维随机向量，值在 [0, 1) 之间
- **输出**: 5 分类，类别 0-4
- **标签规则**: y = argmax(x)，即向量最大值所在的维度索引

## 实现方案

本项目将提供两个版本的实现：

### 版本一：纯 PyTorch 实现（MultiClassTrain.py）

**技术选择**: 纯 PyTorch 实现
**参考**: 基于 TorchDemo.py 的代码结构

### 版本二：手写模拟实现（MultiClassDiy.py）

**技术选择**: numpy 手动实现前向传播和反向传播
**参考**: 基于 GradientDescent.py 和 DNNforward.py 的教学风格

---

## 版本一：纯 PyTorch 实现详细设计

### 组件设计（PyTorch 版本）

#### 1. 数据生成模块

```python
def build_sample():
    """生成单个训练样本"""
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

#### 2. 模型结构

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
        return x  # 输出 logits
```

**结构说明**:
- 输入层: 5 维
- 隐藏层: 10 维，ReLU 激活
- 输出层: 5 维（对应 5 个类别的 logits）

#### 3. 损失函数与优化器

- **损失函数**: `nn.CrossEntropyLoss()`
  - 内部自动处理 Softmax
  - 接受原始 logits 和类别索引（不需要 one-hot 编码）
- **优化器**: `torch.optim.Adam(model.parameters(), lr=0.01)`

#### 4. 评估函数

```python
def evaluate(model, test_data):
    """评估模型准确率"""
    correct = 0
    for x, y in test_data:
        output = model(torch.FloatTensor(x))
        predict = torch.argmax(output).item()
        if predict == y:
            correct += 1
    return correct / len(test_data)
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 训练样本数 | 5000 |
| 测试样本数 | 500 |
| Batch size | 20 |
| Epochs | 20 |
| 学习率 | 0.01 |

### 训练流程

1. 生成训练集和测试集
2. 初始化模型、损失函数、优化器
3. 每个 epoch:
   - 按 batch 遍历训练数据
   - 前向传播计算输出
   - 计算 loss
   - 反向传播
   - 更新参数
4. 记录每个 epoch 的 loss 和 accuracy
5. 绘制训练曲线
6. 保存模型到 `multiclass_model.bin`

### 可视化输出

- Loss 曲线: 每个 epoch 的平均 loss
- Accuracy 曲线: 训练集和测试集上的准确率变化

### 文件结构

```
week2/
├── homework/
│   └── job.desc
├── MultiClassTrain.py   # 新建：PyTorch 版本
├── multiclass_model.bin # 训练后生成：PyTorch 模型权重
└── [现有文件...]
```

## 验收标准（PyTorch 版本）

- 模型在测试集上的准确率应接近 100%（这个任务规律比较简单）
- 训练曲线显示 loss 下降，accuracy 上升
- 模型能够保存和加载
- 加载后的模型能够正确预测新样本

---

## 版本二：手写模拟实现详细设计

### 设计目标

使用纯 numpy 手动实现神经网络的前向传播、损失函数、反向传播和梯度下降，不依赖 PyTorch 的自动微分功能。

### 组件设计（手写版本）

#### 1. 数据生成模块

与 PyTorch 版本相同。

#### 2. 手动模型结构

```python
class DiyMultiClassModel:
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=5):
        # 随机初始化权重
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        """前向传播，返回中间结果用于反向传播"""
        # 第一层
        self.z1 = x @ self.W1 + self.b1  # (batch, hidden)
        self.a1 = np.maximum(0, self.z1)  # ReLU, (batch, hidden)
        # 第二层
        self.z2 = self.a1 @ self.W2 + self.b2  # (batch, output)
        return self.z2  # logits
```

#### 3. Softmax 和交叉熵损失（手动实现）

```python
def softmax(z):
    """Softmax 函数，防止数值溢出"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(logits, y_true):
    """
    计算交叉熵损失
    logits: (batch, 5)
    y_true: (batch,) 类别索引
    """
    m = len(y_true)
    # Softmax
    a = softmax(logits)
    # One-hot 编码
    y_one_hot = np.zeros_like(a)
    y_one_hot[np.arange(m), y_true] = 1
    # 交叉熵
    log_likelihood = -np.log(a[np.arange(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss, a, y_one_hot
```

#### 4. 反向传播（手动计算梯度）

```python
def backward(model, x, a2, y_one_hot, a1):
    """
    手动反向传播计算梯度
    返回 dW1, db1, dW2, db2
    """
    m = len(x)

    # 输出层梯度
    dz2 = a2 - y_one_hot  # (batch, 5)
    dW2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0) / m

    # 隐藏层梯度
    dz1 = (dz2 @ model.W2.T) * (a1 > 0)  # ReLU 导数
    dW1 = (x.T @ dz1) / m
    db1 = np.sum(dz1, axis=0) / m

    return dW1, db1, dW2, db2
```

#### 5. 梯度下降更新

```python
def update_weights(model, dW1, db1, dW2, db2, lr):
    """使用梯度下降更新权重"""
    model.W1 -= lr * dW1
    model.b1 -= lr * db1
    model.W2 -= lr * dW2
    model.b2 -= lr * db2
```

#### 6. 评估函数

```python
def evaluate_diy(model, test_data):
    """评估手动模型的准确率"""
    correct = 0
    for x, y in test_data:
        logits = model.forward(x.reshape(1, -1))
        predict = np.argmax(logits)
        if predict == y:
            correct += 1
    return correct / len(test_data)
```

### 训练配置（手写版本）

| 参数 | 值 |
|------|-----|
| 训练样本数 | 5000 |
| 测试样本数 | 500 |
| Batch size | 20 |
| Epochs | 50 |  # 手写版本可能需要更多轮
| 学习率 | 0.1 |  # 使用 SGD，学习率稍大

### 训练流程（手写版本）

1. 生成训练集和测试集
2. 初始化 DiyMultiClassModel
3. 每个 epoch:
   - 按 batch 遍历训练数据
   - 前向传播: model.forward()
   - 计算 loss 和 softmax 输出
   - 反向传播计算梯度
   - 梯度下降更新权重
4. 记录每个 epoch 的 loss 和 accuracy
5. 绘制训练曲线
6. 保存模型参数（可以保存为 npz 格式）

### 文件结构（最终）

```
week2/
├── homework/
│   └── job.desc
├── MultiClassTrain.py   # 新建：PyTorch 版本
├── MultiClassDiy.py     # 新建：手写版本
├── multiclass_model.bin # 训练后生成：PyTorch 模型权重
├── multiclass_diy.npz # 训练后生成：手写版本参数
└── [现有文件...]
```

## 验收标准（两个版本）

### PyTorch 版本
- 模型在测试集上的准确率应接近 100%
- 训练曲线显示 loss 下降，accuracy 上升
- 模型能够保存和加载
- 加载后的模型能够正确预测新样本

### 手写版本
- 模型在测试集上的准确率应接近 100%
- 训练曲线显示 loss 下降，accuracy 上升
- 手动实现的 Softmax、交叉熵损失、反向传播计算正确
- 可以与 PyTorch 版本的结果对比验证

### 共同点
两个版本共享相同的数据生成逻辑和任务目标，便于对比学习 PyTorch 内部工作原理。
