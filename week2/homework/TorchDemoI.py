import torch
import torch.nn as nn
import numpy as np
from torch import optim
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activate = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x):
        out = self.layer1(x)
        out = self.activate(out)
        out=self.layer2(out)
        return out

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

def evaluate(model,test_data):
    """评估模型准确率"""
    model.eval()
    correct, wrong = 0, 0
    with torch.no_grad():
        for x, y in test_data:
            y_pred = model(torch.FloatTensor([x]))
            predict = torch.argmax(y_pred).item()
            if predict == y:
                correct += 1
            else:
                #print(f"预测错误：{x} -> {predict} (真实标签：{y})")
                wrong += 1

    return correct / (correct + wrong)
def train_model():
    # 训练记录
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    log = []
    epoch_num=50
    batch_size=100
    learning_rate=0.01
    train_sample = 5000  # 每轮训练总共训练的样本总数
    model=TorchModel(input_size=5,hidden_size=10,output_size=5)
    test_size = 500
    lossF = nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)# 选择优化器
    # 创建训练集，正常任务是读取训练集
    train_data = build_dataset(train_sample)

    test_data = build_dataset(test_size)

    for epoch in range(epoch_num):
        model.train() #训练模式下，dropout等层会启
        watch_loss = []
        total_loss = 0
        for i in range(0,len(train_data),batch_size):
           batch = train_data[i:i+batch_size]
           x_batch = torch.FloatTensor([x for x,y in batch])
           y_batch = torch.LongTensor([y for x,y in batch])
           y_pred = model(x_batch)#获取预测值
           loss = lossF(y_pred, y_batch)#计算损失

           optimizer.zero_grad()
           loss.backward()# 计算梯度
           optimizer.step() # 更新权重

           watch_loss.append(loss.item())
           total_loss += loss.item()

        avg_loss = total_loss / (len(train_data) / batch_size)
      #  train_acc = evaluate(model, train_data)
       # test_acc = evaluate(model, test_data)

        train_losses.append(avg_loss)
      #  train_accuracies.append(train_acc)
      #  test_accuracies.append(test_acc)
        print(f"Epoch {epoch+1}/{epoch_num}")
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))


   # acc = evaluate(model)  # 测试本轮模型结果
   # log.append([acc, float(np.mean(watch_loss))])
    # 画图
    #print(log)
   # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    #plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
   # plt.legend()
   # plt.show()
    # 可视化
    # 测试几个样本
    test_acc = evaluate(model, test_data)
    print(f" Test Acc: {test_acc:.4f}")
    for _ in range(200):
        x, y = build_sample()
        output = model(torch.FloatTensor(x))
        pred = torch.argmax(output).item()
        print(f"x: {np.round(x, 3)}, true: {y}, pred: {pred}, correct: {y == pred}")
    return

if __name__ == "__main__":
   train_model()
