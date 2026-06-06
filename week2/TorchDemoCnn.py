# coding:utf8

# 解决 OpenMP 库冲突问题
#todo 待优化，辅助考察卷积核和线性网络的关系
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
基于PyTorch框架使用CNN实现找规律任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本
"""


class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        # 一维卷积层：输入通道1，输出通道16，卷积核大小3
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()  # 激活函数
        # 最大池化层：池化核大小2，步长2
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 全连接层：将卷积输出映射到1维（用于二分类）
        self.fc = nn.Linear(16 * 2, 1)  # 经过池化后长度为 5//2 = 2
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 调整输入形状：(batch_size, input_size) -> (batch_size, 1, input_size)
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.conv1(x)  # 卷积: (batch_size, 1, 5) -> (batch_size, 16, 5)
        x = self.relu(x)  # 激活
        x = self.pool(x)  # 池化: (batch_size, 16, 5) -> (batch_size, 16, 2)
        x = x.view(x.size(0), -1)  # 展平: (batch_size, 16*2)
        x = self.fc(x)  # 全连接: (batch_size, 32) -> (batch_size, 1)
        y_pred = self.activation(x)  # 归一化到[0,1]
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率

    # 建立CNN模型
    model = CNNModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "cnn_model.bin")

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = CNNModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))


if __name__ == "__main__":
    main()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("cnn_model.bin", test_vec)