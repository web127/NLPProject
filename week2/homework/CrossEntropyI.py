import torch
import torch.nn as nn
import numpy as np

'''
手动实现交叉熵的计算
'''

#使用torch计算交叉熵
ce_loss = nn.CrossEntropyLoss()
#假设有3个样本，每个都在做3分类
pred = torch.FloatTensor([[0.3, 2.1, 0.3],
                          [0.19, 1.2, 0.9],
                          [0.5, 0.4, 0.2]]) #n*class_num
#正确的类别分别为1,2,0
target = torch.LongTensor([1,2,0])     #n
'''
也可以写成这种形式：
torch.LongTensor([
    [0,1,0],
    [0,0,1],
    [1,0,0],
])
'''
loss = ce_loss(pred, target)
print(loss, "torch输出交叉熵")


#实现softmax函数
def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

#验证softmax函数
# print(torch.softmax(pred, dim=1))
# print(softmax(pred.numpy()))


#将输入转化为onehot矩阵
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target

#手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred) # 对预测值进行softmax处理，转成0-1之间的概率
    print(pred)
    target = to_one_hot(target, pred.shape)
    entropy = - np.sum(target * np.log(pred), axis=1) # 对每个样本的预测值进行交叉熵计算并求和除平均，理想情况下，每个样本的预测值都为1，其他接近0，则交叉熵接近1
    print(entropy)
    return sum(entropy) / batch_size

print(cross_entropy(pred.numpy(), target.numpy()), "手动实现交叉熵")

# print(np.log(2.7))

'''
[[0.12422905 0.75154185 0.12422905]
 [0.17302258 0.475051   0.35192642]
 [0.37797815 0.3420088  0.2800131 ]]
 
   [0,1,0],
    [0,0,1],
    [1,0,0],
    
    0*log(0.12422905),1*log(0.75154185),0*log(0.12422905),
    0*log(0.17302258),0*log(0.475051),1*log(0.35192642),
    1*log(0.37797815),0*log(0.3420088),1*log(0.2800131)

'''