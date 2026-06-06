from stringprep import b1_set

import torch.nn as nn
import torch
import numpy as np
class TorchModelI(nn.Module):
    def __init__(self,input_size=3,hidden_size=5,hidden_size2=2):
        # 第一事情，先定义网络结构
        super(TorchModelI, self).__init__()
        self.layer1=nn.Linear(input_size,hidden_size)
        self.layer2=nn.Linear(hidden_size,hidden_size2)

    def forward(self,x):
        x = self.layer1(x) #x的类型是什么
        y_pred=self.layer2(x)
        return y_pred
        #第二件事情：表示这些网络是如何连接的

#自定义模型
class DiyModel:
    def __init__(self,w1,b1,w2,b2):
        self.w1 = w1
        self.b1=b1
        self.w2=w2
        self.b2=b2

    def forward(self,x):
        hidden =np.dot(x,self.w1.T)+self.b1
        y_pred=np.dot(hidden,self.w2.T)+self.b2
        return y_pred

x =np.array([[3.1, 1.3, 1.2],
              [2.1, 1.3, 13]])

torch_model = TorchModelI(3,5,2)
print(torch_model.state_dict())

print("-----------")
#打印模型权重，权重为随机初始化
torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy()
torch_model_b1 = torch_model.state_dict()["layer1.bias"].numpy()
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
torch_model_b2 = torch_model.state_dict()["layer2.bias"].numpy()
print(torch_model_w1, "torch w1 权重")
print(torch_model_b1, "torch b1 权重")
print("-----------")
print(torch_model_w2, "torch w2 权重")
print(torch_model_b2, "torch b2 权重")
print("-----------")

#使用torch模型做预测
torch_x = torch.FloatTensor(x)
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：", y_pred)


# #把torch模型权重拿过来自己实现计算过程
diy_model = DiyModel(torch_model_w1, torch_model_b1, torch_model_w2, torch_model_b2)
# #用自己的模型来预测
y_pred_diy = diy_model.forward(x)
print("diy模型预测结果：", y_pred_diy)
