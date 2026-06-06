
import torch.nn as nn
'''
定义网络
'''
class SimpleNet(nn.Module):
    def __init__(self,input_size,output_dim=1):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(input_size, output_dim)
        self.activation = nn.Softmax(dim=1)
        self.loss = nn.MSELoss()
    ''' self.loss = nn.functional.mse_loss  # loss函数采用均方差损失 
        是一个回归问题，所以用均方误差
    '''
