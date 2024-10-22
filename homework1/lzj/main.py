import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# 构建Dataset，便于DataLoader读取数据
class bostonDataset(Dataset):
    def __init__(self, file_path):
        raw_df = pd.read_csv(file_path, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        # target = target.reshape(-1,1)
        self.x = torch.tensor(data,dtype=torch.float32)
        self.label = torch.tensor(target,dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.label[item]


# 构建线性模型，这里采用的是单隐藏层的MLP模型
class LinearModel(nn.Module):
    def __init__(self, num_features, num_outputs, num_hidden):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(num_features, num_hidden)
        self.hidden = nn.ReLU()
        self.output = nn.Linear(num_hidden,num_outputs)

    def forward(self, x):
        y_hat = self.output(self.hidden(self.linear(x)))
        return y_hat


if __name__ == "__main__":
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    data = bostonDataset(file_path=data_url)
    dataloader = DataLoader(data, shuffle=False, batch_size=1)

    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    _X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])

    X = MinMaxScaler().fit_transform(_X)
    _y = raw_df.values[1::2, 2]

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, _y, test_size=0.2, random_state=5)
    X_train = torch.tensor(X_train,dtype=torch.float32)
    y_train = torch.tensor(y_train,dtype=torch.float32)
    X_test = torch.tensor(X_test,dtype=torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.float32)

    net = LinearModel(13, 1, 50)
    # 损失函数
    loss = torch.nn.MSELoss(reduction="mean")
    # 优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    num_epochs = 5000
    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)
    train_loss=[]
    test_loss = []
    for epoch in range(num_epochs):
        # for X,y in dataloader:
        l = loss(net(X_train), y_train)
        if epoch % 100 == 0:
            print(f'epoch {epoch}, train loss {l:f}')
            train_loss.append([epoch,l.item()])
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if epoch%100 ==0:
            l = loss(net(X_test), y_test)
            print(f'epoch {epoch}, test loss {l:f}')
            test_loss.append([epoch, l.item()])

    # 可视化
    train = np.array(train_loss)
    test = np.array(test_loss)
    plt.plot(train[:,0],train[:,1],label="train loss")
    plt.plot(test[:,0],test[:,1],color='red',linewidth=1.0,linestyle='--',label="test_loss")
    plt.legend()
    plt.xlabel('epoches')   # 设置x、y坐标轴信息
    plt.ylabel('loss')
    plt.show()