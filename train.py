import torch
from torch.utils.data import TensorDataset, DataLoader
from mydataloader import MyDataLoader
from myconfig import MyConfig
from mymodel import SimpleMLP, WhatNet
import matplotlib.pyplot as plt
import time

config = MyConfig()
data_path = config.data_path
results_path = config.results_path
train_ratio = config.train_ratio
label_pos = config.label_pos
batch_size = config.batch_size
num_epoch = config.num_epoch
lr = config.lr
device = config.device

data = MyDataLoader.load(data_path)

##############################
### 对特征进行Z-score标准化 ###
##############################

mu = data[:, label_pos+1:].mean(dim=0, keepdim=True)
sigma = data[:, label_pos+1:].std(dim=0, keepdim=True)
data[:, label_pos+1:] = (data[:, label_pos+1:] - mu) / sigma

#################
### 划分数据集 ###
#################

num_train = int(data.shape[0] * train_ratio)
num_valid = len(data) - num_train

# 直接搬运到device
train_X = data[:num_train, label_pos+1:].to(device)
train_y = data[:num_train, label_pos].reshape(-1, 1).to(device)
valid_X = data[num_train:, label_pos+1:].to(device)
valid_y = data[num_train:, label_pos].reshape(-1, 1).to(device)

train_dataset = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

############
### 训练 ###
############

net = WhatNet()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
net.to(device) # 搬模型可原地，搬数据需要赋值
train_hist, valid_hist = [], []

for epoch in range(num_epoch):
    t0 = time.time()
    net.train()
    loss_train = 0
    for X, y in train_loader:
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item() * len(y)

    net.eval()
    with torch.no_grad():
        y_hat_valid = net(valid_X)
        loss_valid = loss_fn(y_hat_valid, valid_y).item()

    loss_train /= num_train
    train_hist.append(loss_train)
    valid_hist.append(loss_valid)
    t1 = time.time()
    print(f"epoch: {epoch+1}, loss_train: {loss_train:.5f}, loss_valid: {loss_valid:.5f}, time: {t1-t0:.2f} s")

##########################
### 用MSE和R**2评估结果 ###
##########################

def cal_r2(y_mse, y_true):
    """
    根据MSE和真值计算决定系数R**2
    """
    y_true = y_true.reshape(-1)
    y_var = y_true.var().item()
    return 1 - (y_mse / y_var)

valid_mse = valid_hist[-1]
with torch.no_grad():
    y_hat_train = net(train_X)
    train_mse = loss_fn(y_hat_train, train_y).item()

valid_r2 = cal_r2(valid_mse, valid_y)
train_r2 = cal_r2(train_mse, train_y)
print(f"train_mse: {train_mse:.5f}, train_r2: {train_r2:.5f}")
print(f"valid_mse: {valid_mse:.5f}, valid_r2: {valid_r2:.5f}")

####################
### 绘制loss曲线 ###
####################

epochs = [epoch + 1 for epoch in range(num_epoch)]

plt.figure()
plt.plot(epochs, train_hist, label="train")
plt.plot(epochs, valid_hist, label="valid")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig(results_path + "/loss.png")

###################
### pred-true图 ###
###################

plt.figure()
plt.scatter(train_y.to("cpu"), y_hat_train.to("cpu"), label="train")
plt.scatter(valid_y.to("cpu"), y_hat_valid.to("cpu"), label="valid")
plt.xlabel("true")
plt.ylabel("pred")
plt.legend()
plt.savefig(results_path + "/pred-true.png")

###############
### 保存结果 ###
###############

torch.save(net.state_dict(), results_path + "/net.params")
torch.save(mu, results_path + "/mu.txt")
torch.save(sigma, results_path + "/sigma.txt")

epochs = torch.tensor(epochs)
train_hist = torch.tensor(train_hist)
valid_hist = torch.tensor(valid_hist)
hist = torch.column_stack((epochs, train_hist, valid_hist))
torch.save(hist, results_path + "/hist.txt")

train = torch.column_stack((train_y, y_hat_train)).to("cpu")
valid = torch.column_stack((valid_y, y_hat_valid)).to("cpu")
torch.save(train, results_path + "/train.txt")
torch.save(valid, results_path + "/valid.txt")