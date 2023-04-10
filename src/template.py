import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data import TrainData
from cfg import DefaultConfig


# 可以使用Ctrl+Alt+L对代码格式进行格式化，会更整洁
#以下代码均需自行按需修改

# 定义网络模型
class Model_name(nn.Module):
    def __init__(self):
        super(Model_name, self).__init__()

$network


# other network layers

def forward(self, x):
    x = self.conv1(x)
    # others
    return x


# 定义训练（不完整 请自行修改）
def train(cfg):
    model = AENetwork.AENetwork()
    model.load_state_dict(torch.load(cfg.load_model_path + 'network_model.pth'))  # 每次跟随上次进度训练
    if cfg.use_gpu:
        model.cuda()  # 是否使用cuda
    MSE_criterion = nn.MSELoss().cuda()  # 定义误差
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # 选择优化器
    train_dataset = TrainData(cfg=cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, drop_last=True)

    for epoch in range(cfg.max_epoch):
        for index, item in enumerate(train_dataloader, 1):
            input_frame = item
            if cfg.use_gpu:
                input_frame = input_frame.cuda()
            optimizer.zero_grad()  # 先进行梯度清零防止存有上一次梯度
            loss = MSE_criterion(outputs, input_frame)
            loss.backward()  # 反向传播梯度，更新权重矩阵
            optimizer.step()
        if epoch % 30 == 0:
            model_dict = model.state_dict()
            torch.save(model_dict, cfg.load_model_path + 'network_model.pth')  # 每30次保存一次网络模型


cfg = DefaultConfig()  # 获取默认超参数
train(cfg)  # 开始训练
