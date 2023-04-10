import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data import TrainData
from cfg import DefaultConfig
import AENetwork
from loss import SSIM


def train(cfg):
    model = AENetwork.AENetwork()
    model.load_state_dict(torch.load(cfg.load_model_path + 'network_model.pth'))  # 每次跟随上次进度训练
    if cfg.use_gpu:
        model.cuda()  # 是否使用cuda
    MSE_criterion = nn.MSELoss().cuda()  # 像素级别误差
    SSIM_criterion = SSIM().cuda()  # 结构级别误差
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # 选用Adam优化，采用二阶动量模型更快收敛
    train_dataset = TrainData(cfg=cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, drop_last=True)

    for epoch in range(cfg.max_epoch):
        running_loss = 0.0
        for index, item in enumerate(train_dataloader, 1):
            input_frame = item
            noise = torch.randn(input_frame.shape)  # 输入噪声防止过拟合
            noise_input = input_frame + noise
            if cfg.use_gpu:
                noise_input = noise_input.cuda()
                input_frame = input_frame.cuda()
            optimizer.zero_grad()  # 先进行梯度清零防止存有上一次梯度
            outputs = model(noise_input)  # 将含有噪声的图像输入网络得到结果
            # 使用定义的损失，权重9：1
            loss = 0.9 * MSE_criterion(outputs, input_frame) + 0.1 * SSIM_criterion(outputs, input_frame)
            loss.backward()  # 反向传播梯度，更新权重矩阵
            optimizer.step()
            running_loss += loss.item()  # 防止每次调用loss.item()耗时，先赋值
            show_loss = running_loss / (1 * len(train_dataset))
            if index == len(train_dataloader):
                print('Epoch:{0}, loss = {1}'.format(str(epoch + 1), show_loss))

        if epoch % 30 == 0:
            model_dict = model.state_dict()
            torch.save(model_dict, cfg.load_model_path + 'network_model.pth')  # 每30次保存一次网络模型


cfg = DefaultConfig()  # 获取默认超参数
train(cfg)  # 开始训练
