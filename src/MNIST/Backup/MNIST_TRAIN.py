import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch.optim as optim
import os


# 网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (n,1,28,28) to (n,784)
        x = f.relu(self.conv1(x))
        # batch_size = x.size(0)
        x = self.pooling(x)
        x = f.relu(self.conv2(x))
        x = self.pooling(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
        # x = f.relu(self.pooling(self.conv1(x)))
        # x = f.relu(self.pooling(self.conv2(x)))
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # return x


# 训练
def train(file_dir, EPOCH):
    # 准备数据集
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307, ], std=[0.3081, ])
    ])
    root = os.getcwd() + '/MNIST/mnist_data/'
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = Model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if os.path.exists(file_dir + '/MNIST/NEW_Network_Model'):
        model_dict = torch.load(file_dir + '/MNIST/NEW_Network_Model')
        model.load_state_dict(model_dict)
    model.to(device)
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    running_loss = 0.0
    for epoch in range(EPOCH):
        for i, data in enumerate(train_dataloader, 0):
            input, target = data
            input, target = input.to(device), target.to(device)
            y_pred = model(input)
            loss = criterion(y_pred, target)
            # print(i+1,epoch+1,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 300 == 299:
                print("Epoch:{}, {}, loss:{:.3f}".format(epoch + 1, i + 1, running_loss / 300))
                running_loss = 0.0
    model_dict = model.state_dict()
    torch.save(model_dict, file_dir + '/MNIST/NEW_Network_Model/network_model.pth')  # 保存一次网络模型
