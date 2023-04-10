import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt
import MNIST_TRAIN
import os


def test(file_dir):
    # 准备数据集
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307, ], std=[0.3081, ])
    ])

    root = os.getcwd() + '/MNIST/mnist_data/'
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=False)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    model = MNIST_TRAIN.Model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.load_state_dict(torch.load(file_dir + '/network_model.pth'))  # 读模型
    model.to(device)

    # 测试
    accuracy_list = []

    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            input, target = data
            input, target = input.to(device), target.to(device)
            y_pred = model(input)
            predicted = torch.argmax(y_pred.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        accuracy = correct / total
        accuracy_list.append(accuracy)
        print("Accuracy on test set:{:.2f} %".format(100 * correct / total))
