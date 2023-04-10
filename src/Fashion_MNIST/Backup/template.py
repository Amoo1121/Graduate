import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os


class NeuralNet(nn.Module):
    def __init__(self, input_channel=1, output_channel=10, num_classes=10):
        super(NeuralNet, self).__init__()
        global device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        # self.conv1 = torch.nn.Sequential(
        #     nn.Conv2d(input_channel, 96, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(96),  # 批标准化
        #     nn.ReLU(),  # relu激活函数
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层
        # )
        # self.conv2 = torch.nn.Sequential(
        #     nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.conv3 = torch.nn.Sequential(
        #     nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        # self.conv4 = torch.nn.Sequential(
        #     nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        # self.conv5 = torch.nn.Sequential(
        #     nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(256 * 2 * 2, 2048),  # 28*28的图像池化三次后图像大小为2*2
        #     nn.ReLU(),
        #     nn.Dropout2d(0.5),
        #     nn.Linear(2048, 2048),
        #     nn.ReLU(),
        #     nn.Dropout2d(0.5),
        #     nn.Linear(2048, output_channel),
        # )
        # '''
        # AlexNet改
        # '''
$network

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.conv2(out)
        # out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv5(out)
        # out = out.view(out.size(0), -1)  # 变换维度，相当于拉直操作
        # out = self.classifier(out)
        # return out


def train(file_dir, EPOCH):
    global device
    # Find your own hyper-parameters
    num_epochs = 15 #这个是需要自己改的目标Epoch，如果不用的话可以把下面输出的部分一起改掉
    batch_size = 100
    # weight_decay = 0
    learning_rate = 0.05
    root = os.getcwd() + '/Fashion_MNIST/data/'
    training_data = torchvision.datasets.FashionMNIST(
        root=root,
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    model = NeuralNet(input_channel=1, output_channel=10)
    if os.path.exists(file_dir + '/Fashion_MNIST/NEW_Network_Model/network_model.pth'):
        model_dict = torch.load(file_dir + '/Fashion_MNIST/NEW_Network_Model/network_model.pth')
        model.load_state_dict(model_dict)
    model.to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    # Train the model
    total_step = len(train_dataloader)
    minLoss = 0.1
    for epoch in range(EPOCH):
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #打印部分存在num_epochs变量，与上面的自定义一起更改
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        model_dict = model.state_dict()
        torch.save(model_dict, file_dir + '/Fashion_MNIST/New_Network_Model/network.pth')
