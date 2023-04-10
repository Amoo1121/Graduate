import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import Fashion_MNIST_TRAIN


def test(file_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # Find your own hyper-parameters
    num_epochs = 15
    batch_size = 100
    # weight_decay = 0
    learning_rate = 0.05
    root = os.getcwd() + '/Fashion_MNIST/data/'
    test_data = torchvision.datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    # Test the model
    model = Fashion_MNIST_TRAIN.NeuralNet(input_channel=1, output_channel=10)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    model.load_state_dict(torch.load(file_dir + '/network_model.pth'))
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
