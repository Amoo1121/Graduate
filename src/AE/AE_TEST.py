import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import AENetwork
# from cfg import DefaultConfig
# import time

def test(file_dir):
    torch.backends.cudnn.deterministic = True  # 因为使用了GPU，需要保证种子固定才能保证每次结果相同
    torch.backends.cudnn.benchmark = True  # 让cuDNN找到最合适的算法
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AENetwork.AENetwork()
    model.eval()  # 不使用BN和dropout，因为测试batch太小，会被BN和dropout影响
    model.load_state_dict(torch.load(file_dir + '/network_model.pth'))  # 读模型
    model.to(device)

    # 读取图像进行预处理，转换为tensor
    image_path = r'./AE/data/bottle//test/defect/295.jpg'
    img = cv2.imread(image_path)
    image = cv2.resize(img, (256, 256))
    image_resized = (image.astype('float32') / 127.5) - 1.0  # to -1 ~ 1
    image_resized = np.transpose(image_resized, [2, 0, 1])
    input_tensor = torch.from_numpy(image_resized).to(device).unsqueeze(0)

    output_image = model(input_tensor)
    # 需要优化
    # output_image = output_image.clone().squeeze(0).cpu().detach().numpy()
    output_image = output_image.clone().squeeze(0).detach().cpu().numpy()
    cv2_frame = ((output_image + 1) * 127.5).transpose(1, 2, 0).astype('uint8')

    # 图像求差得到缺陷图
    res_image = cv2.cvtColor(np.power(cv2.subtract(cv2_frame, image), 1), cv2.COLOR_BGR2GRAY)
    res_image = cv2.medianBlur(res_image, 3)

    # 打印结果
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 输出中文字体
    plt.figure(figsize=(19, 19))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('原始图像')
    plt.subplot(1, 3, 2)
    plt.imshow(cv2_frame)
    plt.title('重构图像')
    plt.subplot(1, 3, 3)
    plt.imshow(res_image)
    plt.title('残差图像')
    plt.show()

    # cost 0.030434608459472656 s
    # cost 0.1248326301574707 s
    # cost 0.007006168365478516 s
    # cost 2.6321322917938232 s
    # cost 0.0009963512420654297 s

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False  # 输出中文字体
    # plt.figure(figsize=(19, 19))
    # plt.subplot(2, 1, 1)
    # plt.imshow(image)
    # plt.title('原始图像')
    # plt.subplot(2, 1, 2)
    # plt.imshow(res_image)
    # plt.title('残差图像')
    # plt.show()