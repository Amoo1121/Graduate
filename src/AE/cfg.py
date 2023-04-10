class DefaultConfig(object):

    train_data_path = './data/bottle/train/good1'  # 训练集路径
    load_model_path = './New_Network_Model/'  # 模型路径

    use_gpu = True  # 是否使用GPU
    train_batch_size = 8  # batch大小，因为训练样本较少，选取的batch_size也较小
    max_epoch = 4000  # 最大迭代次数
    lr = 0.0001  # 学习率，从0.001开始，不合适增大或缩小10倍，最后选择0.0001
    lr_decay = 0.90  # 学习率衰减，没用到
    weight_decay = 1e-5  # Adam权重衰减，代替正则化，最后没有用到
