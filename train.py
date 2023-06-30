import sys
import time
# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
from tqdm import tqdm, trange  # 进度条
# from torchvision.utils import save_image
from torch.autograd import Variable
# from torch.optim.lr_scheduler import MultiStepLR
# import utils
from model import Net
from DataLoad import load_data
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils import torchPSNR as psnr
# from PIL import Image
import argparse

class CoinDataset(Dataset):
    def __init__(self, images, labels):
        # images
        self.X = images
        # labels
        self.y = labels
        # Transformation for converting original image array to an image and then convert it to a tensor
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = self.transform(self.X[idx])
        target = self.transform(self.y[idx])
        return data, target

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    img_size = 500
    #------------------------------------------------数据集加载---------------------------------------
    X, Y = load_data()
    print('所有数据', len(X))
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1,
                                                        shuffle=True)  # 70% training, 30% testing
    print(len(X_train))
    print(len(X_valid))
    train_set = CoinDataset(X_train, y_train)
    valid_set = CoinDataset(X_valid, y_valid)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_Loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_Loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)

    # ------------------------------------------------建立模型---------------------------------------
    myNet = Net()  # 实例化网络
    myNet = myNet.cuda()  # 网络放入GPU中
    criterion = nn.MSELoss().cuda()

    optimizer = optim.Adam(myNet.parameters(), lr=args.lr)  # 网络参数优化算法

    # ------------------------------------------------开始训练---------------------------------------
    if os.path.exists('./model_best.pth'):  # 判断是否预训练
        myNet.load_state_dict(torch.load('./model_best.pth'))

    loss_list = []
    best_psnr = 0  # 训练最好的峰值信噪比
    best_epoch = 0  # 峰值信噪比最好时的 epoch

    for epoch in range(args.epochs):
        myNet.train()  # 指定网络模型训练状态
        iters = tqdm(train_Loader, file=sys.stdout)  # 实例化 tqdm，自定义
        epochLoss = 0  # 每次训练的损失
        timeStart = time.time()  # 每次训练开始时间
        for index, (x, y) in enumerate(iters, 0):
            myNet.zero_grad()  # 模型参数梯度置0
            optimizer.zero_grad()  # 同上等效

            input_train, target = Variable(x).cuda(), Variable(y).cuda()  # 转为可求导变量并放入 GPU
            output_train = myNet(input_train)  # 输入网络，得到相应输出

            loss = criterion(output_train, target)  # 计算网络输出与目标输出的损失

            loss.backward()  # 反向传播
            optimizer.step()  # 更新网络参数
            epochLoss += loss.item()  # 累计一次训练的损失

            # 自定义进度条前缀
            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch + 1, args.epochs, loss.item()))

        # 评估
        myNet.eval()
        psnr_val_rgb = []
        for index, (x, y) in enumerate(val_Loader, 0):
            input_, target_value = x.cuda(), y.cuda()
            with torch.no_grad():
                output_value = myNet(input_)
            for output_value, target_value in zip(output_value, target_value):
                psnr_val_rgb.append(psnr(output_value, target_value))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save(myNet.state_dict(), 'model_best.pth')

        loss_list.append(epochLoss)  # 插入每次训练的损失值
        torch.save(myNet.state_dict(), 'model.pth')  # 每次训练结束保存模型参数
        timeEnd = time.time()  # 每次训练结束时间
        print("------------------------------------------------------------")
        print(
            "Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}.".format(epoch + 1, timeEnd - timeStart, epochLoss))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))

if __name__ == '__main__':  # 只有在 main 中才能开多线程
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='./convnext_tiny_1k_224_ema.pth',
                        help='initial weights path')
    # # 是否冻结head以外所有权重
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

