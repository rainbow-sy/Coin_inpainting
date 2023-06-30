import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import Net
# from scipy.io import savemat
def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    net.load_state_dict(torch.load('./model_best.pth'))

    im = Image.open(r'./yingbi.png')  #读取方式[H,W,C]
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
    #展示图片信息
    # def imshow(img):
    img =outputs/2+0.5   #反标准化
    img=torch.clamp(img*255,0,255)
    npimg=img.numpy()
    npimg=np.squeeze(npimg)#把shape中为1的维度去掉
    npimg=np.transpose(npimg, (1, 2, 0))
    print(npimg)
    # file_name = './data.mat'
    # savemat(file_name, {'a':npimg})
    plt.imshow(npimg)
    plt.show()
    plt.imsave('./CoinPre.png', npimg)
    # imshow(outputs)

if __name__ == '__main__':
    main()
