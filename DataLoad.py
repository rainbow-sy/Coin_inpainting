from imgaug import augmenters as iaa
from tqdm import tqdm
import cv2
import os
import numpy as np

image_size = 500

def load_data():
    trainPath = r'E:\CleanedCoin'
    x_train = []
    for file in tqdm(os.listdir(trainPath)):
        if (file.endswith('.png')):
            image = cv2.imread(os.path.join(trainPath, file), 1)  # load images in gray.0代表单通道
            image = cv2.bilateralFilter(image, 2, 50, 50)  # remove images noise.
            image = cv2.resize(image, (image_size, image_size))  # resize images into 150*150.
            x_train.append(image)
    x_train = np.array(x_train)
    #     x_train=x_train.resahpe(-1,500,500,3)
    images_aug1 = seq1.augment_images(images=x_train)
    images_aug2 = seq2.augment_images(images=x_train)
    images_aug3 = seq3.augment_images(images=x_train)
    images_aug4 = seq4.augment_images(images=x_train)
    images_aug5 = seq5.augment_images(images=x_train)
    images_aug6 = seq5.augment_images(images=x_train)
    images_aug7 = seq1.augment_images(images=x_train)
    images_aug8 = seq2.augment_images(images=x_train)
    images_aug9 = seq3.augment_images(images=x_train)
    Y_all = np.concatenate((x_train, images_aug1, images_aug2, images_aug3, images_aug4, images_aug5,
                            images_aug6, images_aug7, images_aug8, images_aug9), axis=0)
    X_all = aug.augment_images(images=Y_all)
    return X_all,Y_all

#——————————————————————————制作原始数据集y的增强方法————————————————————————————————
seq1 = iaa.SomeOf((0, None), [
    iaa.Affine(rotate=(-180, 180)),   #旋转-15到15度
    iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)#添加高斯噪声
])
seq2 = iaa.Sequential([
    iaa.Affine(rotate=(-180, 180)),
    iaa.Invert(0.5)
])
seq3 = iaa.Sequential([
    iaa.Fliplr(), #水平翻转图像
    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
    iaa.Add((-40, 40), per_channel=0.5)

])
seq4 = iaa.OneOf([
    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    iaa.Affine(rotate=(-180, 180)),
    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})#平移图像
])
seq5 =  iaa.Affine(
        rotate=(-180, 180),
    )
#——————————————————制作破坏数据集x的增强方法————————————————————————————————
#制作模糊图像
def img_func(images, random_state, parents, hooks):
    for img in images:
        img[::4] = 0
    return images

def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images

aug=iaa.OneOf([
    iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),   # 每个像素的移动强度范围为0到5.
    iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),  #矩形丢弃
    iaa.Dropout(p=(0, 0.2)),
    iaa.GaussianBlur(sigma=(0, 0.5)),#添加高斯噪声
    iaa.Lambda(img_func, keypoint_func),
    iaa.Superpixels(p_replace=(0.1, 0.3), n_segments=(16, 128)),
])