# -*- coding: utf-8 -*-

# 导入相关模块进行配置
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")

plt.ion() # 交互模式


# 导入特征表观察
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

# 展示图片和它的特征

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)  # 显示照片
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')  # 画特征点
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
               landmarks)
plt.show()

# 在pytorch中读取图片和预处理

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        在__init__方法里读取csv,但不读取图片(图片在__getitem__中读),因为图片不需要全部保存在内存中.

        :param csv_file: (str) csv的路径
        :param root_dir: (str) 所有图片的路径
        :param transform: (callable) 图片转换的方法
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        重写父类的__len__方法,获得数据集的大小
        :return: 返回数据集的大小
        """
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        """
        重写__getitem__方法.
        img_name使用图片路径root_dir和csv中第一列图片名称拼出单个图片的路径.
        image是读取的图片.
        landmarks是通过csv获得的特征点(第一列到最后一列).转换成2列的形状,既每个点一行.
        sample是由图片和特征点构成的字典.
        如果transfrom方法存在则对样本进行变换.

        :param idx: 重写__getitem__方法,以后可以方便索引预处理之后的样本.
        :return: 处理好的样本
        """
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

# 读取图片并展示

face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):  # 遍历dataset
    sample = face_dataset[i]  # 取一个样本

    print(i, sample['image'].shape, sample['landmarks'].shape)  # 获得图片id,图片形状,特征形状
    ax = plt.subplot(1, 4, i + 1) # 做出一个图片
    plt.tight_layout()  # 规整各个图片
    ax.set_title('Sample #{}'.format(i))  # 给每个子图添加标题
    ax.axis('off')  # 不显示坐标轴
    show_landmarks(**sample)  # 画出特征点

    if i == 3:  # 如果循环到第三个图片,则停止
        plt.show()
        break

# 构造transform函数

class Rescale(object):

    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        """
        将图片转换为指定的形状
        :param output_size: (tuple or int) 如果是元组则输出图片是元组的形状.如果是int,则将最小边设置成output_size,长边等距离变换
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]  # 获得图片的高和宽
        if isinstance(self.output_size, int):  # 如果是int,短边设置output_size,长边等比例缩放
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size  # 如果是元组则直接转换为元组的大小

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))  # pytorch自带的转换函数

        landmarks = landmarks * [new_w / w, new_h / h]  # 特征点同样的缩放

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        """
        随机剪切图片
        :param output_size: (tuple or int) 输出的图片的大小
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

# 比较不同的转换方法

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])  # 集合了上述两种方法

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()


# 读取图片并转换

transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break

# 载入图片

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)  # win下不支持多线程,所以调成0,或者默认,默认是0


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

######################################################################
# Afterword: torchvision
# ----------------------
#
# In this tutorial, we have seen how to write and use datasets, transforms
# and dataloader. ``torchvision`` package provides some common datasets and
# transforms. You might not even have to write custom classes. One of the
# more generic datasets available in torchvision is ``ImageFolder``.
# It assumes that images are organized in the following way: ::
#
#     root/ants/xxx.png
#     root/ants/xxy.jpeg
#     root/ants/xxz.png
#     .
#     .
#     .
#     root/bees/123.jpg
#     root/bees/nsdf3.png
#     root/bees/asd932_.png
#
# where 'ants', 'bees' etc. are class labels. Similarly generic transforms
# which operate on ``PIL.Image`` like  ``RandomHorizontalFlip``, ``Scale``,
# are also available. You can use these to write a dataloader like this: ::
#
#   import torch
#   from torchvision import transforms, datasets
#
#   data_transform = transforms.Compose([
#           transforms.RandomSizedCrop(224),
#           transforms.RandomHorizontalFlip(),
#           transforms.ToTensor(),
#           transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#       ])
#   hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
#                                              transform=data_transform)
#   dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                                batch_size=4, shuffle=True,
#                                                num_workers=4)
#
# For an example with training code, please see
# :doc:`transfer_learning_tutorial`.
