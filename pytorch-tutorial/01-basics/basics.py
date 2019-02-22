import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189)


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create tensors. 在神经网络里,tensor存储的就是所要求的参数.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph. # 只有在形成计算图之后才能计算tensor的梯度
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward() # 就像我们考试的时候计算倒数一样,就是求y的各个因变量的导数

# Print out the gradients.
print(x.grad)    # x.grad = 2
print(w.grad)    # w.grad = 1
print(b.grad)    # b.grad = 1


## 反向传播计算导数是重叠的
w1 = torch.tensor([1.0,2.0,3.0],requires_grad=True)
d = torch.mean(w1) # 取均值,相当于将每个元素都*1/3
d.backward()  # 第一次反向传播,
w1.grad  # 查看w1的梯度,是0.333,正是求均值乘以的0.333
d.backward()  # 继续反向传播
w1.grad  # 查看w1的梯度,是0.666,是这次的梯度加上上一次的梯度,每次反向求导都是累加的结果

## 反向传播一般方法
w1.grad.data.zero_()
w1.grad
learning_rate = 0.001
w1.data.sub_(learning_rate*w1.grad.data)   # 梯度下降


## 如果每次我们都手动对每个tensor来进行梯度计算的话太麻烦,torch.optim封装了这些方法
## optimizer = optim.SGD(net.parameters(), lr = 0.01) 首先将网络中额
## optimizer.zero_grad()  相当于w1.grad.data.zero_(),会将网络中所有
## optimizer.step()  相当于w1.data.sub_(learning_rate*w1.grad.data),更新了参数






# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build a fully connected layer.
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)  # 当建立一个模块之后,模块的参数会自动生成
print ('b: ', linear.bias)

# Build loss function and optimizer.
criterion = nn.MSELoss()  # 计算损失函数的方法
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)  # 优化器

# Forward pass.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)  # 计算误差,这里的loss就是损失函数
print('loss: ', loss.item())

# Backward pass.
loss.backward()  # 对损失函数求导

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad)
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
optimizer.step()  # 梯度下降,执行一次

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# Create a numpy array.
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()


# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #

# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print (image.size())
print (label)

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass


# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #

# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0

# You can then use the prebuilt data loader.
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64,
                                           shuffle=True)


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))