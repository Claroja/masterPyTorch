import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters 超参数设置
input_size = 1  # 输入节点数
output_size = 1  # 输出节点数
num_epochs = 60  # 迭代此次数
learning_rate = 0.001  # 学习速率

# Toy dataset  数据集
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model  线性回归模型
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # 计算误差
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 梯度下降

# Train the model  训练模型
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors  将Numpy转换为torch
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass  # 前馈
    outputs = model(inputs)  # 输入
    loss = criterion(outputs, targets)  # 计算误差
    
    # Backward and optimize
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 向后传播
    optimizer.step()  # 迭代
    
    if (epoch+1) % 5 == 0:  # 每5次打印一次
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')  # 保留模型


