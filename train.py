# -*- coding: UTF-8 -*-
# @Author : zyr

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

from lenet import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=36,
                          shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=5000,
                         shuffle=False, num_workers=0)
test_data_iter = iter(test_loader)
test_image, test_label = next(test_data_iter)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5):  # 训练集迭代轮数

    running_loss = 0.0  # 每个epoch的损失值
    for step, data in enumerate(train_loader, start=0):  # 使用enumerate()函数遍历train_loader，将数据组合成一个索引序列
        inputs, labels = data  # 读取数据
        optimizer.zero_grad()  # 每计算一个batch，进行一次梯度清零
        outputs = net(inputs)  # 将数据输入网络
        loss = loss_function(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 打印训练信息
        running_loss += loss.item()
        if step % 500 == 499:  # 每500个batch打印一次训练信息
            with torch.no_grad():  # 不计算梯度
                outputs = net(test_image)  # 将测试图片输入网络
                predict_y = torch.max(outputs, dim=1)[1]  # 计算预测值
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)  # 计算准确率

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')

save_path = './model/Lenet.pth'
torch.save(net.state_dict(), save_path)

# 显示图像
# def imshow(img):
#     img = img / 2 + 0.5    # unnormalize
#     nping = img.numpy()
#     plt.imshow(np.transpose(nping, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_image))




# def main():
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#     # 50000张训练图片
#     # 第一次使用时要将download设置为True才会自动去下载数据集
#     train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                              download=False, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
#                                                shuffle=True, num_workers=0)
#
#     # 10000张验证图片
#     # 第一次使用时要将download设置为True才会自动去下载数据集
#     val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                            download=False, transform=transform)
#     val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
#                                              shuffle=False, num_workers=0)
#     val_data_iter = iter(val_loader)
#     val_image, val_label = next(val_data_iter)
#
#     # classes = ('plane', 'car', 'bird', 'cat',
#     #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#     net = LeNet()
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
#
#     for epoch in range(5):  # loop over the dataset multiple times
#
#         running_loss = 0.0
#         for step, data in enumerate(train_loader, start=0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = loss_function(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             running_loss += loss.item()
#             if step % 500 == 499:  # print every 500 mini-batches
#                 with torch.no_grad():
#                     outputs = net(val_image)  # [batch, 10]
#                     predict_y = torch.max(outputs, dim=1)[1]
#                     accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
#
#                     print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
#                           (epoch + 1, step + 1, running_loss / 500, accuracy))
#                     running_loss = 0.0
#
#     print('Finished Training')
#
#     save_path = './Lenet.pth'
#     torch.save(net.state_dict(), save_path)
#
#
# if __name__ == '__main__':
#     main()
