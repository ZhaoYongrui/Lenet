# -*- coding: UTF-8 -*-
# @Author : zyr
import torch
import torchvision.transforms as transforms
from PIL import Image

from lenet import LeNet

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = LeNet()
net.load_state_dict(torch.load('model/Lenet.pth'))
im = Image.open('image/飞机.jpg')
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].numpy()
    predict2 = torch.softmax(outputs, dim=1)

print(classes[int(predict)])
print(predict2)

# def main():
#     transform = transforms.Compose(
#         [transforms.Resize((32, 32)),
#          transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#     classes = ('plane', 'car', 'bird', 'cat',
#                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#     net = LeNet()
#     net.load_state_dict(torch.load('Lenet.pth'))
#
#     im = Image.open('1.jpg')
#     im = transform(im)  # [C, H, W]
#     im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
#
#     with torch.no_grad():
#         outputs = net(im)
#         predict = torch.max(outputs, dim=1)[1].numpy()
#     print(classes[int(predict)])
#
#
# if __name__ == '__main__':
#     main()
