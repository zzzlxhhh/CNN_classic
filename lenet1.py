import torchvision as tv
import torch
import time
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 128
resize_transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])
train_dataset = datasets.ImageFolder('data/mnist/trainset',transform=resize_transform)
test_dataset = datasets.ImageFolder('data/mnist/testset',transform=resize_transform)
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.conv2 = nn.Conv2d(4, 12, 5)
        self.fc1   = nn.Linear(12*5*5, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

def compute_accuracy(model, data_loader):
    correct, total = 0, 0
    for i, (images, labels) in enumerate(data_loader):            
        images=images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return correct.float()/total * 100

net = Net().cuda()
print(net)

from torch import optim
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)

start_time = time.time()
for epoch in range(10):

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):

        # 输入数据
        inputs=inputs.cuda()
        labels=labels.cuda() 
        

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        if i % 50 == 49:  
            print('[%d, %5d] loss: %.3f' \
                  % (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

    net.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%%' % (
              epoch+1, 10, 
              compute_accuracy(net, train_loader)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))           
print('Finished Training')

with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(net, test_loader)))