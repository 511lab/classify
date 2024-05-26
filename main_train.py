import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from torch.autograd import Variable
from torch.utils.data import Subset
import numpy as np
import pandas as pd
# 设置超参数
BATCH_SIZE = 8
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(50),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# 读取数据
dataset_train = datasets.ImageFolder(r'C:\Users\Administrator\PycharmProjects\pythonProject\de\train_set',transform)
# print(dataset_train.imgs)
# 对应文件夹的label
print(dataset_train.class_to_idx)
# 确定验证集占总数据集的百分比
validation_split = 0.1
# 计算验证集的大小
num_validation = int(validation_split * len(dataset_train))
# 随机选择用于验证集的索引
indices = np.random.choice(range(len(dataset_train)), num_validation, replace=False)
# 使用选定的索引创建验证集
validation_dataset = Subset(dataset_train, indices)
# 剩下的数据用作训练集
indices_train = list(set(range(len(dataset_train))) - set(indices))
train_dataset = Subset(dataset_train, indices_train)
# 切分数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
modellr = 1e-4
 # 实例化模型并且移动到GPU
criterion = nn.CrossEntropyLoss()
# model = effnetv2_s()
model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
num_ftrs=model.fc.in_features
# num_ftrs = model.classifier[-1].in_features  # 获取最后一个全连接层的输入特征数
model.classifier = nn.Linear(num_ftrs, 2)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=modellr)
train_accuracies = []
val_accuracies = []
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 15))
    print("学习率lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew
# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    losslist=[]
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    # print(total_num, len(train_loader))
    samples=0
    correct=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
        print_loss = loss.data.item()
        losslist.append(print_loss)
        sum_loss += print_loss
        samples += data.shape[0]
        correct  += torch.sum(pre_lab == target)

    if (batch_idx + 1) % 8 == 0 or batch_idx == len(train_loader)-1:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.3f} Accuracy{:.3f}'.format(
                epoch,
                samples,
                len(train_loader.dataset),
                100.*(samples)/(len(train_loader.dataset)),
                loss.item(),
                float(100*correct/samples)
                ))
    train_acc = 100 * correct / total_num  # 计算当前epoch的训练准确率
    train_accuracies.append(train_acc)
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{} ave_loss:{:.4f}'.format(epoch, ave_loss))
    # plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
def val(model, device, val_loader):

    model.eval() #将模型设置为评估模式，这会关闭特定层（如Dropout和BatchNorm）的训练特定行为。
    val_loss = 0
    correct = 0
    total_num = len(val_loader.dataset)
    print(total_num, len(val_loader))
    with torch.no_grad():   # 将模型设置为评估模式，这会关闭特定层（如Dropout和BatchNorm）的训练特定行为。
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            val_loss += print_loss

        correct = correct.data.item()
        acc = correct / total_num
        avgloss = val_loss / len(val_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(val_loader.dataset), 100 * acc))
    # plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, DEVICE, train_loader, optimizer, epoch)
    val(model, DEVICE, val_loader)
df_accuracy = pd.DataFrame({
    'Epoch': range(1, EPOCHS + 1),
    'Train_Accuracy': train_accuracies,
    'Validation_Accuracy': val_accuracies})
df_accuracy.to_excel('accuracy.xlsx', index=False)


