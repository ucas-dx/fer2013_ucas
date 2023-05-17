# -*- coding: utf-8 -*-
# @Author  : Dengxun
# @Time    : 2023/5/16 15:51
# @Function: 国科大情感计算大报告
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import h5py
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
#导入预训练权重文件
from torchvision.models import ResNet18_Weights, ResNet34_Weights, AlexNet_Weights, ResNet50_Weights, \
    EfficientNet_B5_Weights, ViT_B_16_Weights

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# FER2013 Dataset

class Easy_CNN(nn.Module):
    """
    设置一个简易的CNN网络，进行全网络参数训练
    """
    def __init__(self):
        super(Easy_CNN,self).__init__()
        self.feature=nn.Sequential(nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ELU(),nn.MaxPool2d(2),nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(128),nn.ELU(),nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(256),nn.ELU(),nn.MaxPool2d(2),nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(256),nn.ELU(),nn.MaxPool2d(2))
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(256*28*28,2048),nn.BatchNorm1d(2048),nn.ELU(),nn.Dropout(0.125),nn.Linear(2048,512),nn.BatchNorm1d(512),nn.ELU(),nn.Linear(512,7))

    def forward(self,x):
        x=self.feature(x)
        x=x.view(x.size(0), -1)
        out=self.fc(x)
        return out

#预处理fer2013数据集，将csv文件转化为训练集，公共测试集和私有测试集
class FER2013(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('data.h5', 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((28709, 48, 48))
            self.train_data = np.array(self.train_data)
        elif self.split == 'PublicTest':
            self.PublicTest_data = self.data['PublicTest_pixel']
            self.PublicTest_labels = self.data['PublicTest_label']
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.PublicTest_data = self.PublicTest_data.reshape((3589, 48, 48))
            self.PublicTest_data = np.array(self.PublicTest_data)
        else:
            self.PrivateTest_data = self.data['PrivateTest_pixel']
            self.PrivateTest_labels = self.data['PrivateTest_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3589, 48, 48))
            self.PrivateTest_data = np.array(self.PrivateTest_data)

    def __getitem__(self, index):
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)



#数据预处理，先拉伸到256*256，使用CenterCrop模式裁剪为224*224，再进行图像归一化处理
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
#创建数据加载器，batchsize设置为32
trainset = FER2013(split='Training', transform=transform_train)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

PublicTestset = FER2013(split='PublicTest', transform=transform_test)
PublicTestloader = DataLoader(PublicTestset, batch_size=32, shuffle=False)

PrivateTestset = FER2013(split='PrivateTest', transform=transform_test)
PrivateTestloader = DataLoader(PrivateTestset, batch_size=32, shuffle=False)

#定义训练与测试函数
def train_loop(model_name,imgname,epochs,learnrate):
    model=model_name.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learnrate)
    num_epochs =epochs
    train_loss_history = []
    train_acc_history = []
    test_loss_history1 = []
    test_acc_history1 = []
    test_loss_history2 = []
    test_acc_history2 = []
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in tqdm.tqdm(enumerate(trainloader),total=len(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(trainloader)
        train_accuracy = correct / total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_accuracy)
        # Evaluation on PublicTest dataset
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(PublicTestloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs =model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_loss = running_loss / len(PublicTestloader)
        test_accuracy = correct / total
        test_loss_history1.append(test_loss)
        test_acc_history1.append(test_accuracy)
        # Evaluation on PrivateTest dataset
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(PrivateTestloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss3 = criterion(outputs, labels)
                running_loss += loss3.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        private_test_loss = running_loss / len(PrivateTestloader)
        private_test_accuracy = correct / total
        test_loss_history2.append(private_test_loss)
        test_acc_history2.append(private_test_accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Public Test Loss: {test_loss:.4f}, Public Test Accuracy: {test_accuracy:.4f}, "
              f"Private Test Loss: {private_test_loss:.4f}, Private Test Accuracy: {private_test_accuracy:.4f}")
        if (epoch+1)==epochs:
            torch.save(model.state_dict(), imgname+'model.pth')



        # Calculate accuracy for each class in the PrivateTest dataset
        def boxmap(testloder,name):
            model.eval()
            class_correct = np.zeros(7)
            class_total = np.zeros(7)
            with torch.no_grad():
                for inputs, labels in testloder:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    c = (predicted == labels).squeeze()
                    for i in range(len(labels)):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
            # Calculate accuracy percentage for each class
            class_accuracy = class_correct / class_total * 100
            # Plotting the class accuracy
            plt.figure(figsize=(8, 6))
            classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            plt.bar(classes, class_accuracy)
            plt.xlabel('Emotion Class')
            plt.ylabel('Accuracy (%)')
            plt.title(imgname+name+'Test Accuracy by Class')
            plt.savefig(imgname+name+'Test Accuracy by Class.png')
            plt.savefig(imgname+name + 'Test Accuracy by Class.pdf')
            plt.show()
        boxmap(PublicTestloader,'PublicTest')
        boxmap(PrivateTestloader,'PrivateTest')
    epoch = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(8, 6))
    # Plotting the loss
    plt.plot(epoch, train_loss_history, label='Training Loss', color="red")
    plt.plot(epoch, test_loss_history1, label='Public Test Loss', color="green")
    plt.plot(epoch, test_loss_history2, label='Private Test Loss', color="blue")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    # Plotting the accuracy
    plt.savefig('loss' + imgname + ".png")
    plt.savefig('loss' + imgname + ".pdf")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epoch, train_acc_history, label='Training Accuracy', color="red")
    plt.plot(epoch, test_acc_history1, label='Public Test Accuracy', color="green")
    plt.plot(epoch, test_acc_history2, label='Private Test Accuracy', color="blue")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    # Adjusting the layout
    plt.savefig('Accuracy' + imgname + ".png")
    plt.savefig('Accuracy' + imgname + ".pdf")
    # Display the plot
    plt.show()
def plot(train_loss_history, train_acc_history, test_loss_history1, test_loss_history2, test_acc_history1,test_acc_history2,imgname):
    print(train_loss_history, train_acc_history, test_loss_history1, test_loss_history2, test_acc_history1,test_acc_history2)
    epoch = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(8, 6))
    # Plotting the loss
    plt.plot(epoch, train_loss_history, label='Training Loss', color="red")
    plt.plot(epoch, test_loss_history1, label='Public Test Loss', color="green")
    plt.plot(epoch, test_loss_history2, label='Private Test Loss', color="blue")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title( imgname + 'Training and Test Loss')
    plt.legend()
    # Plotting the accuracy
    plt.savefig('loss' + imgname + ".png")
    plt.savefig('loss' + imgname + ".pdf")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epoch, train_acc_history, label='Training Accuracy', color="red")
    plt.plot(epoch, test_acc_history1, label='Public Test Accuracy', color="green")
    plt.plot(epoch, test_acc_history2, label='Private Test Accuracy', color="blue")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title( imgname + 'Training and Test Accuracy')
    plt.legend()
    # Adjusting the layout
    plt.savefig('Accuracy' + imgname + ".png")
    plt.savefig('Accuracy' + imgname + ".pdf")
    # Display the plot
    plt.show()








    # ***********
if __name__=="__main__":
    Easy_CNN = Easy_CNN()
    resnet18=models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    efficientnet_b5 = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
    vit_b_16 = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # 选择冻结预训练模型的参数
    # ***********************************
    for param in resnet18.parameters():
        param.requires_grad = True
    resnet18.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(512, 7, bias=True))  # 修改分类层的输出维度为7,添加Dropout(0.2)
    for param in resnet18.fc.parameters():
        param.requires_grad = True
    # ************************************
    # ***********************************
    for param in resnet34.parameters():
        param.requires_grad = True
    resnet34.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(512, 7, bias=True))  # 修改分类层的输出维度为7
    for param in resnet34.fc.parameters():
        param.requires_grad = True
    # ************************************
    for param in resnet50.parameters():
        param.requires_grad = True
    resnet50.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(2048, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(512, 7, bias=True))  # 修改分类层的输出维度为7
    for param in resnet50.fc.parameters():
        param.requires_grad = True
    # *************************************
    for param in efficientnet_b5.parameters():
        param.requires_grad = True
    efficientnet_b5.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                               nn.Linear(in_features=2048, out_features=512, bias=True),
                                               nn.BatchNorm1d(512),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(p=0.5, inplace=False),
                                               nn.Linear(in_features=512, out_features=7, bias=True))  # 修改分类层的输出维度为7
    for param in efficientnet_b5.classifier.parameters():
        param.requires_grad = True
    # *************************************
    for param in vit_b_16.parameters():
        param.requires_grad = True
    vit_b_16.heads = nn.Sequential(nn.Dropout(0.2),nn.Linear(768, 256, bias=True), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                                   nn.Linear(256, 7, bias=True))  # 修改分类层的输出维度为7
    for param in vit_b_16.heads.parameters():
        param.requires_grad = True
    # *************************************
    for param in alexnet.parameters():
        param.requires_grad = False
    alexnet.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                       nn.Linear(in_features=9216, out_features=4096, bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=0.5, inplace=False),
                                       nn.Linear(in_features=4096, out_features=512, bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=0.5, inplace=False),
                                       nn.Linear(in_features=512, out_features=7, bias=True))  # 修改分类层的输出维度为7
    for param in alexnet.classifier.parameters():
        param.requires_grad = True
    device = torch.device("cuda")
    # 加载预训练模型
    # model = models.resnet18(pretrained=True)
    # model=models.densenet121(pretrained=True)
    # model = model
    # print(model)
    # # 冻结预训练模型的所有参数
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.classifier = nn.Sequential(nn.Linear(1024, 128),nn.Dropout(0.125),nn.Linear(128,7))  # 修改分类层的输出维度为7
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
    #train_loop(Easy_CNN,imgname='Easy_CNN',epochs=30,learnrate=0.01)
    #train_loop(resnet18, imgname='resnet18',epochs=10,learnrate=0.0001)
    #train_loop(alexnet, imgname='alexnet',epochs=20,learnrate=0.0001)
    #train_loop(resnet34, imgname='resnet34',epochs=10,learnrate=0.0001)
    train_loop(resnet50, imgname='resnet50',epochs=10,learnrate=0.0001)
    #train_loop(vit_b_16, imgname='vit_b_16',epochs=10,learnrate=0.0001)
    #train_loop(efficientnet_b5, imgname='efficientnet_b5',epochs=10,learnrate=0.0001)