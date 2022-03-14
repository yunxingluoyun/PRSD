# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 15:41:11 2021

@author: 陨星落云
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from prsd.utils import BasicDataset
from prsd.segmentation import UNet,ResUnet,ResUnetPlusPlus
# from prsd.io import imread

def model_val(model,data_loader):
    # 验证精度
    model.eval()
    accuracy = 0
    total = 0 
    for batch in data_loader:

        # get the inputs and wrap in Variable
        imgs= batch['image']
        true_masks= batch['mask']
        #print(true_masks)
        inputs = imgs.to(device=device, dtype=torch.float32)
        labels = true_masks.to(device=device, dtype=torch.float32)

        # forward
        outputs = model(inputs).argmax(dim=1)
        
        # 精度
        accuracy += torch.sum(outputs==labels).item()
        total += labels.shape[0]
        
    return accuracy/(total*256*256)

def model_train(model,num_epochs,device,train_loader,val_loader,criterion,optimizer):
    # 训练
    #num_epochs = 10
    model.train()
    val_acc_max = 0
    for epoch in range(0, num_epochs):
        accuracy = 0
        total = 0 
        for i,batch in enumerate(train_loader):

            # get the inputs and wrap in Variable
            imgs= batch['image']
            true_masks= batch['mask']
            #print(true_masks.shape)
            inputs = imgs.to(device=device, dtype=torch.float32)
            labels = true_masks.to(device=device, dtype=torch.long)
 
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            #print(outputs.shape)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()
            
            # 训练精度
            accuracy += torch.sum(outputs.argmax(dim=1)==labels).item()
            total += labels.shape[0]
            
            if i%15==0:
                print('Epoch: {} \tTraining Loss: {:.6f} \tTraining accuracy: {:.6f} '.format(epoch, loss,accuracy/(total*256*256)))
        
        val_acc = model_val(model,val_loader)
        print('Epoch: {} \tVal accuracy: {:.6f} '.format(epoch,val_acc))
        
        if val_acc>=val_acc_max:
            print('Vaildation acc increased({:.06f} --> {:.6f}).  Saving model .....'.format(val_acc_max,val_acc))
            torch.save(model.state_dict(),"model_g/"+str(epoch)+'_'+str(val_acc)+'_best_model.pt')
            #val_acc_max  = val_acc
            
    return None


if __name__ == '__main__':
    dir_imgs = "data/train_val/imgs/"
    dir_mask = "data/train_val/gts/"
    # 构建训练与验证集
    dataset = BasicDataset(dir_imgs, dir_mask)
    val_percent = 0.25
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # print(len(dataset))
    
    # dataiter = iter(train_loader)
    # batch = dataiter.next()
    # imgs = batch['image']
    # labels = batch['mask']
    # print(imgs,labels.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')#
    #print(device)
    
    # 模型构建
    n_channels = 3 #图像通道数
    n_classes = 6 #地物类别数
    #model = UNet(n_channels, n_classes).to(device=device)
    
    #model = ResUnetPlusPlus(n_channels, n_classes,filters=[16, 32, 32, 64, 128]).to(device=device)
    model = ResUnet(n_channels, n_classes,filters=[16, 32, 64, 128]).to(device=device)
    print(model)
    
    #损失与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs =100
    model_train(model,num_epochs,device,train_loader,val_loader,criterion,optimizer)
    

    
    

    
    
