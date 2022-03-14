import os
import numpy as np
import torch
from prsd.segmentation import UNet,ResUnet
from sklearn.metrics import classification_report,cohen_kappa_score
from prsd import io

# 标签渲染
def label2rgb(label):
    # 设置标签颜色
    # colors = [[0,0,0],[255,0,0],[0,255,0],[255,255,0],[0,0,255],[0,255,255]] # GF
    colors = [[0,0,0],[0,0,255],[0,255,255],[255,255,0],[0,255,0],[255,0,0]] #SD
    #colors = [[0,0,0],[255,255,0],[255,0,0],[0,255,0],[0,0,255],[255,255,255],[0,255,255]] 
    #colors = [[255,255,255],[184,40,99],[74,77,145],[35,102,193],[238,110,105],[117,249,76],[114,251,253],[126,196,59],[234,65,247],[90,196,111]]#UP
    # colors = [[255,255,255],[184,40,99],[74,77,145],[35,102,193],[238,110,105],[117,249,76],[114,251,253],[126,196,59],[234,65,247],[90,196,111],[255,235,205],[227,23,13],[255,192,203],[255,0,255],[199,97,20],[153,51,250],[255,255,0]]

    h,w = label.shape
    label_rgb = np.zeros((h,w,3))
    for i,rgb in zip(range(len(colors)),colors):
        label_rgb[label==i] = rgb
    return label_rgb

if __name__ == '__main__':
    dir_imgs = "data/test/imgs/"
    dir_mask = "data/test/gts/"
    imgs = sorted(os.listdir(dir_imgs))#[::4]
    masks = sorted(os.listdir(dir_mask))#[::4]
    target_names =["1","2","3","4","5","6"]
    device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else
    
    # 模型构建
    n_channels = 3 #图像通道数
    n_classes = 6 #地物类别数
    #model = UNet(n_channels, n_classes)
    model = ResUnet(n_channels, n_classes,filters=[16, 32, 64, 128])#.to(device=device)
    model.load_state_dict(torch.load('model_g/'+'23_0.8530036465875034_best_model.pt'))
    
    preds = []
    labels = []
    model.eval()
    for i, fn in enumerate(imgs):
        print(i)
        img_file = os.path.join(dir_imgs,imgs[i])
        mask_file = os.path.join(dir_mask,masks[i])
        img,_,_ = io.imread(img_file)
        mask,geotrans,proj = io.imread(mask_file)
        img = torch.from_numpy(img).float()#.to(device=device, dtype=torch.float32)
        #mask = torch.from_numpy(mask).float()#.to(device=device, dtype=torch.float32)
        #print(img.shape,mask.shape)
        pred = model(img.unsqueeze(0)).argmax(dim=1).squeeze(0).numpy()
        #print(pred)
        #print(mask)
        preds.append(pred)
        labels.append(mask)
        io.imsave("preds/"+fn,label2rgb(pred).transpose(2,0,1),geotrans,proj)
	#output = gdal_array.SaveArray(pred,'perds/'+fn,format="GTiff",prototype=fn)
    #preds = np.array(preds).flatten()
    #labels = np.array(labels).flatten()
    #print(classification_report(labels,preds,target_names=target_names,digits=4))
    #print('kappa:',cohen_kappa_score(labels,preds))
    
    


