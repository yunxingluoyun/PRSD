import shutil
import numpy as np
import os 

def copyimgs(sourcePath,targetPath,imglist):
    all_imgs = sorted(os.listdir(sourcePath))

    for i in imglist:
        objName = all_imgs[i] 
        shutil.copy(sourcePath + '/' + objName, targetPath + '/' + objName)
 
if __name__ == '__main__':
    # 复制影像的文件夹
    sourceimgsPath = 'high_quality_dataset256/img'
    sourcegtsPath = 'high_quality_dataset256/gt'

    num_imgs = len(os.listdir(sourceimgsPath))
    img_list = np.arange(num_imgs)
    np.random.shuffle(img_list)
    print(img_list)
    train_imgList = img_list[:int(num_imgs*0.8)]
    test_imgList = img_list[int(num_imgs*0.8):]
    print(np.sort(train_imgList),np.sort(test_imgList))
    
    # train文件夹
    train_imgstargetPath = 'data/train_val/imgs'
    copyimgs(sourceimgsPath,train_imgstargetPath,train_imgList)

    train_gtstargetPath = 'data/train_val/gts'
    copyimgs(sourcegtsPath,train_gtstargetPath,train_imgList)

    # test文件夹
    test_imgstargetPath = 'data/test/imgs'
    copyimgs(sourceimgsPath,test_imgstargetPath,test_imgList)

    test_gtstargetPath = 'data/test/gts'
    copyimgs(sourcegtsPath,test_gtstargetPath,test_imgList) 
    
