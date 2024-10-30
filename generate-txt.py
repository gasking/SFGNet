from glob import glob
import os
import random



# -------------------------------------------------------#
#  生成训练文件,文件以此格式存放
#  VOCdevkit
#    VOC2023
#      Annotations #标签文件
#      JPEGImages  #原始图像
#      MASKImages  #掩码图像

def Generate(JPG = None,
             MASK = None,
             mode = None,
             suffix = '.tif'
             ):

    # 训练文件
    traintxt = f'{mode}.txt'

    images = glob(os.path.join(JPG,'*'+'.tiff'))
    labels = glob(os.path.join(MASK,'*'+suffix))

    assert len(images)==len(labels),'train image not equal train label'

    traintxt = open(traintxt,'w+')

    for data in images:
        traintxt.write(data+' '+data.replace('.tiff','.tif').replace('val','val_labels')+'\n')

    traintxt.close()


if __name__ == '__main__':

    # 训练
    Generate(JPG = r'C:\Users\gasking\Desktop\轻量化目标检测\数据集\Massachusetts_Buildings_Dataset_datasets\Massachusetts_Buildings_Dataset_tiff_datasets\val',
             MASK = r'C:\Users\gasking\Desktop\轻量化目标检测\数据集\Massachusetts_Buildings_Dataset_datasets\Massachusetts_Buildings_Dataset_tiff_datasets\val_labels',
             mode = 'val')

    # 测试
    #Generate()

    # 验证
    #Generate()