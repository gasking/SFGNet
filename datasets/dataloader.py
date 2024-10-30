import torch
import numpy as np
import cv2
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image  # 这个读取数据是RGB
from torch.utils.data import DataLoader, Dataset
import random
from utils.tools import resize

from datasets.transformer import *
from datasets.boundary import convert_boundary


class BuildDataset(Dataset):
    def __init__(self, input_shape = (512, 512),
                 train_file = None,
                 num_classes = 1 + 1,
                 augment = False #训练 True
                 ):

        self.image_size = input_shape

        self.num_class = num_classes #类别

        self.train = []
        self.test = []
        self.val = []

        trainlines = open(train_file)


        if augment: #推理
            self.data_aug = Data_Augment([RandomHue(),
                          #RandomSaturation(),
                          #RandomBrightness(),
                          RandomHFlip(),
                          RandomVFlip(),
                         #  RandomBlur(),
                         # RandomRotate(),
                          Noramlize()        ])

        else:
            self.data_aug = Data_Augment([

                Noramlize()])

        for line in trainlines.readlines():
            splited = line.strip().split()
            self.train.append([splited[ 0 ],splited[1]])


        trainlines.close()



    def handler(self,fname,maskname):
        image = Image.open(fname).convert('RGB')

        mask = Image.open(maskname).convert('L')

        w,h = image.size #TODO

        assert np.array(image.size).all() == np.array(mask.size).all(), "Not Match! Error"

        patch_h = random.randint(0,h - self.image_size[0] - 1)
        patch_w = random.randint(0,w - self.image_size[1] - 1)


        image = image.crop((patch_w,patch_h ,patch_w + self.image_size[1],patch_h + self.image_size[0]))
        mask = mask.crop((patch_w, patch_h , patch_w + self.image_size[1], patch_h + self.image_size[ 0 ]))


        if random.uniform(0,1) > 0.5:
            w, h = image.size

            scalew, scaleh = self.image_size[ 0 ] / w, self.image_size[ 1 ] / h
            scale = min(scaleh, scalew)
            neww, newh = int(scale * w), int(scale * h)

            dx = self.image_size[ 0 ] - neww
            dy = self.image_size[ 1 ] - newh

            # 128 过于接近云
            new_image = Image.new("RGB", self.image_size, (0, 0, 0))
            image = image.resize((neww, newh))

            new_mask = Image.new("L", self.image_size, (0))
            mask = mask.resize((neww, newh))

            if random.uniform(0, 1) > 0.5:
                dx //= 2
                dy //= 2

            new_image.paste(image, (dx, dy))
            new_mask.paste(mask, (dx, dy))
        else :
            new_image = image.resize(self.image_size)
            new_mask = mask.resize(self.image_size)



        image = np.array(new_image, dtype = np.float32)


        # t = image.copy()
        # t = t.astype(np.uint8)
        # crop = image[dy:dy+newh,dx:dx+neww,:]
        # crop = crop.astype(np.uint8)
        # #crop = cv2.resize(crop,(ow,oh))
        # print(crop.shape)
        # cv2.imshow('im', t)
        # cv2.imshow('crop',crop)
        # cv2.waitKey(0)

        # ----------------------------------#
        # 二分类
        # ----------------------------------#
        mask = np.array(new_mask)  # 单通道图像

        result = self.data_aug({
            'image': image,
            'mask': mask
        })

        image = result[ 'image' ]
        mask = result[ 'mask' ]

        modify_png = np.zeros_like(mask)
        modify_png[ mask > 180 ] = 1

        heat2, boundary2 = convert_boundary(mask, 2)
        heat4, boundary4 = convert_boundary(mask, 4)
        heat8, boundary8 = convert_boundary(mask, 8)
        heat16, boundary16 = convert_boundary(mask, 16)

        # -------------------------------#
        # 多分类
        # -------------------------------#
        # for c in range(self.num_class):
        #     mask[mask==c] = c

        T_mask = np.zeros((self.image_size[ 1 ], self.image_size[ 0 ], self.num_class))

        # --------------------------------------#
        # 两种构建one-hot编码形式
        # --------------------------------------#
        for c in range(self.num_class):
            T_mask[ modify_png == c, c ] = 1
        T_mask = np.transpose(T_mask, (2, 0, 1))
        # T_mask = np.eye(self.num_class)[mask.reshape(-1)] #
        # T_mask = np.reshape(T_mask,(self.image_size[1],self.image_size[0],self.num_class))

        # vision
        """
        back = T_mask[0,...]
        fg = T_mask[1,...]
        cv2.imshow('im',image.astype(np.uint8))
        cv2.imshow('bg',back)
        cv2.imshow('fg',fg)
        cv2.waitKey(0)
        """

        img = np.transpose(image, (2, 0, 1))
        heat2 = np.transpose(heat2,(2,0,1))
        heat4 = np.transpose(heat4, (2, 0, 1))
        boundary2 = np.transpose(boundary2, (2, 0, 1))
        boundary4 = np.transpose(boundary4, (2, 0, 1))
        boundary8 = np.transpose(boundary8, (2, 0, 1))
        boundary16 = np.transpose(boundary16, (2, 0, 1))


        return img,T_mask,modify_png,\
               (heat2,heat4),(boundary2,boundary4,boundary8,boundary16)

    def __getitem__(self, idx):
        ind1 = idx % len(self.train)

        imagename = self.train[ind1][0]
        maskname = self.train[ind1][1]

        image,label,png,(heat2,heat4),\
        (boundary2,boundary4,boundary8,boundary16) = \
            self.handler(imagename,maskname)

        return image,label,png,(heat2,heat4),\
        (boundary2,boundary4,boundary8,boundary16)

    def __len__(self):
        return len(self.train)

    # ------------数据增强-----------------#
    def _norm(self, img):
        img = img/255.
        # img -= self.mean
        # img /= self.std
        return img


def convert(data):
    gtlables = np.array(data, np.float32)
    gtlables = torch.from_numpy(gtlables).float()

    return gtlables


def collate_seg(batch):

     gtimages,gtlables,gtpng,heat2s,\
     heat4s,boundary2s,boundary4s,boundary8s,boundary16s= \
         [],[],[],[],[],[],[],[],[]

     for image,label,png,(heat2,heat4),(boundary2,boundary4,boundary8,boundary16) in batch:

         gtimages.append(image)
         gtlables.append(label)
         gtpng.append(png)

         #loc loss
         heat2s.append(heat2)
         heat4s.append(heat4)

         # boundary loss
         boundary2s.append(boundary2)
         boundary4s.append(boundary4)
         boundary8s.append(boundary8)
         boundary16s.append(boundary16)

     # image
     gtimages = convert(gtimages)

     # label
     gtlables = convert(gtlables)

     # one-hot
     gtpng = np.array(gtpng,np.long)
     gtpng = torch.from_numpy(gtpng).long()

     # heatmap loc
     heat2s = convert(heat2s)
     heat4s = convert(heat4s)

     # boundary
     boundary2s = convert(boundary2s)
     boundary4s = convert(boundary4s)
     boundary8s = convert(boundary8s)
     boundary16s = convert(boundary16s)


     return gtimages,gtlables,gtpng,(heat2s,heat4s),\
            (boundary2s,boundary4s,boundary8s,boundary16s)


if __name__ == '__main__':
 def gt():
    data = BuildDataset(train_file = '../val.txt')
    train_loader = DataLoader(data, batch_size = 8, shuffle = True,collate_fn = collate_seg)
    #train_iter = iter(train_loader)
    for i,batch in enumerate(train_loader):

        image, label, png, (heat2, heat4), \
        (boundary2, boundary4, boundary8, boundary16) = batch

        print(image.shape,label.shape,png.shape,png.dtype,boundary2.shape)


        for ind in range(image.shape[0]):

         sim = image[ind].numpy()
         sim = np.transpose(sim,(1,2,0)) * 255.
         sim = sim.astype(np.uint8)[...,::-1]

         # 热力图
         h2 = heat2[ind].numpy()
         h2 = np.transpose(h2,(1,2,0)) * 255.
         h2 = h2.astype(np.uint8)

         # 边界
         boundary2ss = boundary2[ ind ].numpy()
         boundary2ss = np.transpose(boundary2ss, (1, 2, 0)) * 255.
         boundary2ss = boundary2ss.astype(np.uint8)

         for c in range(label.shape[1]):

             T = label[ind].numpy()

             T = np.transpose(T,(1,2,0))

             cv2.imshow(f"im_{c}",T[...,c])


         cv2.imshow('h2',h2)
         cv2.imshow('b2',boundary2ss)
         cv2.imshow('sim',sim)

         cv2.waitKey(0)


 gt()


