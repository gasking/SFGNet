import random
import cv2
import numpy as np
import math

class RGB2HSV:
    def __call__(self,img):
     return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

class HSV2RGB:
    def __call__(self,img):
     return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


class Noramlize():
    def __init__(self):
        self.mean = [ 0.485, 0.456, 0.406 ]
        self.std = [ 0.229, 0.224, 0.225 ]
    def __call__(self,sampler):
        image = sampler['image']
        mask = sampler['mask']

        image = image.astype(np.float32)

        image = image / 255.
        # image -= self.mean
        # image /= self.std

        return {
            'image':image,
            'mask':mask
        }

class RandomHFlip():
    def __call__(self,sampler):
      image = sampler[ 'image' ]
      mask = sampler[ 'mask' ]
      if random.random() < 0.5:
        image = image[:,::-1,:]
        mask = mask[:,::-1]

      return {
            'image': image,
            'mask': mask
        }


class RandomVFlip():
    def __call__(self,sampler):
      image = sampler[ 'image' ]
      mask = sampler[ 'mask' ]
      if random.random() < 0.5:
        image = image[::-1,:,:]
        mask = mask[::-1,:]

      return {
            'image': image,
            'mask': mask
        }

class RandomRotate():
    def __init__(self,scale = 1,degree = 90):
        self.scale = scale
        self.degree = degree
    def __call__(self, sampler):
        image = sampler[ 'image' ]
        mask = sampler[ 'mask' ]
        if random.random() < 0.5:


            scale = random.uniform(0,self.scale)
            degree = random.uniform(-1*self.degree,self.degree)

            h,w,_ = image.shape
            cx,cy = (w//2,h//2)

            M = cv2.getRotationMatrix2D((cx,cy),degree,scale)

            image = cv2.warpAffine(image,M,(w,h),borderValue = (128,128,128))

            mask = cv2.warpAffine(mask,M,(w,h),borderValue = (0))

        return {
                'image': image,
                'mask': mask
            }


class RandomBrightness():
 def __call__(self, sampler):
     image = sampler[ 'image' ]
     mask = sampler[ 'mask' ]

     if random.random() < 0.5:


        hsv = RGB2HSV()(image)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([ -1.5, 1.5 ])
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        image = HSV2RGB()(hsv)

     return {
            'image': image,
            'mask': mask
        }



class RandomSaturation():
 def __call__(self, sampler):

     image = sampler[ 'image' ]
     mask = sampler[ 'mask' ]
     if random.random() < 0.5:
         hsv = RGB2HSV()(image)
         h, s, v = cv2.split(hsv)
         adjust = random.choice([ -1.5, 1.5 ])
         s = s * adjust
         s = np.clip(s, 0, 255).astype(hsv.dtype)
         hsv = cv2.merge((h, s, v))
         image = HSV2RGB()(hsv)

     return {
         'image': image,
         'mask': mask
     }


class RandomHue():
 def __call__(self, sampler):
     image = sampler[ 'image' ]
     mask = sampler[ 'mask' ]
     if random.random() < 0.5:
         hsv = RGB2HSV()(image)
         h, s, v = cv2.split(hsv)
         adjust = random.choice([ 0.5, 1.5 ])
         h = h * adjust
         h = np.clip(h, 0, 255).astype(hsv.dtype)
         hsv = cv2.merge((h, s, v))
         image = HSV2RGB()(hsv)

     return {
         'image': image,
         'mask': mask
     }



class RandomBlur():
 def __call__(self, sampler):
     image = sampler[ 'image' ]
     mask = sampler[ 'mask' ]
     if random.random() < 0.5:
         image = cv2.GaussianBlur(image,ksize = (3,3),sigmaX = 1.5)

     return {
         'image': image,
         'mask': mask
     }


class Data_Augment():
    def __init__(self, para):
        self.compose = para
    def __call__(self,img):
        for x in self.compose:
            img = x(img)
        return img

if __name__ == '__main__':
    image = cv2.imread(r'C:\Users\gasking\Desktop\Unet\VOCdevkit\VOC2023\Images\00079.jpg')

    D = Data_Augment(*[RandomHue(),
                      RandomSaturation(),
                      RandomBrightness(),
                      RandomBlur(),
                      ])
    shape = image.shape
    print(shape)
    if len(shape)>2:
        h,w,_ = shape
    else:h,w = shape
    degree = 50

    M = cv2.getRotationMatrix2D((w//2,h//2),degree,scale = 1)

    wrap_fill = cv2.warpAffine(image,M,[w,h],borderValue = (128,128,128))
    print(wrap_fill.shape)
    img = D({'image':image,
             'mask':image})
    wrap_fill = cv2.GaussianBlur(wrap_fill, ksize = (3,3), sigmaX = 1.5)
    cv2.imshow('1',wrap_fill)

    #cv2.imshow('2',img)
    cv2.waitKey(0)