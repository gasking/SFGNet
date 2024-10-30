import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from math import pi,cos
import random
from copy import  deepcopy

#----------------------------------#
# 固定随机参数
#----------------------------------#
def fix_seed(seed = 12):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prob2entropy(feature):
    """feature map convert to weighted entrophy"""

    b,c,h,w = feature.shape

    x = -torch.mul(feature,torch.log2(feature + 1e-20)) / np.log2(c)

    return x



#----------------------------------#
# 权重初始化
#----------------------------------#
def weight__init(net,init_type = 'kaiming',gain = 0.02):
    def __init__par(func):
        m = func.__class__.__name__
        if isinstance(m,nn.Conv2d):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data,0.0,gain = gain)
                    #nn.init.normal_(m.bias.data,0.01)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data,a = 0, mode = 'fan_in')
                    #nn.init.kaiming_normal_(m.bias.data, 0.01)
                else:
                    raise TypeError("NOT init weight Type")
        elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight.data,1.0,gain)
                nn.init.constant_(m.bias.data,0.0)
    print("Parameters Init....")
    return net.apply(__init__par)


def resize(image,input_shape):

    w,h = image.size

    scalew, scaleh = input_shape[ 1 ] / w, input_shape[0] / h
    scale = min(scaleh, scalew)

    neww, newh = int(scale * w), int(scale * h)


    image = image.resize((neww,newh))
    new_image  = Image.new("RGB",input_shape,(0,0,0))
    dx = (input_shape[1] - neww)//2
    dy = (input_shape[0] - newh)//2
    new_image.paste(image,(dx,dy))

    return new_image,neww, newh ,dx,dy,w,h

#-------------------------------#
# Get lr
#-------------------------------#
def get_lr(optimizer):
    for para in optimizer.param_groups:
        return para['lr']

#------------------------------#
# 余弦退火法
#------------------------------#
def warm_up(optimizer,max_iters,cur_iters,min_lr,max_lr,warm_iter):

    feature = optimizer
    if cur_iters < warm_iter:
        # warmup_lr_start = 5e-4
        # lr = 6.5e-4

        warmup_lr_start = 6e-4
        lr = 8e-4

        lr = (lr - warmup_lr_start) * pow(cur_iters / float(warm_iter), 2) + warmup_lr_start

    else:
        lr = min_lr + 0.5*(max_lr - min_lr)*( 1 + cos((cur_iters / max_iters) * pi))

    set_lr(feature,lr)

    # IF
    # set_lr(domain_aux,lr )
    # set_lr(domain_main,lr)

#---------------------------------#
# 设置学习率
#---------------------------------#
def set_lr(optimizer,lr):
    for para in optimizer.param_groups:
        para['lr'] = lr


def _norm(img):
    img /= 255.
    # img -= np.array(Config['mean'],dtype = np.float32)
    # img /= np.array(Config['std'],dtype = np.float32)
    return img



#-------------------------------------------#
#            TODO self supervised
#-------------------------------------------#
class ModelEMA(object):
    def __init__(self,
                 model,
                 decay = 0.995,
                 updates = 0):
        # 复制参数
        # 用于模型评估
        self.ema = deepcopy(model).eval() #进入评估模式

        self.updates = updates

        #self.decay = lambda x:decay* ( math.exp( -x / 1000000.))

        self.decay = decay

        for para in self.ema.parameters():
            para.requires_grad = False


    def update(self,model):

        with torch.no_grad():
            self.updates += 1

            d = self.decay


            # 获取模型的state_dict
            # 如果使用了多卡 model.module.state_dict()
            oldmodel = model.module.state_dict()

            for k,v in self.ema.state_dict().items():
             if v.dtype.is_floating_point:
                v *= d

                v += (1. - d) * oldmodel[k].detach()