import torch
import torch.nn as nn
import torch.nn.functional as F



#----------------------------------#
# dice loss
#----------------------------------#
def Dice_loss(pre,tar):
    assert pre.shape == tar.shape,"Compute dice loss error!"

    pre = F.softmax(pre,dim = 1)

    union = (pre * tar).sum()

    A1 = pre.sum()
    A2 = tar.sum()

    loss = union/(A1 + A2 - union + 1e-5)
    return 1 - loss.mean()


#----------------------------------#
# focal loss
#----------------------------------#
def focal_loss(pre,tar,alpha = 0.25,bate = 2.):

    assert pre.shape == tar.shape, "Compute focal loss error!"

    b,c,h,w = pre.shape
    if c <3:
        pre = F.sigmoid(pre)
    else:
        pre = F.softmax(pre)

    obj = (tar != 0 ).float()
    noobj = (tar == 0).float()

    obj_loss = (-1) * alpha * (1 - pre)**bate*torch.log(pre) * obj

    noobj_loss = (-1) * (1 - alpha)*pre**bate*torch.log(1 - pre) * noobj

    loss = obj_loss + noobj_loss

    return loss.mean()

#-------------------------------------------#
# score
#-------------------------------------------#
def score(pre,tar,thes = 0.5):
    assert pre.shape == tar.shape,"Compute dice loss error!"
    pre = F.softmax(pre,dim = 1)
    #------------------------------#
    #
    #------------------------------#
    pre = torch.ge(pre,thes)

    union = (pre * tar).sum()

    A1 = pre.sum()
    A2 = tar.sum()

    cost = union/(A1 + A2 - union + 1e-5)
    return cost.mean()


class PixContrasrt(nn.Module):
    def __init__(self,
                 thresh = 5,
                 max_epoch = None):
        super(PixContrasrt, self).__init__()

        self.dist = nn.Parameter(torch.tensor(0.2,
                                              requires_grad=True))
        self.thresh = thresh

        self.max_epoch = max_epoch

    def forward(self, feature, label,iteration):

        m = 1 - (iteration / self.max_epoch)

        b,*_ = feature.shape
        label = torch.unsqueeze(label,dim=1)


        # 前景激活层
        masked = feature * label

        # 背景激活层
        unmasked = feature * (1 - label)

        #余弦对抗
        # s_neg = 0.5 * nn.CosineSimilarity()(masked,(unmasked ))
        # #loss = -torch.log2((s_neg + 1e-8))
        # loss = s_neg.mean()
        #
        # return loss

        # 以余弦距离来构建三元组损失

        undist = nn.CosineSimilarity()(label, unmasked).mean()

        maskdist = nn.CosineSimilarity()(label, masked).mean()



        distloss = torch.where(
            undist > maskdist + self.dist,
            1.5 * undist + 0.8 * maskdist,
            undist + 0.5 * maskdist
        )



        return torch.where( torch.tensor(iteration) < self.thresh,
                    distloss,
                    (m)/(1 - m + 1e-5) * distloss)



class Heatmaploss(nn.Module):
    	pass
