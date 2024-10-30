from tqdm import tqdm
import torch
import numpy as np
from nets.criterion import Dice_loss,score,\
     Heatmaploss,focal_loss
import torch.nn as nn
from utils.tools import get_lr
from utils.tools import prob2entropy
import torch.nn.functional as F
from utils.config import Config


def epoch_fit(cur_epoch,total_epoch,save_step,
              model,optimizer,dataloader,
              device,logger,args,save_path,
              eval,local_rank): #评估函数是对测试集



    model = model.to(device)

    if torch.cuda.device_count() > 1:
        #-----------------------------------------------------#
        #                  训练的BS要能被显卡数整除
        #-----------------------------------------------------#
        print(f'GPUS Numbers :{torch.cuda.device_count()}')

        #-------------------------------------------------------#
        #                         设置多卡训练
        #-------------------------------------------------------#
        model = nn.DataParallel(model)



    traindataloader,testdataloader,valdataloader = dataloader


    # 损失函数定义
    criteror = nn.BCEWithLogitsLoss()
    heatloss = Heatmaploss() # TODO ? loc loss ?

    with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
              total = len(traindataloader),mininterval = 0.3,postfix = dict,colour = '#6DB9EF') as pb:

        model = model.train()


        #-------------------------------------#
        # 训练损失计算
        #-------------------------------------#
        total_loss = 0.
        total_cls_loss = 0.
        total_boundary_loss = 0.
        total_heat_loss = 0.

        # 源域
        for ind,batch in enumerate(traindataloader):
          image,label,png,(heat2,heat4),(boundary2,boundary4,boundary8,boundary16) = batch
          with torch.no_grad():
              # TODO 1
              image = image.to(device)
              label = label.to(device)
              png = png.to(device)

              # TODO 2 heatmap
              heat2 = heat2.to(device)
              heat4 = heat4.to(device)

              # TODO 3 boundary
              boundary2 = boundary2.to(device)
              boundary4 = boundary4.to(device)
              boundary8 = boundary8.to(device)
              boundary16 = boundary16.to(device)

          #---------------------------------#
          #        源域特征提取 + 分割模型
          #---------------------------------#
          optimizer.zero_grad()

        


          pb.set_postfix(**{
              'total_loss': total_loss/(ind + 1),
              'total_cls_loss':total_cls_loss/(ind + 1),
              'total_boundary_loss':total_boundary_loss/(ind + 1),
              'total_heat_loss':total_heat_loss/(ind + 1)
          })


          pb.update(1)


    #----------------------------------------------------------------------------#
    # TODO freeze BN
    #----------------------------------------------------------------------------#
    model = model.eval()

    #Evaluatation
    with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
              total = len(valdataloader),mininterval = 0.3,postfix = dict,colour = '#7E89EF') as pb:

     with torch.no_grad():
        # TODO
        for ind,batch in enumerate(valdataloader):
            image,label,png,(_,_),*_ = batch

            image = image.to(device)

            png = png.to(device).detach().cpu().numpy()

            if args.method == 'Ours':
                (feat_out8, feat_out16, feat_out32), \
                (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
                (pred_loc2, pred_loc4) = model(image)

                OneHotlabel8 = F.softmax(feat_out8, dim=1)
                #OneHotlabel16 = F.softmax(feat_out16, dim = 1)
                #OneHotlabel32 = F.softmax(feat_out32, dim = 1)

                targetOneHotlabel8 = torch.argmax((OneHotlabel8), dim=1).detach().cpu().numpy()  # 目标域
                #targetOneHotlabel16 = torch.argmax((OneHotlabel16), dim = 1).detach().cpu().numpy()  # 目标域
                #targetOneHotlabel32 = torch.argmax((OneHotlabel32), dim = 1).detach().cpu().numpy()  # 目标域
                targetlabel = targetOneHotlabel8
                #targetlabel = (targetOneHotlabel8 | targetOneHotlabel16 | targetOneHotlabel32)

         

            eval.init(png,targetlabel)

            pb.update(1)

    # 评估结果
    eval.show()

    if ((cur_epoch + 1)%save_step == 0) :
            torch.save(model.module.state_dict(),f'{save_path}/{(cur_epoch + 1)}.pth')
    if (cur_epoch + 1) == total_epoch:
     torch.save(model.module.state_dict(),f'{save_path}/last.pth')
