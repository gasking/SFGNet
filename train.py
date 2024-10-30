import torch
import torch.nn as nn
# -------------------------------------#
import numpy as np
import torch.optim as optim
from utils.utils_fit import epoch_fit
from utils.logger import Logger
import random
import shutil, os
import platform

# ---------------------------------------------------------------#
#                       导入解释器
# ---------------------------------------------------------------#
import argparse

from utils.config import Config
from utils.tools import warm_up, weight__init, fix_seed
from utils.callback import History
from torch.utils.data import DistributedSampler
from datasets.dataloader import BuildDataset, collate_seg
from torch.utils.data import DataLoader
from utils.summary import summary

# -------------------------------------#
# 更换模型
# -------------------------------------#
from nets.BuildRT import BuildRT
# 评估函数
from utils.utils_eval import Eval

if __name__ == '__main__':

    parse_augment = argparse.ArgumentParser(description = 'UDA')

    # add baic augment

    parse_augment.add_argument('--init_epoch', type = int,
                               default = Config[ 'init_epoch' ], help = 'Max epoch...')

    parse_augment.add_argument('--freeze_epoch', type = int,
                               default = Config[ 'freeze_epoch' ], help = 'Max epoch...')

    parse_augment.add_argument('--unfreeze_epoch', type = int,
                               default = Config[ 'unfreeze_epoch' ], help = 'Max epoch...')
    # input_size
    parse_augment.add_argument('--input_shape', '--input', type = tuple,
                               default = Config[ 'input_shape' ], help = 'train image size')

    # batch size
    parse_augment.add_argument('--batch_size', '--bs', type = int,
                               default = Config[ 'bs' ], help = 'train batch size')

    parse_augment.add_argument('--warm_up', type = int, default = 15, help = 'warm_up paramters...')

    # 训练集
    parse_augment.add_argument('--train_txt', '--st', type = str, default = Config[ 'train_txt' ],
                               help = 'train file')

    # 测试集
    parse_augment.add_argument('--test_txt', '-tt', type = str, default = Config[ 'test_txt' ],
                               help = 'test file')

    # 验证集
    parse_augment.add_argument('--val_txt', '--sv', type = str, default = Config[ 'val_txt' ],
                               help = 'val file')
    # 方法
    parse_augment.add_argument('--method',type = str,default = 'ours',
                               required = True,help = 'Compare Method...')
    # 训练模式
    parse_augment.add_argument('--mode', '--md', type = str, default = 'co-train',
                               help = 'select train mode')

    # DDP train
    parse_augment.add_argument('--distributed','--dist',action = 'store_true',
                               default = False,help = 'DDP training...')

    # num_worker
    parse_augment.add_argument('--num_worker', '--nw', type = int, default = Config[ 'num_worker' ],
                               help = 'num_worker...')

    # decay rate
    parse_augment.add_argument('--decay_rate', '--d', type = float, default = Config[ 'decay_rate' ])

    # momentum
    parse_augment.add_argument('--momentum', '--m', type = float, default = Config[ 'momentum' ])

    # optimizer type
    parse_augment.add_argument('--optimizer_type', '--op', type = str, default = Config[ 'optimizer' ])

    # min lr
    parse_augment.add_argument('--min_lr', '--mlr', type = float, default = Config[ 'min_lr' ])

    #model_path
    parse_augment.add_argument('--model_path', '--model', type = str, default = Config[ 'model_path' ])

    # adam max lr
    parse_augment.add_argument('--adam_max_lr', '--adam_lr', type = float, default = Config[ 'adam_max_lr' ])

    # sgd max lr
    parse_augment.add_argument('--sgd_max_lr', '--sgd_lr', type = float, default = Config[ 'sgd_max_lr' ])

    # seed
    parse_augment.add_argument('--seed', '--s', type = int, default = 1234)

    # save step
    parse_augment.add_argument('--save_step', '--save', type = int, default = Config[ 'save_step' ],
                               help = 'save model step')

    # accumulate
    parse_augment.add_argument('--accumulate', '--ac', type = int,
                               default = 4, help = 'accumulate grad...')
    # class
    parse_augment.add_argument('--classes','--c',type = tuple,default = Config['_Class'])
    # 解释
    args = parse_augment.parse_args()

    # -------------------------------------------------------------------------------#
    #                              多卡训练
    # -------------------------------------------------------------------------------#
    platformer = platform.system()

    if args.distributed:
        backendsname = 'gloo'if platformer == 'Windows' else 'nccl'

        #初始化后端
        torch.distributed.init_process_group(backend = backendsname)
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0 #设置



    # ------------------------------------------------------------------------------#
    #                               参数赋值
    # ------------------------------------------------------------------------------#
    init_epoch = args.init_epoch
    freeze_epoch = args.freeze_epoch
    unfreeze_epoch = args.unfreeze_epoch

    num_worker = args.num_worker

    freeze_batch = args.batch_size

    input_shape = args.input_shape

    # ---------------------------------------------------#
    #                 训练标签
    # ---------------------------------------------------#
    train_txt = args.train_txt
    test_txt = args.test_txt
    val_txt = args.val_txt


    class_num = len(args.classes) + 1

    model_path = args.model_path

    save_step = args.save_step

    min_lr = args.min_lr

    warm_iter = args.warm_up

    # ------------------------------#
    # 优化器种类
    # ------------------------------#
    optimizer_type = args.optimizer_type

    max_lr = args.adam_max_lr if optimizer_type == 'adam' else args.sgd_max_lr

    decay_rate = args.decay_rate
    momentum = args.momentum

    # fix seed
    seed = args.seed

    fix_seed(seed)

    # Evaluate
    eval = Eval(class_num = class_num)

    # ---------------------------------#
    #            权重保存路径
    # ---------------------------------#
    # 对比方法
    method = args.method
    save_path = f'Pth/{method}'

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    #设置多卡
    device = torch.device('cuda',local_rank) if torch.cuda.is_available() else torch.device('cpu')

    # -------------------------------------------#
    #                初始化模型
    # -------------------------------------------#

    if (method == 'Ours'):
     model = BuildRT(n_classes = class_num)

    elif method == 'bisenetv1':
     model = bisenetv1.BiSeNetV1(class_num)

    elif method == 'bisenetv2':
     model = bisenetv2.BiSeNetV2(class_num)

    elif method == 'pidnet':
     model = pidnet.get_pred_model(name='pidnet_s', num_classes=class_num)

    elif method == 'ddrnet':
     model = get_seg_model(num_classes = class_num)

    elif method == 'ffnet':
     model = segmentation_ffnet50_dAAA()

    elif method == 'stdc':
     model = model_stage.BiSeNet('STDCNet813', class_num,use_boundary_8 = True)

    elif method == 'lpsnet':
      model = get_lspnet_s()

    else:
        raise  NotImplementedError('No exist Method!!!')

    # -------------------------------------------#
    # 初始化日志器
    # -------------------------------------------#
    logger = Logger(file_path = os.path.join(save_path, 'looger'))

    model_path = args.model_path

    if os.path.exists(model_path) & (model_path != ''):
        miss = [ ]  # 缺少模块
        exist = [ ]  # 存在模块
        net = model.state_dict()  # 当前模型的图结构

        pretrainnet = torch.load(model_path, device)  # 预训练结构

        temp = {}  # 收集共有层
        for k, v in pretrainnet.items():

            if (k in net.keys()) & (np.shape(net[ k ]) == np.shape(v)):
                temp[ k ] = v
                exist.append(k)
            else:
                miss.append(k)
        net.update(temp)
        # 导入
        model.load_state_dict(net)

    if not os.path.exists(model_path):
        weight__init(model)  # 初始化网络参数
        logger.info('没有导入预训练权重，初始化网络参数')

    # --------------------------------------------#
    # 优化器
    # --------------------------------------------#

    modeloptimizer = {
         'adam': optim.Adam(model.parameters(),lr = max_lr,weight_decay = decay_rate),
         'sgd': optim.SGD(model.parameters(),lr = max_lr,momentum = momentum,weight_decay = decay_rate )
    }[optimizer_type]

    # modeloptimizer = optim.SGD(model.optim_parameters(2.5e-4), lr = 2.5e-4, momentum = 0.9,
    #                              weight_decay = 0.0005)


    # --------------------------------------------#
    # 数据加载
    # --------------------------------------------#
    trainseg = BuildDataset(input_shape = input_shape, train_file = train_txt,num_classes = class_num)

    trainSegData = DataLoader(trainseg, batch_size = freeze_batch, num_workers = num_worker, collate_fn = collate_seg,
                              shuffle = True,pin_memory = True)

    # 测试集
    testseg = BuildDataset(input_shape = input_shape, train_file = test_txt,
                         num_classes = class_num)
    testSegData = DataLoader(testseg, batch_size = freeze_batch, num_workers = num_worker, collate_fn = collate_seg,
                            shuffle = False)

    # 验证集
    valseg = BuildDataset(input_shape = input_shape, train_file = val_txt,
                           num_classes = class_num)
    valSegData = DataLoader(valseg, batch_size = freeze_batch, num_workers = num_worker, collate_fn = collate_seg,
                             shuffle = False)


    # wait for all porcesses to synchronize
    if args.distributed:
        torch.distributed.barrier()


    # 训练
    for epoch in range(init_epoch, unfreeze_epoch):

        if args.distributed:
            trainSegData.batch_sampler.sampler.set_epoch(epoch)

        # -----------------------------------#
        # 使用余弦退火法
        # -----------------------------------#
        warm_up( modeloptimizer, freeze_epoch, epoch,
                min_lr, max_lr, warm_iter)

        epoch_fit(epoch, unfreeze_epoch, save_step,
                  model,modeloptimizer, [trainSegData,testSegData,valSegData],
                  device, logger, args, save_path, eval,local_rank)




