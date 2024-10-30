import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


class Eval():
    def __init__(self, class_num):

        self.num_classes = class_num

        # 初始化混淆矩阵
        self.mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        self.color_palette = [176, 235, 180, 255, 255, 255, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153,
                              250,
                              170, 30,
                              220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142,
                              0,
                              0, 70,
                              0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

        for l in range(255 * 3 - len(self.color_palette)):
            self.color_palette.append(0)

        self.color_map_arr = np.array(self.color_palette)

    def init(self, a, b):
        # -----------------------------#
        # convert to 1D
        # a is predict b is label
        # -----------------------------#
        a = np.reshape(a, (-1))
        b = np.reshape(b, (-1))

        k = (a >= 0) & (a < self.num_classes)
        # # --------------------------------------------------------------------------------#
        # #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        # #   返回中，写对角线上的为分类正确的像素点
        # # --------------------------------------------------------------------------------#
        t = np.bincount(self.num_classes * a[k].astype(int) + b[k].astype(int),
                        minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

        # print(t)
        # assert a_shape == b_shape, "Input dimension not same,Error!"

        self.mat += np.bincount(self.num_classes * a.astype(int) + b.astype(int),
                                minlength=self.num_classes ** 2).reshape(self.num_classes,
                                                                         self.num_classes)

        # print(self.mat)

    def _iou(self):
        #
        Iou = np.diag(self.mat) / (np.sum(self.mat, axis=1) + np.sum(self.mat, axis=0) - (np.diag(self.mat)))
        return Iou

    def vis(self, gt,
            mask,
            name,
            path):


        gt = np.array(gt*255,np.uint8)
        mask = np.array(mask,np.uint8)

        gt = Image.fromarray(gt).convert('RGB')

        mask = Image.fromarray(mask).convert('P')

        mask.putpalette(self.color_palette)

        gt.save(f'{path}/{name}_gt.jpg')
        mask.save(f'{path}/{name}_label.png')

    # PA
    def _accuracy(self):

        acc = np.diag(self.mat).sum() / np.maximum(np.sum(self.mat), 1.)
        return acc

    def _recall(self):
        recall = np.diag(self.mat) / np.maximum(np.sum(self.mat, axis=1), 1.)
        return recall

    # CPA -> mean(CPA.sum())
    def _precision(self):
        p = np.diag(self.mat) / np.maximum(np.sum(self.mat, axis=0), 1.)
        return p

    # Equal dice
    def F1(self):
        return 2 * (self._precision() * self._recall()) / (self._precision() + self._recall())

    def get_res(self):
        print(f'混淆矩阵: {self.mat}')

        iou = self._iou()
        miou = np.mean(iou)

        PA = self._accuracy()

        recall = self._recall()

        f1 = self.F1()
        cpa = self._precision()

        mcpa = np.mean(cpa)

        result = {
            'iou': iou,
            'miou': miou,
            'PA': PA,
            'CPA': cpa,
            'MCAP': mcpa,
            'recall': recall,
            'f1 score': f1
        }

        return result

    def show(self):
        result = self.get_res()
        print(f"{'*' * 20} 评估开始   {'*' * 20}\t")

        for k, v in result.items():
            print(f"{' ' * 15}  {'|'}\t {k}: {v}\t | \t")

        print(f"{'*' * 20} 评估结束   {'*' * 20}\t")


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


if __name__ == '__main__':
    a = np.array([[1, 0],
                  [0, 1]])
    eval = Eval(class_num=2)
    b = np.array([[1, 1],
                  [0, 1]])

    hist = np.array([[18849854, 4249219],
                     [9022062, 3989201]])

    # ------------------------------------------#
    #    IOU =
    # ------------------------------------------#
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    # ------------------------------------------------#
    #   逐类别输出一下mIoU值
    # ------------------------------------------------#
    name_classes = ['__background__', "cloud"]
    if name_classes is not None:
        for ind_class in range(len(name_classes)):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                  + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
                round(Precision[ind_class] * 100, 2)))

    # eval.init(a, b)
    eval.mat = hist

    eval.show()
    # print(eval._IOU(a,b))

    # res = {
    #     'a':12,
    #     'b':45,
    #     'c':8
    # }
    #
    # x = 0.115454656





