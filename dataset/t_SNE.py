import csv
import os
import os.path as osp
import pandas as pd
import numpy as np
import nibabel as nib
import h5py
from torch.utils.data import Dataset
import cv2

from utils import GaussDiscreter
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import SimpleITK as sitk

def FindLabelBox2D(img, offset):
    '''
    img:ct-label-data
    offset:copy-level
    '''
    xdim = np.zeros(2)  # bouding box 和 x轴的交点
    ydim = np.zeros(2)  # bouding box 和 y轴的交点
    # zdim = np.zeros(2)  # bouding box 和 z轴的交点
    tmp = np.squeeze(np.sum(np.sum(img, axis=2), axis=1))
    # print("for x:", len(tmp))
    for i in range(len(tmp)):
        if tmp[i] == 0:
            xdim[0] = i
        else:
            break
    xdim[1] = len(tmp)
    for i in reversed(range(len(tmp))):
        if tmp[i] == 0:
            xdim[1] = i
        else:
            break
    # for y
    tmp = np.squeeze(np.sum(np.sum(img, axis=2), axis=0))
    # print("for y:", len(tmp))
    for i in range(len(tmp)):
        if tmp[i] == 0:
            ydim[0] = i
        else:
            break

    ydim[1] = len(tmp)
    for i in reversed(range(len(tmp))):
        if tmp[i] == 0:
            ydim[1] = i
        else:
            break


    # offset
    xdim[0] = max(0, xdim[0] - offset)
    xdim[1] = min(np.size(img, 0), xdim[1] + offset)

    ydim[0] = max(0, ydim[0] - offset)
    ydim[1] = min(np.size(img, 1), ydim[1] + offset)


    return xdim, ydim



class BrainSet(Dataset):
    # def __init__(self, root_dir='/home/hefb/data/brain_t1', mode='train', transform=None) -> None:
    def __init__(self, root_dir='/data/ssd_2/liuyy/biaoshu_pre_exp/data_slice/texture', mode='train', transform=None) -> None:
        super().__init__()
        # assert mode in ['train', 'val', 'test']

        info_pd = pd.read_csv(f'{root_dir}/../all_texture.csv')

        self.ids = info_pd['subject_id'].to_list()
        self.labels = info_pd['label'].to_list()
        self.data_dir = f'{root_dir}'

        self.transform = transform
        self.max_x = 0
        self.max_y = 0

    def __getitem__(self, index):
        print(index)
        data_path = self.ids[index] + '.jpg'
        data = cv2.imread(data_path)
        xdim, ydim = FindLabelBox2D(data, 0)

        data_roi = data[int(xdim[0]):int(xdim[1]), int(ydim[0]):int(ydim[1])]
        self.max_x, self.max_y = max(self.max_x, xdim[1] - xdim[0]), max(self.max_y, ydim[1] - ydim[0])
        print(self.max_x, self.max_y)
        data_roi = data_roi.flatten()


        # data_path = osp.join(self.data_dir, str(self.ids[index]), 'T1.nrrd')
        # data = sitk.ReadImage(data_path)
        # data = sitk.GetArrayFromImage(data)[np.newaxis]
        #
        # seg_path = osp.join(self.data_dir, str(self.ids[index]), 'seg.nrrd')
        # seg = sitk.ReadImage(seg_path)
        # seg = sitk.GetArrayFromImage(seg)[np.newaxis]

        label = np.array(self.labels[index], dtype=np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return {'data':data_roi, 'label':label, 'id':self.ids[index]}

    def __len__(self):
        return len(self.ids)

pth = '/data/ssd_2/liuyy/biaoshu_pre_exp/data_slice/data/'


def preprocess():
    pth = '/data/ssd_2/liuyy/biaoshu_pre_exp/data/data/'
    # norms_pth = pth + '0/'
    # patients_pth = pth + '1/'
    out_dir = '/data/ssd_2/liuyy/biaoshu_pre_exp/data_slice/data/'

    # file = open('/data/ssd_2/liuyy/biaoshu_pre_exp/data_slice/all.csv', 'w')
    # writer = csv.writer(file)

    norms = os.listdir(pth)

    for i in range(len(norms)):
        path = pth + norms[i] + '/'
        print(path)
        raw = sitk.ReadImage(path + 'T1.nrrd')
        raw = sitk.GetArrayFromImage(raw)
        seg = sitk.ReadImage(path + 'seg.nrrd')
        seg = sitk.GetArrayFromImage(seg)

        xinji = raw * seg
        a,b,c = raw.shape
        for j in range(a):
            if seg[j,].sum() > 0:
                name = norms[i] + '_' + str(j)
                print(name)
                if not os.path.exists(out_dir + name):
                    os.mkdir(out_dir + name)
                plt.imsave(out_dir + name + '/' + 'xinji.jpg', xinji[j,], cmap='gray')
                plt.imsave(out_dir + name + '/' + 'raw.jpg', raw[j,], cmap='gray')
                plt.imsave(out_dir + name + '/' + 'seg.jpg', seg[j,], cmap='gray')

                # sitk.WriteImage(sitk.GetImageFromArray(xinji[i,]), out_dir + name + '/' + 'xinji.nrrd')
                # sitk.WriteImage(sitk.GetImageFromArray(xinji[i,]), out_dir + name + '/' + 'raw.nrrd')


def local_binary_pattern(image, P, R):
    # Compute LBP image
    lbp = np.zeros_like(image, dtype=np.uint8)
    for i in range(R, image.shape[0] - R):
        for j in range(R, image.shape[1] - R):
            center = image[i, j]
            values = []
            for p in range(P):
                x = int(round(i + R * np.cos(2 * np.pi * p / P)))
                y = int(round(j - R * np.sin(2 * np.pi * p / P)))
                values.append(1 if image[x, y] >= center else 0)
            lbp[i, j] = sum([v * (2 ** p) for p, v in enumerate(values)])

    return lbp

def Texture_LBP():
    names = os.listdir(pth)
    out_dir = '/data/ssd_2/liuyy/biaoshu_pre_exp/data_slice/texture/'

    file = open('/data/ssd_2/liuyy/biaoshu_pre_exp/data_slice/all_texture.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(['subject_id', 'label'])
    for i in range(len(names)):
        path = pth + names[i]
        print(path + 'xinji.jpg')
        image = cv2.imread(path + '/xinji.jpg', cv2.IMREAD_GRAYSCALE)
        lbp_image = local_binary_pattern(image, 8, 1)
        plt.imsave(out_dir + names[i] + '.jpg', lbp_image, cmap='gray')
        print(eval(names[i]))
        if eval(names[i]) <= 1005:
            writer.writerow([out_dir + names[i], 1])
        else:
            writer.writerow([out_dir + names[i], 0])
    file.close()

# 加载数据
def get_data():
    """
    :return: 数据集、标签、样本数量、特征数量
    """

    dataset = BrainSet()
    dataloader = DataLoader(dataset, 37)
    batch = next(iter(dataloader))
    data = batch['data']  # 图片特征
    label = batch['label']  # 图片标签
    n_samples = 'Null'
    n_features = ''
    return data, label, n_samples, n_features


# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
                 fontdict={'weight': 'bold', 'size': 7})
    plt.xticks()  # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig

'''下面的两个函数，
一个定义了二维数据，一个定义了3维数据的可视化
不作详解，也无需再修改感兴趣可以了解matplotlib的常见用法
'''
def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    label = np.array(label)
    label_ = [[]]

    plt.figure(figsize=(6, 5))
    for i in range(data.shape[0]):
        print(label[i])
        if label[i] == 0:
            plt.text(data[i, 0], data[i, 1], int(label[i]),
                 color='b',
                 fontdict={'weight': 'bold', 'size': 9})
        else:
            plt.text(data[i, 0], data[i, 1],  int(label[i]),
                     color='r',
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()  # gca:get current axis得到当前轴
        # 设置图片的右边框和上边框为不显示
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax = plt.gca()  # gca:get current axis得到当前轴
        # 设置图片的右边框和上边框为不显示
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        # plt.show()
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    # plt.title(title)
    plt.show()
    return fig

# 主函数，执行t-SNE降维
def main():
    import torch
    # preprocess()
    # Texture_LBP()

    print()

    data, label, n_samples, n_features = get_data()  # 调用函数，获取数据集信息
    print(label)
    print('Starting compute t-SNE Embedding...')

    ts = TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降
    reslut = ts.fit_transform(data)
    print(reslut)

    label = torch.tensor([1., 0., 0., 1., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1.,1., 0.,
 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0.,1., 1.,1.])
    # reslut = [[ -482.4774 ,-2529.143  ], [-2420.325  ,  3003.0415 ], [-5434.954 ,  -3620.3206 ],[ 7310.7104 ,  1659.4625 ], [-3287.4807 ,  6863.5923 ], [ 3089.5789 ,  3358.328  ], [ 5803.441  , -2476.8289 ], [-1804.8011 , -6327.163  ],[-5248.051   ,  455.05743], [ 2391.9539 ,  -559.8208 ]]

    # 调用函数，绘制图像
    fig = plot_embedding_2D(reslut, label, 't-SNE Embedding of digits')
    # 显示图像
    # plt.show()


# 主函数
if __name__ == '__main__':
    # preprocess()
    main()

    # ValueError: Found array with dim 5. Estimator expected <= 2.