import os

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import csv
import SimpleITK as sitk
import pandas as pd
import random
import csv
import os
import shutil


def _copy(input_dir, output_dir):
    folds = os.listdir(input_dir)
    for i in range(len(folds)):
        shutil.copy(input_dir + folds[i], output_dir + folds[i])

# 数据清洗：
# 使得csv和patients_dir的数据对应！
def data_clean(patients_dir, csv_pth, output_csv_dir):
    # with open(csv_pth, 'rt') as csvfile:
    #     data = np.loadtxt(csvfile,  dtype='int', delimiter=',')

    cols_names = ['性别', '身份证号', '检查号', 'age']
    data = pd.read_csv(csv_pth)
    data = np.array(data)
    print(data.shape)

    # 写入csv
    all_csv = open(output_csv_dir + 'all.csv', 'w')
    writer = csv.writer(all_csv)
    writer.writerow(cols_names)


    patients_list = os.listdir(patients_dir)
    print(len(patients_list))

    for i in range(len(patients_list)):
        names = patients_list[i]
        j = 0

        print(names, data[j])
        while names != data[j][2]:
            j += 1
        # print('end')
        writer.writerow(data[j])

    all_csv.close()

    cols_names = ['性别', '身份证号', '检查号', 'age']
    data = pd.read_csv(output_csv_dir + 'all.csv', names=cols_names)
    print(data.shape)


# 每个病人按照脑区分开,单纯的只有脑区
def split_no_ratio(patients_dir, csv_pth, output_dir, mni_pth):
    # with open(csv_pth, 'rt') as csvfile:
    #     data = np.loadtxt(csvfile,  dtype='int', delimiter=',')
    cols_names = ['性别','身份证号','检查号','age']
    csv_data = pd.read_csv(csv_pth, names=cols_names)
    csv_data = np.array(csv_data)
    print(csv_data.shape)

    patients_list = os.listdir(patients_dir)
    print(len(patients_list))

    mni_template = sitk.ReadImage(mni_pth)
    mni_template = sitk.GetArrayFromImage(mni_template)
    print(mni_template.shape)
    print(mni_template.min(), mni_template.max())

    for i in range(len(patients_list)):
        names = patients_list[i]


def split_train_val_test():

    input_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju/'
    out_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split/'

    csv_old_data = open(input_dir + 'all_2column.csv', 'r')
    # csv_all_new_data = open(input_dir + 'all_new_data.csv', 'r')

    csv_reader_old = csv.reader(csv_old_data)
    # csv_reader_new = csv.reader(csv_all_new_data)

    dist = {}
    list = []

    train_list = []
    val_list = []
    test_list = []

    train_dir = out_dir + 'train/'
    val_dir = out_dir + 'val/'
    test_dir = out_dir + 'test/'

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for row in csv_reader_old:
        if 'age' in row:
            continue
        dist[row[0]] = row[1]
        list.append(row[0])

    random.seed(42)
    random.shuffle(list)

    train_list = list[: int(0.8 * len(list))]
    val_list = list[int(0.8 * len(list)): int(0.9 * len(list))]
    test_list = list[int(0.9 * len(list)):]

    print(len(dist))
    print(len(list))

    file_train = open(out_dir + 'train.csv', 'w')
    writer_train = csv.writer(file_train)
    writer_train.writerow(['subject_id', 'age'])

    file_val = open(out_dir + 'val.csv', 'w')
    writer_val = csv.writer(file_val)
    writer_val.writerow(['subject_id', 'age'])

    file_test = open(out_dir + 'test.csv', 'w')
    writer_test = csv.writer(file_test)
    writer_test.writerow(['subject_id', 'age'])

    for i in range(len(train_list)):
        # writer_all.writerow([train_list[i], dist[train_list[i]]])
        writer_train.writerow([train_list[i], dist[train_list[i]]])
        if not os.path.exists(train_dir + train_list[i]):
            os.mkdir(train_dir + train_list[i])

        _copy(input_dir + 'data_final_20240203/' + train_list[i] + '/', train_dir + train_list[i] + '/')

    for i in range(len(val_list)):
        # writer_all.writerow([val_list[i], dist[val_list[i]]])
        writer_val.writerow([val_list[i], dist[val_list[i]]])
        if not os.path.exists(val_dir + val_list[i]):
            os.mkdir(val_dir + val_list[i])
        _copy(input_dir + 'data_final_20240203/' + val_list[i] + '/', val_dir + val_list[i] + '/')
        # shutil.copy(input_dir + 'data/' + val_list[i] + '/smwp1T1_resample.nii',
        #             val_dir + val_list[i] + '/smwp1T1_resample.nii')

    for i in range(len(test_list)):
        # writer_all.writerow([test_list[i], dist[test_list[i]]])
        writer_test.writerow([test_list[i], dist[test_list[i]]])
        if not os.path.exists(test_dir + test_list[i]):
            os.mkdir(test_dir + test_list[i])
        _copy(input_dir + 'data_final_20240203/' + test_list[i] + '/', test_dir + test_list[i] + '/')


    # file.close()
    file_train.close()
    file_val.close()
    file_test.close()
    print(len(list))
    print(len(train_list), len(val_list), len(test_list))

def check_mni_mask(mni_pth):
    mni_pth='/data/ssd_2/liuyy/BrainAgeLujm/smwp1T1_resample.nii'
    save_pth = '/data/ssd_2/liuyy/BrainAgeLujm/pic_temp/gray_byj/'

    mni = sitk.ReadImage(mni_pth)
    mni = sitk.GetArrayFromImage(mni)

    # part = mni.copy()
    part = np.zeros(mni.shape, dtype=np.uint8)
    part[mni == 0] = 1

    a,b,c = mni.shape
    print(mni.shape)

    for i in range(0, a):
        plt.imshow(mni[i,:,:])
        plt.show()
        plt.imsave(save_pth + str(i) + '.jpg', mni[i,])


if __name__ == '__main__':
    patients_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju/data_final_20240203/'
    csv_pth = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju/all.csv'
    output_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data/noRatio/'
    mni_pth = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1.5mm.nii/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1.5mm.nii'

    # # 1 数据清洗
    # output_csv_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju/'
    # data_clean(patients_dir, csv_pth, output_csv_dir)

    # 2 脑零分区--不这么做
    # split_no_ratio(patients_dir, csv_pth, output_dir, mni_pth)

    # 3 划分Train:Val:Test
    # split_train_val_test()

    # 4 check
    check_mni_mask(mni_pth)
    print()
