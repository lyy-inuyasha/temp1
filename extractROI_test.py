import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from skimage import measure


def FindLabelBox3D(img, offset):
    '''
    img:ct-label-data
    offset:copy-level
    '''
    xdim = np.zeros(2)  # bouding box 和 x轴的交点
    ydim = np.zeros(2)  # bouding box 和 y轴的交点
    zdim = np.zeros(2)  # bouding box 和 z轴的交点
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
    # for z
    tmp = np.squeeze(np.sum(np.sum(img, axis=1), axis=0))
    # print("for z:", len(tmp))
    for i in range(len(tmp)):
        if tmp[i] == 0:
            zdim[0] = i
        else:
            break

    zdim[1] = len(tmp)
    for i in reversed(range(len(tmp))):
        if tmp[i] == 0:
            zdim[1] = i
        else:
            break

    # offset
    xdim[0] = max(0, xdim[0] - offset)
    xdim[1] = min(np.size(img, 0), xdim[1] + offset)

    ydim[0] = max(0, ydim[0] - offset)
    ydim[1] = min(np.size(img, 1), ydim[1] + offset)

    zdim[0] = max(0, zdim[0] - offset)
    zdim[1] = min(np.size(img, 2), zdim[1] + offset)

    return xdim, ydim, zdim


def _nii_region_split(file_pth, out_dir, writer_pth, age, subject_id, save_name):
    max_x = max_y = max_z = 0

    mask = sitk.ReadImage(
        '/data/ssd_2/liuyy/BrainAgeLujm/data_nju/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1.5mm.nii/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1.5mm.nii')
    mask = sitk.GetArrayFromImage(mask)

    file = sitk.ReadImage(file_pth)
    file = sitk.GetArrayFromImage(file)
    # print(mask.max())

    for i in range(1, int(mask.max()) + 1):
        mask_part = np.zeros(mask.shape)
        mask_part[mask == i] = 1
        # print(file.shape)
        # print(mask_part.shape)
        file_part = mask_part * file
        # 脑区上下和周围都没有offset
        xdim, ydim, zdim = FindLabelBox3D(file_part, 0)
        image_bouding_box = file_part[int(xdim[0]):int(xdim[1]), int(ydim[0]):int(ydim[1]), int(zdim[0]):int(zdim[1])]
        a,b,c = image_bouding_box.shape

        temple_shape = (26,33,23)
        norm_image_bounding_box = np.zeros(temple_shape)
        norm_image_bounding_box[(temple_shape[0] - a)//2:(temple_shape[0] - a)//2 + a,(temple_shape[1] - b)//2:(temple_shape[1] - b)//2 + b, (temple_shape[2] - c)//2:(temple_shape[2] - c)//2 + c] = image_bouding_box
        # print(norm_image_bounding_box.shape)

        if not os.path.exists(out_dir + subject_id + '_' + str(i)):
            os.mkdir(out_dir + subject_id + '_' + str(i))

        sitk.WriteImage(sitk.GetImageFromArray(norm_image_bounding_box), out_dir + subject_id + '_' + str(i) + '/' + save_name)

        # 写入文件
        file_wrier = open(writer_pth, 'a')
        writer = csv.writer(file_wrier)
        writer.writerow([subject_id + '_' + str(i), age])
        file_wrier.close()

        # max_x, max_y, max_z = max(max_x, xdim[1]-xdim[0]), max(max_y, ydim[1]-ydim[0]), max(max_z, zdim[1]-zdim[0])

    # return max_x, max_y, max_z


def val_region_split():
    csv_pth = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split/val.csv'
    val_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split/val/'
    out_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_region_split/'

    val_csv_reader = open(csv_pth)
    val_csv_reader = csv.reader(val_csv_reader)

    writer_pth = out_dir + 'val.csv'
    file = open(writer_pth, 'w')
    writer = csv.writer(file)
    writer.writerow(['subject_id', 'age'])
    file.close()


    for row in val_csv_reader:
        if 'age' in row:
            continue


        print(val_dir + row[0])
        _nii_region_split(val_dir + row[0] + '/' + 'flair_mni_1.5m.nii.gz', out_dir + 'val/', writer_pth, row[1], row[0],  'flair_mni_1.5m.nii.gz')
        _nii_region_split(val_dir + row[0] + '/' + 'mwp1T1_postproc.nii', out_dir + 'val/', writer_pth, row[1], row[0], 'mwp1T1_postproc.nii')
        _nii_region_split(val_dir + row[0] + '/' + 'mwp2T1_postproc.nii', out_dir + 'val/', writer_pth, row[1], row[0], 'mwp2T1_postproc.nii')
        _nii_region_split(val_dir + row[0] + '/' + 'wmT1_postproc.nii', out_dir + 'val/', writer_pth, row[1], row[0], 'wmT1_postproc.nii')
        #max_x, max_y, max_z = max(max_x4, max_x3, max_x2, max_x1),max(max_y4, max_y3, max_y2, max_y1),max(max_z4, max_z3, max_z2, max_z1)
        #print(max_x,max_y,max_z)


    # writer.close()

def train_region_split():
    csv_pth = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split/train.csv'
    val_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split/train/'
    out_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_region_split/'

    val_csv_reader = open(csv_pth)
    val_csv_reader = csv.reader(val_csv_reader)

    writer_pth = out_dir + 'train.csv'
    file = open(writer_pth, 'w')
    writer = csv.writer(file)
    writer.writerow(['subject_id', 'age'])
    file.close()


    for row in val_csv_reader:
        if 'age' in row:
            continue


        print(val_dir + row[0])
        _nii_region_split(val_dir + row[0] + '/' + 'flair_mni_1.5m.nii.gz', out_dir + 'train/', writer_pth, row[1], row[0],  'flair_mni_1.5m.nii.gz')
        _nii_region_split(val_dir + row[0] + '/' + 'mwp1T1_postproc.nii', out_dir + 'train/', writer_pth, row[1], row[0], 'mwp1T1_postproc.nii')
        _nii_region_split(val_dir + row[0] + '/' + 'mwp2T1_postproc.nii', out_dir + 'train/', writer_pth, row[1], row[0], 'mwp2T1_postproc.nii')
        _nii_region_split(val_dir + row[0] + '/' + 'wmT1_postproc.nii', out_dir + 'train/', writer_pth, row[1], row[0], 'wmT1_postproc.nii')
        #max_x, max_y, max_z = max(max_x4, max_x3, max_x2, max_x1),max(max_y4, max_y3, max_y2, max_y1),max(max_z4, max_z3, max_z2, max_z1)
        #print(max_x,max_y,max_z)


    # writer.close()

def test_region_split():
    count=0
    csv_pth = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split/test.csv'
    val_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split/test/'
    out_dir = '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_region_split/'

    val_csv_reader = open(csv_pth)
    val_csv_reader = csv.reader(val_csv_reader)

    writer_pth = out_dir + 'test.csv'
    file = open(writer_pth, 'w')
    writer = csv.writer(file)
    writer.writerow(['subject_id', 'age'])
    file.close()


    for row in val_csv_reader:
        if 'age' in row:
            continue


        print(val_dir + row[0])
        _nii_region_split(val_dir + row[0] + '/' + 'flair_mni_1.5m.nii.gz', out_dir + 'test/', writer_pth, row[1], row[0],  'flair_mni_1.5m.nii.gz')
        _nii_region_split(val_dir + row[0] + '/' + 'mwp1T1_postproc.nii', out_dir + 'test/', writer_pth, row[1], row[0], 'mwp1T1_postproc.nii')
        _nii_region_split(val_dir + row[0] + '/' + 'mwp2T1_postproc.nii', out_dir + 'test/', writer_pth, row[1], row[0], 'mwp2T1_postproc.nii')
        _nii_region_split(val_dir + row[0] + '/' + 'wmT1_postproc.nii', out_dir + 'test/', writer_pth, row[1], row[0], 'wmT1_postproc.nii')
        #max_x, max_y, max_z = max(max_x4, max_x3, max_x2, max_x1),max(max_y4, max_y3, max_y2, max_y1),max(max_z4, max_z3, max_z2, max_z1)
        #print(max_x,max_y,max_z)
        count += 1
    return count
    # writer.close()


if __name__ == '__main__':
    print(test_region_split())

    # mask = sitk.ReadImage(
    #     '/data/ssd_2/liuyy/BrainAgeLujm/data_nju/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1.5mm.nii/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1.5mm.nii')
    # mask = sitk.GetArrayFromImage(mask)
    # image_data = sitk.ReadImage('/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split/train/2302201243/wmT1_postproc.nii')
    # image_data = sitk.GetArrayFromImage(image_data)
    #
    # mask_part = np.zeros((mask.shape))
    # print(mask.shape, mask_part.shape)
    # mask_part[mask==1] = 1
    #
    # sitk.WriteImage(sitk.GetImageFromArray(mask_part), '/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split/temp/mask_part.nii')
    #
    #
    # xdim, ydim, zdim = FindLabelBox3D(mask_part, 1)
    #
    #
    #
    #
    #
    # ##image-roi
    # image_bouding_box = image_data[int(xdim[0]):int(xdim[1]), int(ydim[0]):int(ydim[1]), int(zdim[0]):int(zdim[1])]
    # ##label-roi
    # label_bouding_box = mask_part[int(xdim[0]):int(xdim[1]), int(ydim[0]):int(ydim[1]), int(zdim[0]):int(zdim[1])]
    #
    # image_bouding_box = image_bouding_box * label_bouding_box
    #
    # for i in range(len(image_bouding_box)):
    #     plt.figure(12)
    #     plt.subplot(131)
    #     plt.imshow(image_bouding_box[i,], cmap='gray')
    #     plt.subplot(132)
    #     plt.imshow(label_bouding_box[i,:,:])
    #     plt.subplot(133)
    #     plt.imshow(mask_part[i,])
    #     plt.show()
