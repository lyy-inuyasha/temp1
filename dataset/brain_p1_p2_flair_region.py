import os.path as osp
import pandas as pd
import numpy as np
import nibabel as nib
import h5py
from torch.utils.data import Dataset
import SimpleITK as sitk
from utils import GaussDiscreter

class BrainSet(Dataset):
    # def __init__(self, root_dir='/home/hefb/data/brain_t1', mode='train', transform=None) -> None:
    def __init__(self, root_dir='/data/ssd_2/liuyy/BrainAgeLujm/data_nju_region_split', mode='train', transform=None) -> None:
        super().__init__()
        assert mode in ['train', 'val', 'test']
        if mode in ['train', 'val']:
            info_pd = pd.read_csv(f'{root_dir}/{mode}.csv')
        else:
            info_pd = pd.read_csv(f'{root_dir}/{mode}_integrity.csv')

        self.ids = info_pd['subject_id'].to_list()
        self.labels = info_pd['age'].to_list()
        self.data_dir = f'{root_dir}/{mode}'
        self.mask = sitk.ReadImage('/data/ssd_2/liuyy/BrainAgeLujm/data_nju/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1.5mm.nii/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1.5mm.nii')
        self.mask = sitk.GetArrayFromImage(self.mask)
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        if self.mode in ['train', 'val']:
            data_p1_path = osp.join(self.data_dir, str(self.ids[index]), 'mwp1T1_postproc.nii')
            data_p1 = sitk.ReadImage(data_p1_path)
            data_p1 = sitk.GetArrayFromImage(data_p1)

            data_p2_path = osp.join(self.data_dir, str(self.ids[index]), 'mwp2T1_postproc.nii')
            data_p2 = sitk.ReadImage(data_p2_path)
            data_p2 = sitk.GetArrayFromImage(data_p2)

            data_flair_path = osp.join(self.data_dir, str(self.ids[index]), 'flair_mni_1.5m.nii.gz')
            data_flair = sitk.ReadImage(data_flair_path)
            data_flair = sitk.GetArrayFromImage(data_flair)
            data = np.stack((data_p1,data_p2, data_flair), axis=0)
            label = np.array(self.labels[index], dtype=np.float32)

        elif self.mode in ['test']:
            # data = np.zeros((400,2,26,33,23))

            for i in range(1, 400 + 1):
                dir_name = str(self.ids[index]) + '_' + str(i)
                # print(dir_name)
                data_p1_path = osp.join(self.data_dir, dir_name, 'mwp1T1_postproc.nii')
                data_p1 = sitk.ReadImage(data_p1_path)
                data_p1 = sitk.GetArrayFromImage(data_p1)
                data_p2_path = osp.join(self.data_dir, str(self.ids[index]), 'mwp2T1_postproc.nii')
                data_p2 = sitk.ReadImage(data_p2_path)
                data_p2 = sitk.GetArrayFromImage(data_p2)
                data_flair_path = osp.join(self.data_dir, dir_name, 'flair_mni_1.5m.nii.gz')
                data_flair = sitk.ReadImage(data_flair_path)
                data_flair = sitk.GetArrayFromImage(data_flair)
                data_ = np.stack((data_p1,data_p2, data_flair), axis=0)[np.newaxis]
                if i == 1:
                    data = data_
                else:
                    # print(data.shape, data_.shape)
                    data = np.concatenate([data, data_], axis=0)

            label = np.array(self.labels[index], dtype=np.float32)
            label = np.repeat(label, repeats=400, axis=0)


        if self.transform is not None:
            data = self.transform(data)
        return {'data': data, 'label': label, 'id': self.ids[index]}


    def __len__(self):
        return len(self.ids)


class DisBrainSet(BrainSet):
    def __init__(self, root_dir='/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split', mode='train', transform=None) -> None:
        super().__init__(root_dir, mode, transform)
        self.discreter = GaussDiscreter(5, 105, 2, 1)
        # ndarray
        self.dis_labels = self.discreter.contin2dis(self.labels)

    def __getitem__(self, index):
        data_p1_path = osp.join(self.data_dir, str(self.ids[index]), 'mwp1T1_postproc.nii')
        data_p1 = sitk.ReadImage(data_p1_path)
        data_p1 = sitk.GetArrayFromImage(data_p1)
        # data_p1 = sitk.GetArrayFromImage(data_p1) * self.mask
        data_p2_path = osp.join(self.data_dir, str(self.ids[index]), 'mwp2T1_postproc.nii')
        data_p2 = sitk.ReadImage(data_p2_path)
        data_p2 = sitk.GetArrayFromImage(data_p2)
        data_flair_path = osp.join(self.data_dir, str(self.ids[index]), 'flair_mni_1.5m.nii.gz')
        data_flair = sitk.ReadImage(data_flair_path)
        data_flair = sitk.GetArrayFromImage(data_flair)
        # data_flair = sitk.GetArrayFromImage(data_flair) * self.mask
        data = np.stack((data_p1,data_p2, data_flair), axis=0)

        label = np.array(self.dis_labels[index], dtype=np.float32)
        label = np.repeat(label, repeats=400, axis=0)
        gt = np.array(self.labels[index], dtype=np.float32)
        gt = np.repeat(gt, repeats=400, axis=0)
        if self.transform is not None:
            data = self.transform(data)

        return {'data':data, 'label':label, 'gt':gt}


class NoisyBrainSet(BrainSet):
    def __init__(self, root_dir='/data/ssd_2/liuyy/BrainAgeLujm/data_nju_region_split', mode='train', transform=None) -> None:
        super().__init__(root_dir, mode, transform)

    def __getitem__(self, index):
        # data_path = osp.join(self.data_dir, str(self.ids[index]), 'mwp1T1_postproc.nii')
        # data = nib.load(data_path).get_fdata(dtype=np.float32)[np.newaxis]
        # print(data.shape)
        data_p1_path = osp.join(self.data_dir, str(self.ids[index]), 'mwp1T1_postproc.nii')
        data_p1 = sitk.ReadImage(data_p1_path)
        data_p1 = sitk.GetArrayFromImage(data_p1)
        data_p2_path = osp.join(self.data_dir, str(self.ids[index]), 'mwp2T1_postproc.nii')
        data_p2 = sitk.ReadImage(data_p2_path)
        data_p2 = sitk.GetArrayFromImage(data_p2)
        # data_p1 = sitk.GetArrayFromImage(data_p1) * self.mask
        data_flair_path = osp.join(self.data_dir, str(self.ids[index]), 'flair_mni_1.5m.nii.gz')
        data_flair = sitk.ReadImage(data_flair_path)
        data_flair = sitk.GetArrayFromImage(data_flair)
        # data_flair = sitk.GetArrayFromImage(data_flair) * self.mask
        data = np.stack((data_p1, data_p2,data_flair), axis=0)

        label = np.array(self.labels[index], dtype=np.float32)
        # add some noise to labels
        label += np.random.normal(size=label.shape).astype(np.float32)
        # label = np.repeat(label, repeats=400, axis=0)
        if self.transform is not None:
            data = self.transform(data)

        return {'data':data, 'label':label}


class BrainTestSet(Dataset):
    def __init__(self, root_dir='/home/hefb/data/brain_external/', npz='NIFD', transform=None) -> None:
        super().__init__()
        assert npz in ['NIFD', 'AD', 'CN', 'EMCI', 'LMCI', 'MCI', 'SMC']

        self.npz = npz
        self.root_dir = root_dir
        self.dataset = np.load(osp.join(root_dir, f'{npz}.npz'))
        self.data = self.dataset['data']
        self.labels = self.dataset['label']
        self.transform = transform

    def __getitem__(self, index):
        # data = self.dataset['data'][index][np.newaxis]
        data = self.data[index][np.newaxis]
        label = np.array(self.labels[index], dtype=np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return {'data':data, 'label':label}

    def __len__(self):
        return len(self.labels)


class DisBrainTestSet(BrainTestSet):
    def __init__(self, root_dir='/home/hefb/data/brain_external', npz='NIFD', transform=None) -> None:
        super().__init__(root_dir, npz, transform)
        self.discreter = GaussDiscreter(5, 105, 2, 1)
        # ndarray
        self.dis_labels = self.discreter.contin2dis(self.labels)

    def __getitem__(self, index):
        data = self.data[index][np.newaxis]
        label = np.array(self.dis_labels[index], dtype=np.float32)
        gt = np.array(self.labels[index], dtype=np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return {'data':data, 'label':label, 'gt':gt}

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = BrainSet(mode='test')
    print(dataset[0]['data'].shape, dataset[0]['label'].shape)
    dataloader = DataLoader(dataset, 1)
    batch = next(iter(dataloader))
    print(batch['data'].shape, batch['data'].dtype)
    print(batch['label'].shape, batch['label'].dtype)
    print(len(dataloader))

    print(batch['label'])
    # print(batch['label'])
