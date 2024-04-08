import os.path as osp
import pandas as pd
import numpy as np
import nibabel as nib
import h5py
from torch.utils.data import Dataset

from utils import GaussDiscreter

class BrainSet(Dataset):
    # def __init__(self, root_dir='/home/hefb/data/brain_t1', mode='train', transform=None) -> None:
    def __init__(self, root_dir='/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split', mode='train', transform=None) -> None:
        super().__init__()
        assert mode in ['train', 'val', 'test']

        info_pd = pd.read_csv(f'{root_dir}/{mode}.csv')

        self.ids = info_pd['subject_id'].to_list()
        self.labels = info_pd['age'].to_list()
        self.data_dir = f'{root_dir}/{mode}'

        self.transform = transform

    def __getitem__(self, index):
        data_path = osp.join(self.data_dir, str(self.ids[index]), 'smwp1T1_resample.nii')
        data = nib.load(data_path).get_fdata(dtype=np.float32)[np.newaxis]
        label = np.array(self.labels[index], dtype=np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return {'data':data, 'label':label, 'id':self.ids[index]}

    def __len__(self):
        return len(self.ids)


class DisBrainSet(BrainSet):
    def __init__(self, root_dir='/data/ssd_2/liuyy/BrainAgeLujm/data_nju_split', mode='train', transform=None) -> None:
        super().__init__(root_dir, mode, transform)
        self.discreter = GaussDiscreter(5, 105, 2, 1)
        # ndarray
        self.dis_labels = self.discreter.contin2dis(self.labels)

    def __getitem__(self, index):
        data_path = osp.join(self.data_dir, str(self.ids[index]), 'smwp1T1_resample.nii')
        data = nib.load(data_path).get_fdata(dtype=np.float32)[np.newaxis]
        label = np.array(self.dis_labels[index], dtype=np.float32)
        gt = np.array(self.labels[index], dtype=np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return {'data':data, 'label':label, 'gt':gt}


class NoisyBrainSet(BrainSet):
    def __init__(self, root_dir='/data/ssd_2/liuyy/brain_t1_new_liuyy', mode='train', transform=None) -> None:
        super().__init__(root_dir, mode, transform)

    def __getitem__(self, index):
        data_path = osp.join(self.data_dir, str(self.ids[index]), 'smwp1T1_resample.nii')
        data = nib.load(data_path).get_fdata(dtype=np.float32)[np.newaxis]
        label = np.array(self.labels[index], dtype=np.float32)
        # add some noise to labels
        label += np.random.normal(size=label.shape).astype(np.float32)

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

    dataset = DisBrainTestSet(npz='NIFD')
    print(dataset[0]['data'].shape, dataset[0]['label'].shape)
    dataloader = DataLoader(dataset, 4)
    batch = next(iter(dataloader))
    print(batch['data'].shape, batch['data'].dtype)
    print(batch['label'].shape, batch['label'].dtype)
    print(len(dataloader))

    print(batch['gt'])
    print(batch['label'])
