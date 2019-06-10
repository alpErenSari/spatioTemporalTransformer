import torch.utils.data as data
import torch
import h5py
import os, glob
from PIL import Image
import numpy as np

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]

class DVDTrainingDataset(data.Dataset):

    def __init__(self,dataset_folder):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.input_folder = os.path.join(dataset_folder, 'input')
        self.gt_folder = os.path.join(dataset_folder, 'GT')
        # self.name_inputs = os.listdir(self.input_folder)
        self.name_inputs = glob.glob(self.input_folder + '/*.jpg')
        self.name_inputs.sort(reverse=False)
        # self.name_gts = os.listdir(self.gt_folder)
        self.name_gts = glob.glob(self.gt_folder + '/*.jpg')
        self.name_gts.sort(reverse=False)
        self.dataset_folder = dataset_folder
        self.img_size = (64,64)

    def __len__(self):
        return len(self.name_inputs)

    def __getitem__(self, idx):
        img_out = np.zeros((15,self.img_size[0],self.img_size[1]), dtype=np.float32)
        for i in range(5):
            if(idx>=2 and idx<(len(self)-2)):
                # img_name = os.path.join(self.input_folder, self.name_inputs[idx+i-2])
                img_name = self.name_inputs[idx+i-2]
                image = Image.open(img_name)
                image = np.array(image.resize(self.img_size))/255.
                image = np.moveaxis(image, 2, 0)
                img_out[3*i:3*(i+1), :, :] = image
            elif(idx < 2):
                img_name = self.name_inputs[i]
                image = Image.open(img_name)
                image = np.array(image.resize(self.img_size))/255.
                image = np.moveaxis(image, 2, 0)
                img_out[3*i:3*(i+1), :, :] = image
            elif(idx >= len(self)-2):
                img_name = self.name_inputs[len(self)-5+i]
                image = Image.open(img_name)
                image = np.array(image.resize(self.img_size))/255.
                image = np.moveaxis(image, 2, 0)
                img_out[3*i:3*(i+1), :, :] = image
        gt_name = self.name_gts[idx]
        gt = Image.open(gt_name)
        gt = np.array(gt.resize(self.img_size))/255.
        gt = np.moveaxis(gt, 2, 0)
        #labels = labels.reshape(-1, 2)

        return torch.from_numpy(img_out).float(), torch.from_numpy(gt).float()
