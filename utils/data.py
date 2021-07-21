import h5py, torch, cv2, os
import numpy as np
from PIL import Image
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

class H5Dataset(torch.utils.data.Dataset):
    
    def __init__(self, f_path, balance = False, resize_shape = (100, 100), length=None, transform=None):
        super().__init__()
        self.f_path = f_path
        self.resize_shape = resize_shape
        self.balance = balance
        self.length = length
        self.data, self.target = self._load_data(self.f_path)
        self.shape = self.data.shape
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        data = Image.fromarray(data.astype('uint8'), 'L')
        data = data.resize(self.resize_shape)
        #data = cv2.equalizeHist(np.array(data))
        target = self.target[idx]
        target = Image.fromarray(target.astype('uint8'), 'L')
        target = target.resize(self.resize_shape)
        
        if self.transform is not None:
            data = self.transform(data)
        return data, target
    
    def _load_data(self, file_path):
        f = h5py.File(file_path, 'r')
        
        if self.length:
            data = f['images'][:self.length]
            target = f['masks'][:self.length]
        else:
            data = f['images'][:][1:]
            target = f['masks'][:][1:]
        if self.balance:
            h, w = data.shape[1], data.shape[2]
            print('Original dataset shape {}'.format(Counter(target)))
            ros = RandomUnderSampler()
            data = np.reshape(data, (data.shape[0], h*w))
            data, target = ros.fit_sample(data, target)
            data = np.reshape(data, (data.shape[0], h, w))
            print('Resampled dataset shape {} \n'.format(Counter(target)))
            print('Dataset size : {}'.format(data.shape))
        return data, target
     
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.all_imgs = os.listdir(os.path.join(root, 'train', 'images'))
        self.all_targets = os.listdir(os.path.join(root, 'train', 'masks'))
        
    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, 'train', 'images', self.all_imgs[idx])
        target_loc = os.path.join(self.root, 'train', 'masks', self.all_targets[idx])
        image = np.array(Image.open(img_loc))
        target = np.array(Image.open(target_loc))
        return image, target