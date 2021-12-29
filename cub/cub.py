from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
import torch


        
class MLDataInstance(data.Dataset):
    """Metric Learning Dataset.
    """
    def __init__(self, src_dir, dataset_name, train = True, transform=None, target_transform=None, nnIndex = None):
       
        data_dir = src_dir + dataset_name + '/'#处理的数据的文件夹
        if train:
            img_data  = np.load(data_dir + '{}_{}_256resized_img.npy'.format('training',dataset_name))
            img_label = np.load(data_dir + '{}_{}_256resized_label.npy'.format('training',dataset_name))
        else:
            img_data  = np.load(data_dir + '{}_{}_256resized_img.npy'.format('validation',dataset_name))
            img_label = np.load(data_dir + '{}_{}_256resized_label.npy'.format('validation',dataset_name))

        self.img_data  = img_data
        self.img_label = img_label
        self.transform = transform
        self.target_transform = target_transform
        self.nnIndex = nnIndex

    def __getitem__(self, index):
        
        if self.nnIndex is not None:

            img1, img2, target = self.img_data[index], self.img_data[self.nnIndex[index]], self.img_label[index]

            img1 = self.transform(img1)
            img2 = self.transform(img2)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img1, img2, target, index
            
        else:
            img, target = self.img_data[index], self.img_label[index]
            img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target, index
        
    def __len__(self):
        return len(self.img_data)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(size=224),
        # transforms.RandomHorizontalFlip(),#依概率水平翻转
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

trainset = MLDataInstance(src_dir = 'data/', dataset_name = 'cub200', train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5864, shuffle=True, num_workers=4, drop_last =True)

testset = MLDataInstance(src_dir ='data/', dataset_name = 'cub200', train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=5924, shuffle=False, num_workers=4)
print(len(trainset))
print(len(testset))
for batch_idx, (inputs1,  targets, indexes) in enumerate(trainloader):
    x_tr, y_tr, indexes = inputs1,  targets, indexes

for batch_idx, (inputs1, inputs2, targets) in enumerate(testloader):
    x_te, y_te, inde = inputs1, inputs2, targets

torch.save((x_tr, y_tr, x_te,  y_te), 'cub200.pt')