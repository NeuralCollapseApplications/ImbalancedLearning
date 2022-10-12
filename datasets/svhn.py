import numpy as np
from .sampler import ClassAwareSampler

import torch
import torchvision
from torchvision import transforms
import torchvision.datasets

class IMBALANCECIFAR10(torchvision.datasets.SVHN):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, split="train",
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, split, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



class SVHN_LT(object):

    def __init__(self, distributed, root='./data/cifar10', imb_type='exp',
                    imb_factor=0.01, batch_size=128, num_works=40):

        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        ])

        
        if imb_factor < 1:
            print('Long Tail Setting ...')
            train_dataset_1 = IMBALANCECIFAR10(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, split="train", download=True, transform=transform)
            train_dataset_2 = IMBALANCECIFAR10(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, split="extra", download=True, transform=transform)

            self.cls_num_list = train_dataset_2.get_cls_num_list()
        else:
            assert(imb_factor == 1)
            print('Normal Case Setting ...')
            train_dataset_1 = torchvision.datasets.SVHN(root=root, split="train", download=True, transform=transform)
            train_dataset_2 = torchvision.datasets.SVHN(root=root, split="extra", download=True, transform=transform)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])
        eval_dataset = torchvision.datasets.SVHN(root=root, split="test", download=True, transform=transform)
        

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        #balance_sampler = ClassAwareSampler(train_dataset)
        #self.train_balance = torch.utils.data.DataLoader(
        #    train_dataset,
        #    batch_size=batch_size, shuffle=False,
        #    num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)