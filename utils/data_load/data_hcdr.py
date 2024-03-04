import random
from torchvision import transforms
import torch
import torch.utils.data as data_utils
import numpy as np
import os
import copy
import csv
import torch
import pickle
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import h5py
from utils.utils import timer


class MyDataset(data_utils.Dataset):

    @timer
    def __init__(self, dataset, args):
        self.data_list = []

        self.metric_name = 'AUC'
        
        # if dataset in ['train', 'valid']:
        if args.model_name == 'node':
            file = '/home/ceyu.cy2/datasets/tabular/Credit/application_train_node.h5'
        else:
            file = '/home/ceyu.cy2/datasets/tabular/Credit/application_train.h5'
        # else:
        #     file = '/home/ceyu.cy/datasets/tabular/Credit/application_test.h5'

        data_h5 = h5py.File(file, 'r')
        data = data_h5['my_dataset'][:]
        text_col_num = 16 #data_h5['text_column_num'].astype(int)[0]
        labels = data_h5['label'].astype(int)[:]

        mean = np.array(data[:,text_col_num:].mean(axis=0))
        std = np.array(data[:,text_col_num:].std(axis=0))

        if dataset in ['train', 'test']: 
            train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size=0.2, random_state=args.seed)
            if args.val_split:
                train_X, _, train_Y, _ = train_test_split(train_X, train_Y, test_size=0.185, random_state=args.seed)


        if args.train_size != 1: # data usage efficiency
            _, train_X, _, train_Y = train_test_split(train_X, train_Y, test_size=args.train_size, random_state=args.seed)
            print('LESS TRIANING SET')


        epsilon = 1e-9

        if dataset == 'train':
            self.cate = train_X[:,:text_col_num]
            self.cont = (train_X[:,text_col_num:] - mean)/(std + epsilon)
            self.label = train_Y
        elif dataset == 'valid':
            self.cate = valid_X[:,:text_col_num]
            self.cont = (valid_X[:,text_col_num:] - mean)/(std + epsilon)
            self.label = valid_Y
        else:
            self.cate = test_X[:,:text_col_num]
            self.cont = (test_X[:,text_col_num:] - mean)/(std + epsilon)
            self.label = test_Y

        self.label = self.label.tolist()


    def __getitem__(self, idx):
        # idx = copy.deepcopy(self.data_list[idx])
        cont = copy.deepcopy(self.cont[idx])
        cate = copy.deepcopy(self.cate[idx])
        label = copy.deepcopy(self.label[idx])


        return torch.LongTensor(cate), torch.FloatTensor(cont), label
    
    def evaluate(self, label, pred):
        return roc_auc_score(label.item(), pred[:,1].item())

    def __len__(self):
        return len(self.label)




class make_split():

    def __init__(self):
        self.data_list = []
        # self.transform = transform

        file = '/home/ceyu.cy2/datasets/tabular/Income/income_evaluation.csv'
        now = 0
        with open(file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                self.data_list.append(row)
                # print(row)

        cols = [[] for _ in range(15)]
        for each in self.data_list:
            for i in range(15):
                cols[i].append(each[i])


        cont = []
        cate = []
        mapping = [{} for _ in range(15)]
        for i in range(15):
            if i == 14:
                # label index
                for j, each in enumerate(set(cols[i])):
                    mapping[i][each] = j
            elif i in [0, 2, 4, 10, 11, 12]:
                cont.append(cols[i])
            else:
                lens = len(set(cols[i]))
                # print(lens)
                for j, each in enumerate(set(cols[i])):
                    if each in mapping[i]:
                        raise ValueError
                    else:
                        mapping[i][each] = now + j
                now += lens

        n = 1
        for i in range(len(self.data_list)):
            j = 14
            if mapping[j] != {}:
                self.data_list[i][j] = mapping[j][self.data_list[i][j]]
        


        self.label = [each[14] for each in self.data_list]

        negative = [i for i in range(len(self.label)) if self.label[i] == 0]
        positive = [i for i in range(len(self.label)) if self.label[i] == 1]

        random.shuffle(negative)
        random.shuffle(positive)                  

        train = positive[:int(0.65*len(positive))]
        train += negative[:int(0.65*len(negative))]

        val = positive[int(0.65*len(positive)): int(0.8*len(positive))]
        val += negative[int(0.65*len(negative)): int(0.8*len(negative))]

        test = positive[int(0.8*len(positive)): ]
        test += negative[int(0.8*len(negative)): ]

        root = '/home/ceyu.cy2/datasets/tabular/Income/train_val_test/split4/'
        if not os.path.exists(root):
            os.mkdir(root)
        with open(root + 'train.pkl', 'wb') as f:
            pickle.dump(train, f)

        with open(root + 'val.pkl', 'wb') as f:
            pickle.dump(val, f)

        with open(root + 'test.pkl', 'wb') as f:
            pickle.dump(test, f)

class test_args():
    def __init__(self) -> None:
        self.seed = 0
        self.cate_mask_value = 0
        self.split_sets = 2

if __name__ == '__main__':
    # make_split()
    train_data = MyDataset(
        dataset='train',
        args=test_args()
    )
        
    # train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
    # train_iter = iter(train_dataloader)
    # cate, cont, label = next(train_iter)
    
    # print('dataset??')
    # pass

