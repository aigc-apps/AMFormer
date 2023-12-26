import torch
import torchvision.transforms as transforms
from config.cfg import BaseConfig
import training 
import os
from utils import data_load
import numpy as np
from time import time

os.environ["mapreduce_input_fileinputformat_split_maxsize"] = "64" 


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def main(args):
    runseed = args.seed
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.manual_seed(runseed)
    np.random.seed(runseed)

    dest_dir = os.path.join(args.save_folder, args.data_name, args.model_name, args.exp_name)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        args.exp_name = os.path.join(args.data_name, args.model_name, args.exp_name, 'ver0')
    else:
        num = 1 if os.listdir(dest_dir) == [] else max([int(x[3:]) for x in os.listdir(dest_dir)])+1
        args.exp_name = os.path.join(args.data_name, args.model_name, args.exp_name, 'ver'+str(num))
        # os.makedirs(args.exp_name)
    print('================ {:^30s} ================'.format(args.exp_name.split('/')[-1]))
    

    # if args.single_optim == True:
    #     trainer = SingleDefaultTrainer(args)
    # else:
    #     trainer = DefaultTrainer(args)
    


    dataset = getattr(data_load, args.data_name.lower())
    train_data = dataset(
        dataset='train',
        args=args
    )
    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True)
    print('================ {:^30s} ================'.format('train set loaded'))
    val_data = dataset(
        dataset=args.validation_set,
        args=args
    )
    val_load = torch.utils.data.DataLoader(val_data, batch_size=max(args.batch_size, args.val_batch_size), shuffle=False, num_workers=args.num_workers)
    print('================ {:^30s} ================'.format('valid set loaded'))

    trainer = getattr(training, args.trainer.lower())(args)
    trainer.train(train_load, val_load)



if __name__ == '__main__':
    print('================ {:^30s} ================'.format('Loading Config'))

    # for i in range(4):
        # print('round = {}'.format(i))
    args = BaseConfig()
    args = args.initialize()
        # args.seed = args.seed + i
        # print(i)
        # print(args.config)
        
        # raise ValueError
    main(args)
    


    

