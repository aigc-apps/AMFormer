import os, sys
import torch
import torch.nn as nn
import numpy as np
import models
from tensorboardX import SummaryWriter

from config.cfg import arg2str
from torch.autograd import Variable
from torchmetrics import Accuracy, AUROC, MeanSquaredError
import copy
from torch.cuda.amp import autocast, GradScaler






class DefaultTrainer(object):

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.lr = self.lr_current = args.lr
        self.start_iter = args.start_iter
        self.max_iter = args.max_iter
        # self.sub_iter = args.max_iter//2 if args.sub_iter == -1 else args.sub_iter
        self.warmup_steps = args.warmup_steps


        # self.model = getattr(models, args.model_name.lower())(args)
        self.model = models.amformer(dim = 192,
                                depth = 3,
                                heads = 8,
                                attn_dropout = 0.2,
                                ff_dropout = 0.1,
                                use_cls_token = True,
                                groups = [120, 120, 120],
                                sum_num_per_group = [32, 32, 32],
                                prod_num_per_group = [6, 6, 6],
                                cluster = True,
                                target_mode = 'mix',
                                token_descent = False, #True,
                                use_prod = True,
                                num_special_tokens = 2,
                                num_unique_categories = 10000,
                                out = 2,
                                num_cont = 104,
                                num_cate = 16,
                                use_sigmoid = True,)










        self.flag = 0
        self.model.cuda()
        # self.max_acc = 0
        # self.min_loss = 1000
        self.scaler = GradScaler()

        self.metrics = {
            'acc':['high', 0],
            'auc':['high', 0],
            'loss':['low', 1000],
            'mse':['low', 1000],
        }
        self.start = 0
        self.wrong = None
        self.log_path = os.path.join(self.args.save_folder, self.args.exp_name, 'result.txt')
        

        print('LR = {}'.format(self.lr))
        

        if args.ckpt not in [None, 'None']:
            state_dict = torch.load(args.ckpt)['net_state_dict']
            self.model.load_state_dict(state_dict, strict=args.ckpt_strict)
            # args.warmup_steps = -1



        params = []
        params_for_pretrain = []
        for keys, param_value in self.model.named_parameters():
            # if 'decoder' not in keys:
            params += [{'params': [param_value], 'lr': self.lr}]
            
            # params_for_pretrain += [{'params': [param_value], 'lr': self.lr * args.lr_ratio}]


        self.optim = torch.optim.Adam(params, lr=self.lr,
                                        betas=(0.9, 0.999), eps=1e-08)
        
        # self.optim2 = torch.optim.Adam(params_for_pretrain, lr=self.lr * args.lr_ratio,
        #                                 betas=(0.9, 0.999), eps=1e-08)
        if args.ckpt not in [None, 'None'] and 'optim' in torch.load(args.ckpt) and args.load_optim:
            # if 'optim' in torch.load(args.ckpt):
            self.optim.load_state_dict(torch.load(args.ckpt)['optim'].state_dict())
            # self.optim = torch.load(args.ckpt)['optim'].state_dict()


        self.grads = []






    def hook_fn(self, grad):
        self.grads.append(grad.abs().max())
        return grad
        


    def train(self, train_dataloader, valid_dataloader=None):
        self.init_writer()

        train_epoch_size = len(train_dataloader)
        train_iter = iter(train_dataloader)
        val_epoch_size = len(valid_dataloader)

        
        self.val_freq = train_epoch_size if self.args.val_freq == -1 else self.args.val_freq
        for step in range(self.start_iter, self.max_iter):
            
            # step = step if self.args.direct_val == False else 500
            # if self.args.direct_val == False:
            if step % train_epoch_size == 0:
                print('Epoch: {} ----- step:{} - train_epoch size:{}'.format(step // train_epoch_size, step,
                                                                            train_epoch_size))
                train_iter = iter(train_dataloader)

            self._adjust_learning_rate_iter(step)
            self.train_iter(step, train_iter)

            if (valid_dataloader is not None) and \
                (step % self.val_freq == 0 or step == self.args.max_iter - 1) and \
                (step > self.args.validation_after):
                
                val_iter = iter(valid_dataloader)
                metric_dict = self.validation(step, val_iter, val_epoch_size)
                
                self.save_best(metric_dict, step)


    def save_best(self, metric_dict, step):
        print(metric_dict)
        for each in metric_dict:
            if (each == 'loss' and metric_dict[each].item() < self.metrics[each][1]):# or (each != 'loss' and metric_dict[each] > self.metrics[each]):
                self.delete_model(best='best_{}'.format(each), index=self.metrics[each][1])
                self.metrics[each][1] = metric_dict[each].item()
                self.save_model(step, best='best_{}'.format(each), index=self.metrics[each][1], gpus=1)   
            if each != 'loss':
                if (self.metrics[each][0] == 'high' and metric_dict[each] > self.metrics[each][1]) or (self.metrics[each][0] == 'low' and metric_dict[each] < self.metrics[each][1]):
                    self.log = open(self.log_path, mode='a')
                    self.metrics[each][1] = metric_dict[each].item()
                    self.log.write('step = {}, metric = {}\n'.format(step, metric_dict))
                    self.log.close()





    def train_main(self, cate, cont, label, step):
        self.model.train()
        self.model.zero_grad()
        self.optim.zero_grad()

        # if step == 3:
        #     print(1)
        if self.args.use_amp == False:
            pred, loss= self.model(cate, cont, label, step=step)
            loss.backward()
            self.optim.step()
            
        else:
            with autocast():
                pred, loss= self.model(cate, cont, label, step=step)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()




        if torch.isnan(loss) or torch.isinf(loss):
            if torch.isnan(loss).any():
                flag = 'nan'
            elif torch.isinf(loss).any():
                flag = 'inf'
            else:
                flag = 'nan_and_inf'
            self.save_model(step, best=flag, index=1, gpus=1)  
            print(flag)
            raise ValueError
        
        return pred, loss

    def train_sub(self, cate, cont, label):
        self.model.train()
        self.optim.zero_grad()
        loss= self.model(cate, cont, label, delta=True)
        loss.backward()
        self.optim2.step()        

    def train_iter(self, step, dataloader):
        
        cate, cont, label= next(dataloader)
        cate, cont, label = cate.cuda(), cont.cuda(), label.cuda()

        pred, loss = self.train_main(cate, cont, label, step)
        if step < self.args.sub_iter:
            self.train_sub(cate, cont, label)


        if self.args.save_metric != 'mse':
            acc = Accuracy(task="multiclass", num_classes=self.args.out, top_k=1).cuda()(pred, label)
            auc = AUROC(task="multiclass", num_classes=self.args.out).cuda()(pred, label)
        else:
            acc = auc = 0

        if step % self.args.display_freq == 0:
            print('Training - Step: {} - Acc: {:.4f} - GPU: {}'.format(step, acc, self.args.gpu_id))
            scalars = [loss.item(), acc, auc, self.lr_current]
            names = ['loss', 'acc', 'AUC', 'lr']
            write_scalars(self.writer, scalars, names, step, 'train')


    def test(self, test_dataloader):
        test_iter = iter(test_dataloader)
        print('============Begin Testing============')
        epoch_size = len(test_dataloader)
        self.model.eval()
        loss_fea = 0.
        loss_recon = 0.
        loss_cls = 0.
        with torch.no_grad():
            for i in range(epoch_size):

                cate, cont, label = next(test_iter)
                cate, cont, label = cate.cuda(), cont.cuda(), label.cuda()

                pred, l1, l2, l3 = self.model(cate, cont, label)

                loss_fea += l1
                loss_recon += l2
                loss_cls += l3

                if i == 0:
                    total_pred = pred
                    total_label = label
                else:
                    total_pred = torch.cat([total_pred, pred], dim=0)
                    total_label = torch.cat([total_label, label], dim=0)

        acc = Accuracy(task="multiclass", num_classes=self.args.out, top_k=1).cuda()(total_pred, total_label)
        auc = AUROC(task='multiclass', num_classes=self.args.out).cuda()(pred, label)

        print(']Loss: {:.4f} \n Acc: {:.4f} \n AUC: {:.4f}'.format((loss_cls).item(), acc, auc))


    def validation(self, step, val_iter, val_epoch_size, dataset='val'):

        print('============Begin Validation============:step:{}'.format(step))

        self.model.eval()
        loss = 0.
        if self.args.use_amp_in_eval:
            with autocast(enabled=True):
                with torch.no_grad():
                    for i in range(val_epoch_size):

                        cate, cont, label= next(val_iter)
                        cate, cont, label= cate.cuda(), cont.cuda(), label.cuda()
                        pred, loss_cel = self.model(cate, cont, label)
                        loss += loss_cel

                        if i == 0:
                            total_pred = pred
                            total_label = label
                        else:
                            total_pred = torch.cat([total_pred, pred], dim=0)
                            total_label = torch.cat([total_label, label], dim=0)
        else:
            with torch.no_grad():
                for i in range(val_epoch_size):

                    cate, cont, label= next(val_iter)
                    cate, cont, label= cate.cuda(), cont.cuda(), label.cuda()
                    pred, loss_cel = self.model(cate, cont, label)
                    loss += loss_cel

                    if i == 0:
                        total_pred = pred
                        total_label = label
                    else:
                        total_pred = torch.cat([total_pred, pred], dim=0)
                        total_label = torch.cat([total_label, label], dim=0)           


        if 'remove' in self.args:
            remove = self.args.remove
            idx = torch.unique(torch.cat([torch.cat(torch.where(total_label.flatten() == label)) for label in self.args.remove]))
            miss_cls_label = torch.gather(total_label.flatten(), 0, idx)
            miss_cls_pred = torch.gather(total_pred, 0, idx.unsqueeze(-1).repeat(1, self.args.out))
            acc_miss = Accuracy(task="multiclass", num_classes=self.args.out, top_k=1).cuda()(miss_cls_pred, miss_cls_label.reshape(-1))
        # if self.args.missing_rate > 0:
        #     pass

        if self.args.save_metric != 'mse':
            acc = Accuracy(task="multiclass", num_classes=self.args.out, top_k=1).cuda()(total_pred, total_label)
            auc = AUROC(task="multiclass", num_classes=self.args.out).cuda()(total_pred, total_label)
            mse = acc
        else:
            auc = 0
            acc = 0
            if self.args.out == 1:
                flag = self.args.label_diverse ** 2 if self.args.use_sigmoid else 1
                mse = MeanSquaredError().cuda()(total_pred.squeeze(1), total_label) * flag
            else:
            # acc = MeanSquaredError().cuda()(torch.argmax(total_pred, dim=1), total_label)
                mse = MeanSquaredError().cuda()((torch.nn.functional.softmax(total_pred, dim=1) * torch.arange(self.args.out).cuda()).sum(dim=1), total_label)
        loss /= len(total_label)
        print('Valid - Step: {} \n Loss: {:.4f}'.format(step, loss.item()))

        scalars = [loss.item(),  acc, auc, mse]
        names = ['loss_simi', 'acc', 'AUC', 'mse']

        if 'remove' in self.args:
            scalars.append(acc_miss)
            names.append('acc_miss')



        write_scalars(self.writer, scalars, names, step, dataset)

        return {'loss': loss, 'acc': acc, 'auc': auc, 'mse':mse}

    def _adjust_learning_rate_iter(self, step):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if step <= self.warmup_steps:  # 增大学习率
            self.lr_current = self.args.lr * float(step) / float(self.warmup_steps)

        if self.args.lr_adjust == 'fix':
            if step in self.args.stepvalues:
                self.lr_current = self.lr_current * self.args.gamma
        elif self.args.lr_adjust == 'poly':
            self.lr_current = self.args.lr * (1 - step / self.args.max_iter) ** 0.9

        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_current

    def init_writer(self):

        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder, exist_ok=True)

        log_path = os.path.join(self.args.save_log, self.args.exp_name)
        log_config_path = os.path.join(log_path, 'configs.log')

        self.writer = SummaryWriter(log_path)
        with open(log_config_path, 'w') as f:
            f.write(arg2str(self.args))


    def delete_model(self, best, index):
        if index == 0 or index == 1000000:
            return
        save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        if os.path.exists(save_path):
            os.remove(save_path)

    def save_model(self, step, best='best_acc', index=None, gpus=1):

        model_save_path = os.path.join(self.args.save_folder, self.args.exp_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        if gpus == 1:
            if isinstance(index, list):
                save_fname = '%s_%s_%s_%s.pth' % (self.model.model_name(), best, index[0], index[1])
            else:
                save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index,
                'optim':self.optim
            }
        else:
            save_fname = '%s_%s_%s.pth' % (self.model.module.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.module.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index,
                'optim':self.optim
            }
        torch.save(save_dict, save_path)
        print(best + ' Model Saved')


def write_scalars(writer, scalars, names, n_iter, tag=None):
    for scalar, name in zip(scalars, names):
        if tag is not None:
            name = '/'.join([tag, name])
        writer.add_scalar(name, scalar, n_iter)
