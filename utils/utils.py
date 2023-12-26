import torch
from torch.autograd import Variable
import numpy as np
import torch
import sys
import time
import os
import torch.distributed as dist
# import tensorflow as tf
# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 60
term_width = int(term_width)
# tensorboard --host 192.168.0.240 --port 6008 --logdir
TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):

    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def label_to_variable(labels, local_rank, gpu=True, volatile=False):
    '''
    :param blobs:
    :param volatile:
    :return:
      cls_targets, loc_targets
    '''
    if not torch.is_tensor(labels):
        with torch.no_grad():
            labels = Variable(torch.from_numpy(labels))
            # v = v.squeeze()
    # else:
    #     v = labels
    # if gpu:
    #     v = v.cuda()

    v = labels.to(local_rank)
    return v


def img_data_to_variable(blobs, local_rank, gpu=True, volatile=False):
    '''
      :param blobs:
      :param volatile:
      :return:
        cls_targets, loc_targets
    '''
    # if isinstance(blobs, tuple) or isinstance(blobs, list):
    #     v = []
    #     with torch.no_grad():
    #         images_slow = Variable(blobs[0])
    #         images_fast = Variable(blobs[1])
    #     if gpu:
    #         images_slow = images_slow.cuda()
    #         images_fast = images_fast.cuda()
    #     v = [images_slow, images_fast]
    # else:
    #     images = blobs
    #     with torch.no_grad():
    #         v = Variable(images)
    #     if gpu:
    #         v = v.cuda()
    if isinstance(blobs, list) or isinstance(list, list):
        img = [each.to(local_rank) for each in blobs]
        # img[0], img = blobs.to(local_rank)
    else:
        img = blobs.to(local_rank)
    return img


def list2onehot(target):
    out = []
    for i in range(len(target)):
        out.append([1.0, 0.0] if target[i] == 0 else [0.0, 1.0])
    return out


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    logits_gather_list = [torch.zeros_like(rt) for _ in range(dist.get_world_size())]
    dist.all_gather(logits_gather_list, rt)
    return torch.cat(logits_gather_list)


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def cal_loss(lossfn, pred, target, **kwargs):
    # print(pred)
    if kwargs['lam'] is not None:# or 'mixup' in kwargs:
        label_a, label_b = target
        return mixup_criterion(lossfn, pred, label_a, label_b, lam=kwargs['lam'])
    else:
        return lossfn(pred, target)
    


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('================ {:^30s} ================'.format('cost {:.3f} seconds'.format(end_time - start_time)))
        return result
    return wrapper


if __name__ == '__main__':
    y = [1, 1, 1, 0, 0, 0]
    y1 = torch.FloatTensor(list2onehot(y))
    print(y1)
