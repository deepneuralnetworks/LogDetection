import os
import random
import pandas as pd
import numpy as np
import torch
from torchmetrics import F1Score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def f1score(output, target):
    f1 = F1Score(num_classes=4).to(device)
    res = f1(output, target)
    return res

class ResultLogger(object):
    def __init__(self, tokenizer):
        self.log = pd.DataFrame(columns=['logtype', 'message', 'pred', 'label'])
        self.tokenizer = tokenizer
        self.type_decode = {
                        0:'DELL LC ', 1:'xid', 2:'sxid',
                        3:'Infiniband switch', 4:'HPE IML', 5:'LSF', 6:'syslog'
                        }

    def update(self, logtype, message, pred, label):
        
        # Decode messages (from token to string)
        message = self.tokenizer.batch_decode(message)

        # Argmax
        pred = torch.argmax(pred, dim=1)
        
        # Update content
        content = pd.DataFrame(
            {
            'logtype':logtype.view(-1).tolist(),
            'message':message,
            'pred':pred.view(-1).tolist(),
            'label':label.view(-1).tolist()
            }
        )
        self.log = pd.concat([self.log, content], ignore_index=True)

    def save(self, epoch, save_path='./'):
        # Define file path and make the directory to save
        os.makedirs(save_path, exist_ok=True)
        save_dir = os.path.join(save_path, f'prediction_results_(epoch{epoch:02d}).csv')
        
        # Decode logtype from id to string
        self.log['logtype'] = self.log.logtype.apply(lambda x: self.type_decode[x])
        
        # Save dataframe to csv file
        self.log.to_csv(save_dir, index=False)

        # Reset dataframe
        self.reset_log()

    def reset_log(self):
        # Reset dataframe
        self.log = pd.DataFrame(columns=['logtype', 'message', 'pred', 'label'])

