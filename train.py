from data.datasets import *
from trainers.quintuplet import Trainer
from torch.utils.data import DataLoader
import argparse
import torch
import models
import losses
import datetime
from os.path import join

parser = argparse.ArgumentParser(description='STMFNet')

# parameters
# model
parser.add_argument('--net', type=str, default='STMFNet')

# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--data_dir', type=str, help='root dir for all datasets')
parser.add_argument('--out_dir', type=str, default='train_results')
parser.add_argument('--load', type=str, default=None)

# Learning Options
parser.add_argument('--epochs', type=int, default=70, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--loss', type=str, default='1*Lap', help='loss function configuration')
parser.add_argument('--patch_size', type=int, default=256, help='crop size')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type, other options include plateau')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--patience', type=int, default=None, help='number of epochs without improvement after which lr will be reduced for plateau scheduler')
parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Options for feature extractor
parser.add_argument('--featc', nargs='+', type=int, default=[64, 128, 256, 512])
parser.add_argument('--featnet', type=str, default='UMultiScaleResNext')
parser.add_argument('--featnorm', type=str, default='batch')
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--finetune_pwc', dest='finetune_pwc', default=False,  action='store_true')


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    # training sets
    vimeo90k_train = Vimeo90k_quintuplet(join(args.data_dir, 'vimeo_septuplet'), train=True,  crop_sz=(args.patch_size,args.patch_size))
    bvidvc_train = BVIDVC_quintuplet(join(args.data_dir, 'bvidvc'), crop_sz=(args.patch_size,args.patch_size))

    # validation set
    vimeo90k_valid = Vimeo90k_quintuplet(join(args.data_dir, 'vimeo_septuplet'), train=False,  
                            crop_sz=(args.patch_size,args.patch_size), augment_s=False, augment_t=False)

    datasets_train = [vimeo90k_train, bvidvc_train]
    train_sampler = Sampler(datasets_train, iter=True)

    # data loaders
    train_loader = DataLoader(dataset=train_sampler, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=vimeo90k_valid, batch_size=args.batch_size, num_workers=0)
    
    # model and loss function
    model = getattr(models, args.net)(args).cuda()
    print('******Model created******')
    
    loss = losses.Loss(args)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    my_trainer = Trainer(args, train_loader, valid_loader, model, loss, start_epoch)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(join(args.out_dir, 'config.txt'), 'a') as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

    while not my_trainer.terminate():
        my_trainer.train()
        my_trainer.save_checkpoint()
        my_trainer.validate()


if __name__ == "__main__":
    main()
