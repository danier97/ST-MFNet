from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image as imwrite
import os
from os.path import join, exists
import utility
import numpy as np



class VFITex_triplet:
    def __init__(self, db_dir):
        self.seq_list = os.listdir(db_dir)
        self.db_dir = db_dir
        self.transform = transforms.Compose([transforms.ToTensor()])


    def eval(self, model, metrics=['PSNR', 'SSIM'], output_dir=None, output_name=None):
        model.eval()
        results_dict = {k : [] for k in metrics}

        logfile = open(join(output_dir, 'results.txt'), 'w')

        for seq in self.seq_list:
            seqpath = join(self.db_dir, seq)
            if not exists(join(output_dir, seq)):
                os.makedirs(join(output_dir, seq))

            # interpolate between every 2 frames
            gt_list, out_list, inputs_list = [], [], []
            tmp_dict = {k : [] for k in metrics}
            num_frames = len([f for f in os.listdir(seqpath) if f.endswith('.png')])
            for t in range(1, num_frames-5, 2):
                im0 = Image.open(join(seqpath, str(t+2).zfill(3)+'.png'))
                im1 = Image.open(join(seqpath, str(t+3).zfill(3)+'.png'))
                im2 = Image.open(join(seqpath, str(t+4).zfill(3)+'.png'))
                # center crop if 4K
                if '4K' in seq:
                    w, h  = im0.size
                    im0 = TF.center_crop(im0, (h//2, w//2))
                    im1 = TF.center_crop(im1, (h//2, w//2))
                    im2 = TF.center_crop(im2, (h//2, w//2))
                im0 = self.transform(im0).cuda().unsqueeze(0)
                im1 = self.transform(im1).cuda().unsqueeze(0)
                im2 = self.transform(im2).cuda().unsqueeze(0)

                with torch.no_grad():
                    out = model(im0, im2)

                gt_list.append(utility.tensor2rgb(im1))
                out_list.append(utility.tensor2rgb(out))
                if t == 1:
                    inputs_list.append(utility.tensor2rgb(im0))
                inputs_list.append(utility.tensor2rgb(im2))

                for metric in metrics:
                    if metric in ['PSNR' ,'SSIM']:
                        score = getattr(utility, 'calc_{}'.format(metric.lower()))(im1, out)[0].item()
                        tmp_dict[metric].append(score)

                imwrite(out, join(output_dir, seq, 'frame{}.png'.format(t+3)), range=(0, 1))

            # compute sequence-level scores
            for metric in metrics:
                if metric in ['PSNR' ,'SSIM']:
                    results_dict[metric].append(np.mean(tmp_dict[metric]))
                else:
                    print('Metric {} is not supported.'.format(metric))

            # log
            msg = '{:<15s} -- {}'.format(seq, {k: round(results_dict[k][-1], 3) for k in metrics}) + '\n'
            print(msg, end='')
            logfile.write(msg)
        
        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]), 3) for k in metrics}) + '\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()


class VFITex_quintuplet:
    def __init__(self, db_dir):
        self.seq_list = os.listdir(db_dir)
        self.db_dir = db_dir
        self.transform = transforms.Compose([transforms.ToTensor()])


    def eval(self, model, metrics=['PSNR', 'SSIM'], output_dir=None, output_name=None):
        model.eval()
        results_dict = {k : [] for k in metrics}

        logfile = open(join(output_dir, 'results.txt'), 'w')

        for seq in self.seq_list:
            seqpath = join(self.db_dir, seq)
            if not exists(join(output_dir, seq)):
                os.makedirs(join(output_dir, seq))

            # interpolate between every 2 frames
            gt_list, out_list, inputs_list = [], [], []
            tmp_dict = {k : [] for k in metrics}
            num_frames = len([f for f in os.listdir(seqpath) if f.endswith('.png')])
            for t in range(1, num_frames-5, 2):
                im1 = Image.open(join(seqpath, str(t).zfill(3)+'.png'))
                im3 = Image.open(join(seqpath, str(t+2).zfill(3)+'.png'))
                im4 = Image.open(join(seqpath, str(t+3).zfill(3)+'.png'))
                im5 = Image.open(join(seqpath, str(t+4).zfill(3)+'.png'))
                im7 = Image.open(join(seqpath, str(t+6).zfill(3)+'.png'))
                # center crop if 4K
                if '4K' in seq:
                    w, h  = im1.size
                    im1 = TF.center_crop(im1, (h//2, w//2))
                    im3 = TF.center_crop(im3, (h//2, w//2))
                    im4 = TF.center_crop(im4, (h//2, w//2))
                    im5 = TF.center_crop(im5, (h//2, w//2))
                    im7 = TF.center_crop(im7, (h//2, w//2))
                im1 = self.transform(im1).cuda().unsqueeze(0)
                im3 = self.transform(im3).cuda().unsqueeze(0)
                im4 = self.transform(im4).cuda().unsqueeze(0)
                im5 = self.transform(im5).cuda().unsqueeze(0)
                im7 = self.transform(im7).cuda().unsqueeze(0)
                with torch.no_grad():
                    out = model(im1, im3, im5, im7)

                # abandoning boundary frames here
                gt_list.append(utility.tensor2rgb(im4))
                out_list.append(utility.tensor2rgb(out))
                if t == 1:
                    inputs_list.append(utility.tensor2rgb(im3))
                inputs_list.append(utility.tensor2rgb(im5))

                for metric in metrics:
                    if metric in ['PSNR' ,'SSIM']:
                        score = getattr(utility, 'calc_{}'.format(metric.lower()))(im4, out)[0].item()
                        tmp_dict[metric].append(score)
                
                imwrite(out, join(output_dir, seq, 'frame{}.png'.format(t+3)), range=(0, 1))

            # compute sequence-level scores
            for metric in metrics:
                if metric in ['PSNR' ,'SSIM']:
                    results_dict[metric].append(np.mean(tmp_dict[metric]))
                else:
                    print('Metric {} is not supported.'.format(metric))

            # log
            msg = '{:<15s} -- {}'.format(seq, {k: round(results_dict[k][-1], 3) for k in metrics}) + '\n'
            print(msg, end='')
            logfile.write(msg)
        
        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]), 3) for k in metrics}) + '\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()


class Davis90_triplet:
    def __init__(self, db_dir):
        self.seq_list = os.listdir(db_dir)
        self.db_dir = db_dir
        self.transform = transforms.Compose([transforms.ToTensor()])


    def eval(self, model, metrics=['PSNR', 'SSIM'], output_dir=None, output_name=None):
        model.eval()
        results_dict = {k : [] for k in metrics}

        logfile = open(join(output_dir, 'results.txt'), 'w')

        for seq in self.seq_list:
            seqpath = join(self.db_dir, seq)
            if not exists(join(output_dir, seq)):
                os.makedirs(join(output_dir, seq))

            # interpolate between every 2 frames
            gt_list, out_list, inputs_list = [], [], []
            tmp_dict = {k : [] for k in metrics}
            num_frames = len(os.listdir(seqpath))
            for t in range(0, num_frames-6, 2):
                im3 = Image.open(join(seqpath, str(t+2).zfill(5)+'.jpg'))
                im4 = Image.open(join(seqpath, str(t+3).zfill(5)+'.jpg'))
                im5 = Image.open(join(seqpath, str(t+4).zfill(5)+'.jpg'))

                im3 = self.transform(im3).cuda().unsqueeze(0)
                im4 = self.transform(im4).cuda().unsqueeze(0)
                im5 = self.transform(im5).cuda().unsqueeze(0)

                with torch.no_grad():
                    out = model(im3, im5)

                # abandoning boundary frames here
                gt_list.append(utility.tensor2rgb(im4))
                out_list.append(utility.tensor2rgb(out))
                if t == 0:
                    inputs_list.append(utility.tensor2rgb(im3))
                inputs_list.append(utility.tensor2rgb(im5))

                for metric in metrics:
                    if metric in ['PSNR' ,'SSIM']:
                        score = getattr(utility, 'calc_{}'.format(metric.lower()))(im4, out)[0].item()
                        tmp_dict[metric].append(score)

                imwrite(out, join(output_dir, seq, 'frame{}.png'.format(t+3)), range=(0, 1))

            # compute sequence-level scores
            for metric in metrics:
                if metric in ['PSNR' ,'SSIM']:
                    results_dict[metric].append(np.mean(tmp_dict[metric]))
                else:
                    print('Metric {} is not supported.'.format(metric))

            # log
            msg = '{:<15s} -- {}'.format(seq, {k: round(results_dict[k][-1], 3) for k in metrics}) + '\n'
            print(msg, end='')
            logfile.write(msg)
        
        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]), 3) for k in metrics}) + '\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()


class Davis90_quintuplet:
    def __init__(self, db_dir):
        self.seq_list = os.listdir(db_dir)
        self.db_dir = db_dir
        self.transform = transforms.Compose([transforms.ToTensor()])


    def eval(self, model, metrics=['PSNR', 'SSIM'], output_dir=None, output_name=None):
        model.eval()
        results_dict = {k : [] for k in metrics}

        logfile = open(join(output_dir, 'results.txt'), 'w')

        for seq in self.seq_list:
            seqpath = join(self.db_dir, seq)
            if not exists(join(output_dir, seq)):
                os.makedirs(join(output_dir, seq))

            # interpolate between every 2 frames
            gt_list, out_list, inputs_list = [], [], []
            tmp_dict = {k : [] for k in metrics}
            num_frames = len(os.listdir(seqpath))
            for t in range(0, num_frames-6, 2):
                im1 = Image.open(join(seqpath, str(t).zfill(5)+'.jpg'))
                im3 = Image.open(join(seqpath, str(t+2).zfill(5)+'.jpg'))
                im4 = Image.open(join(seqpath, str(t+3).zfill(5)+'.jpg'))
                im5 = Image.open(join(seqpath, str(t+4).zfill(5)+'.jpg'))
                im7 = Image.open(join(seqpath, str(t+6).zfill(5)+'.jpg'))

                im1 = self.transform(im1).cuda().unsqueeze(0)
                im3 = self.transform(im3).cuda().unsqueeze(0)
                im4 = self.transform(im4).cuda().unsqueeze(0)
                im5 = self.transform(im5).cuda().unsqueeze(0)
                im7 = self.transform(im7).cuda().unsqueeze(0)

                with torch.no_grad():
                    out = model(im1, im3, im5, im7)

                # abandoning boundary frames here
                gt_list.append(utility.tensor2rgb(im4))
                out_list.append(utility.tensor2rgb(out))
                if t == 0:
                    inputs_list.append(utility.tensor2rgb(im3))
                inputs_list.append(utility.tensor2rgb(im5))

                for metric in metrics:
                    if metric in ['PSNR' ,'SSIM']:
                        score = getattr(utility, 'calc_{}'.format(metric.lower()))(im4, out)[0].item()
                        tmp_dict[metric].append(score)

                imwrite(out, join(output_dir, seq, 'frame{}.png'.format(t+3)), range=(0, 1))

            # compute sequence-level scores
            for metric in metrics:
                if metric in ['PSNR' ,'SSIM']:
                    results_dict[metric].append(np.mean(tmp_dict[metric]))
                else:
                    print('Metric {} is not supported.'.format(metric))

            # log
            msg = '{:<15s} -- {}'.format(seq, {k: round(results_dict[k][-1], 3) for k in metrics}) + '\n'
            print(msg, end='')
            logfile.write(msg)
        
        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]), 3) for k in metrics}) + '\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()


class Ucf101_triplet:
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.im_list = os.listdir(db_dir)

        self.input3_list = []
        self.input5_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input3_list.append(self.transform(Image.open(join(db_dir, item , 'frame1.png'))).cuda().unsqueeze(0))
            self.input5_list.append(self.transform(Image.open(join(db_dir, item , 'frame2.png'))).cuda().unsqueeze(0))
            self.gt_list.append(self.transform(Image.open(join(db_dir, item , 'framet.png'))).cuda().unsqueeze(0))

    def eval(self, model, metrics=['PSNR', 'SSIM'], output_dir=None, output_name='output.png'):
        model.eval()
        results_dict = {k : [] for k in metrics}
        
        logfile = open(join(output_dir, 'results.txt'), 'a')

        for idx in range(len(self.im_list)):
            if not exists(join(output_dir, self.im_list[idx])):
                os.makedirs(join(output_dir, self.im_list[idx]))

            with torch.no_grad():
                out = model(self.input3_list[idx], self.input5_list[idx])
            gt = self.gt_list[idx]

            for metric in metrics:
                if metric in ['PSNR', 'SSIM']:
                    score = getattr(utility, 'calc_{}'.format(metric.lower()))(gt, out)[0].item()
                    results_dict[metric].append(score)
                else:
                    print('Metric {} is not supported.'.format(metric))

            imwrite(out, join(output_dir, self.im_list[idx], output_name), range=(0, 1))

            msg = '{:<15s} -- {}'.format(self.im_list[idx], {k: round(results_dict[k][-1],3) for k in metrics}) + '\n'
            print(msg, end='')
            logfile.write(msg)

        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]),3) for k in metrics}) + '\n\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()


class Ucf101_quintuplet:
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.im_list = os.listdir(db_dir)

        self.input1_list = []
        self.input3_list = []
        self.input5_list = []
        self.input7_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input1_list.append(self.transform(Image.open(join(db_dir, item , 'frame0.png'))).cuda().unsqueeze(0))
            self.input3_list.append(self.transform(Image.open(join(db_dir, item , 'frame1.png'))).cuda().unsqueeze(0))
            self.input5_list.append(self.transform(Image.open(join(db_dir, item , 'frame2.png'))).cuda().unsqueeze(0))
            self.input7_list.append(self.transform(Image.open(join(db_dir, item , 'frame3.png'))).cuda().unsqueeze(0))
            self.gt_list.append(self.transform(Image.open(join(db_dir, item , 'framet.png'))).cuda().unsqueeze(0))

    def eval(self, model, metrics=['PSNR', 'SSIM'], output_dir=None, output_name='output.png'):
        model.eval()
        results_dict = {k : [] for k in metrics}
        
        logfile = open(join(output_dir, 'results.txt'), 'a')

        for idx in range(len(self.im_list)):
            if not exists(join(output_dir, self.im_list[idx])):
                os.makedirs(join(output_dir, self.im_list[idx]))

            with torch.no_grad():
                out = model(self.input1_list[idx], self.input3_list[idx], self.input5_list[idx], self.input7_list[idx])
            gt = self.gt_list[idx]

            for metric in metrics:
                if metric in ['PSNR', 'SSIM']:
                    score = getattr(utility, 'calc_{}'.format(metric.lower()))(gt, out)[0].item()
                    results_dict[metric].append(score)
                else:
                    print('Metric {} is not supported.'.format(metric))

            imwrite(out, join(output_dir, self.im_list[idx], output_name), range=(0, 1))

            msg = '{:<15s} -- {}'.format(self.im_list[idx], {k: round(results_dict[k][-1],3) for k in metrics}) + '\n'
            print(msg, end='')
            logfile.write(msg)

        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]),3) for k in metrics}) + '\n\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()


class Snufilm_extreme_quintuplet:
    def __init__(self, db_dir, mode='extreme'):
        self.db_dir = db_dir[:-19]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.mode = mode
        self.input1_list = []
        self.input3_list = []
        self.input5_list = []
        self.input7_list = []
        self.gt_list = []
        with open(join(self.db_dir, 'test-{}.txt'.format(mode)), 'r') as f:
            self.triplet_list = f.read().splitlines()

    def eval(self, model, metrics=['PSNR', 'SSIM'], output_dir=None, output_name='output.png'):
        model.eval()
        results_dict = {k : [] for k in metrics}
        
        logfile = open(join(output_dir, 'results.txt'), 'a')

        for i, triplet in enumerate(self.triplet_list, 1):
            lst = triplet.split(' ')
            lst = self.get_quintuplet(lst)
            try:
                im1 = self.transform(Image.open(join(self.db_dir, lst[0]))).cuda().unsqueeze(0)
                im3 = self.transform(Image.open(join(self.db_dir, lst[1]))).cuda().unsqueeze(0)
                im4 = self.transform(Image.open(join(self.db_dir, lst[2]))).cuda().unsqueeze(0)
                im5 = self.transform(Image.open(join(self.db_dir, lst[3]))).cuda().unsqueeze(0)
                im7 = self.transform(Image.open(join(self.db_dir, lst[4]))).cuda().unsqueeze(0)
            except:
                # skip boundary cases
                continue

            with torch.no_grad():
                out = model(im1, im3, im5, im7)

            for metric in metrics:
                if metric in ['PSNR', 'SSIM']:
                    score = getattr(utility, 'calc_{}'.format(metric.lower()))(im4, out)[0].item()
                    results_dict[metric].append(score)
                else:
                    print('Metric {} is not supported.'.format(metric))

            if not exists(join(output_dir, '{}-{}'.format(self.mode, str(i).zfill(3)))):
                os.makedirs(join(output_dir, '{}-{}'.format(self.mode, str(i).zfill(3))))
            imwrite(out, join(output_dir, '{}-{}'.format(self.mode, str(i).zfill(3)), output_name), range=(0, 1))

            msg = '{:<15s} -- {}'.format('{}-{}'.format(self.mode, str(i).zfill(3)), {k: round(results_dict[k][-1],3) for k in metrics}) + '\n'
            print(msg, end='')
            logfile.write(msg)

        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]),3) for k in metrics}) + '\n\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()

    
    def get_quintuplet(self, lst):
        """
        lst -- list of paths of a triplet
        """
        if self.mode == 'extreme':
            offset = 16
        elif self.mode == 'hard':
            offset = 8
        elif self.mode == 'medium':
            offset = 4
        else:
            offset = 2
        im3_idx_str = lst[0].split('/')[-1].split('.')[0]
        im1_idx_str = str(int(im3_idx_str) - offset).zfill(len(im3_idx_str))
        im7_idx_str = str(int(im3_idx_str) + offset*2).zfill(len(im3_idx_str))
        im1_pth = '/'.join([item if not item.endswith('.png') else im1_idx_str+'.png' for item in lst[0].split('/')])
        im7_pth = '/'.join([item if not item.endswith('.png') else im7_idx_str+'.png' for item in lst[0].split('/')])
        return [im1_pth, *lst, im7_pth]


class Snufilm_easy_quintuplet(Snufilm_extreme_quintuplet):
    def __init__(self, db_dir):
        db_dir = db_dir+'xxx'
        super(Snufilm_easy_quintuplet, self).__init__(db_dir, mode='easy')

class Snufilm_medium_quintuplet(Snufilm_extreme_quintuplet):
    def __init__(self, db_dir):
        db_dir = db_dir+'x'
        super(Snufilm_medium_quintuplet, self).__init__(db_dir, mode='medium')

class Snufilm_hard_quintuplet(Snufilm_extreme_quintuplet):
    def __init__(self, db_dir):
        db_dir = db_dir+'xxx'
        super(Snufilm_hard_quintuplet, self).__init__(db_dir, mode='hard')