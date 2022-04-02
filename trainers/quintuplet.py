import os
import utility
import torch



class Trainer:
    def __init__(self, args, train_loader, valid_loader, model, loss, start_epoch=0):
        self.args = args
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.valid_loader = valid_loader
        self.model = model
        self.loss = loss
        self.current_epoch = start_epoch

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        self.out_dir = args.out_dir
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        self.ckpt_dir = os.path.join(args.out_dir, 'checkpoint')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
 
        self.logfile = open(os.path.join(args.out_dir, 'log.txt'), 'a', buffering=1)
        self.logfile.write('\n********STARTING FROM EPOCH {}********\n'.format(self.current_epoch))


    def train(self):
        # Train
        self.model.train()
        psnr_list = []
        for batch_idx, (frame1, frame3, frame4, frame5, frame7) in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()

            frame1 = frame1.cuda()
            frame3 = frame3.cuda()
            frame5 = frame5.cuda()
            frame7 = frame7.cuda()
            frame4 = frame4.cuda()

            output = self.model(frame1, frame3, frame5, frame7)

            loss = self.loss(output, frame4, [frame3, frame5])
            loss.backward()
            self.optimizer.step()

            psnr_list.append(utility.calc_psnr(frame4, output['frame1']).detach()) # (B,)
            if batch_idx % max((self.max_step//5),1) == 0:
                msg = '{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ', \
                                                                            '[' + str(self.current_epoch+1) + '/' + str(self.args.epochs) + ']', \
                                                                            'Step: ', '[' + str(batch_idx) + '/' + str(self.max_step) + ']', \
                                                                            'train loss: ', loss.item())
                print(msg)
                self.logfile.write(msg+'\n')

        self.cur_train_psnr = torch.cat(psnr_list).mean().item()

        self.current_epoch += 1
        if self.args.decay_type != 'plateau':
            self.scheduler.step()

    def validate(self):
        # Validate
        self.model.eval()
        psnr_list, ssim_list = [], []
        for frame1, frame3, frame4, frame5, frame7 in self.valid_loader:
            with torch.no_grad():
                frame1 = frame1.cuda()
                frame3 = frame3.cuda()
                frame5 = frame5.cuda()
                frame7 = frame7.cuda()
                frame4 = frame4.cuda()

                output = self.model(frame1, frame3, frame5, frame7)

            psnr_list.append(utility.calc_psnr(frame4, output)) # (B,)
            ssim_list.append(utility.calc_ssim(frame4, output)) # (B,)

        psnr = torch.cat(psnr_list).mean().item()
        ssim = torch.cat(ssim_list).mean().item()
        msg = 'Train Epoch: ' +  '['  + str(self.current_epoch) + '/' + str(self.args.epochs) + ']\t' + \
              'Train PSNR: ' + '{:<3.2f}\t'.format(self.cur_train_psnr) + \
              'Valid PSNR: ' + '{:<3.2f}'.format(psnr) + \
              'Valid SSIM: ' + '{:<3.2f}'.format(ssim)
        print(msg)
        self.logfile.write(msg+'\n')
        if self.args.decay_type == 'plateau':
            self.scheduler.step(psnr)

    
    def save_checkpoint(self):
        torch.save({'epoch': self.current_epoch, 'state_dict': self.model.state_dict()}, \
                   os.path.join(self.ckpt_dir, 'model_epoch'+str(self.current_epoch).zfill(3)+'.pth'))

    def terminate(self):
        end = (self.current_epoch >= self.args.epochs)
        if end:
            self.logfile.close()
        return end
