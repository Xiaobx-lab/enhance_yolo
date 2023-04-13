import os
from webbrowser import get
import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from utils.utils import get_lr
from util_funs import ssim, psnr


def fit_one_epoch(model_train, model, enhance_loss,yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss_enhance = 0
    loss_enhance_val = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, images_clear, targets = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                images_clear  = images_clear.cuda(local_rank)
                save_image(images_clear,'train_clear.jpg')
                targets = [ann.cuda(local_rank) for ann in targets]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()

        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)
            enhance_image   = outputs
            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            en_loss = enhance_loss(enhance_image, images_clear)
            en_loss.backward()
            
            optimizer.step()
        else:
            pass

        loss_enhance += en_loss.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{
                                'enhance_loss': loss_enhance / (iteration + 1),
                                'lr'    : get_lr(optimizer),
                               })
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    ssims = []
    psnrs = []
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, images_clear, targets = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                images_clear  = images_clear.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                if iteration < 10:
                    save_image(images, 'input.jpg')
                    save_image(images_clear, 'clear.jpg')
            optimizer.zero_grad()
            outputs         = model_train(images)
            enhance_image = outputs
            ssim_ = ssim(enhance_image, images_clear).item()  # 真实值
            psnr_ = psnr(enhance_image, images_clear)
            ssims.append(ssim_)
            psnrs.append(psnr_)
            if iteration < 10:
                save_image(enhance_image,'enhance.jpg')
            outputs = outputs[1:]

            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            en_loss_val = enhance_loss(enhance_image, images_clear)        
        ssim_result = np.mean(ssims)
        psnr_result = np.mean(psnrs)

        loss_enhance_val += en_loss_val
        if local_rank == 0:
            pbar.set_postfix(**{
                                'en_loss_val': loss_enhance_val / (iteration + 1)
            })
            pbar.update(1)
    print('ssim_result',ssim_result,'psnr_result',psnr_result)
 
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
     #   loss_history.append_loss(epoch + 1,  en_loss.item(), en_loss_val.item())
     #   eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('en_loss_train: %.3f || en_loss_val: %.3f' % (loss_enhance/epoch_step , loss_enhance_val/epoch_step_val))
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights_detector.pth"))