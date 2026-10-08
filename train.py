import os
import numpy as np
import torch
import torch.utils
import torch.nn as nn
from torch.utils.data import DataLoader
from mydataset import ImageToImage2D, JointTransform2D, correct_dims
from utils import DiceLoss,MyDC,DCloss
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
import random
from newmodel import U_Net
import time

def main():
    seed_value = int(time.time())
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    device = torch.device('cuda')
    for fold in range(5):
        os.makedirs(f"./newmodelend1/fold{fold}", exist_ok=True)

        print('Creating model...')
        model = U_Net(in_ch=3,out_ch=3)
        base_lr = 0.0003
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=3e-5, amsgrad=False)
        model.to(device)

        print("Setting up data...")
        tf_train = JointTransform2D(img_size=256, low_img_size=128, ori_size=256, crop=None, p_flip=0.0, p_rota=0.0,
                                    p_scale=0.0, p_gaussn=0.0,
                                    p_contr=0.0, p_gama=0.0, p_distor=0.0, color_jitter_params=None,
                                    long_mask=True)  # image reprocessing
        tf_val = JointTransform2D(img_size=256, low_img_size=128, ori_size=256, crop=None, p_flip=0.0,
                                  color_jitter_params=None, long_mask=True)

        train_split = f"train_fold_{fold}"
        val_split = f"val_fold_{fold}"

        trainfold = ImageToImage2D("./data", train_split, tf_train, img_size=256)
        valfold = ImageToImage2D("./data", val_split, tf_val, img_size=256)
        trainloader = DataLoader(trainfold, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
        valloader = DataLoader(valfold, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

        print("Starting training")
        max_epoch = 60
        start_epoch = 0
        max_iterations = max_epoch * len(trainloader)
        dice_loss_func = DiceLoss(3)
        ce_loss_func = CrossEntropyLoss()
        muti_dice_loss_func = DCloss()
        iter_num = 0
        best = float('inf')
        best_model_state = None
        for epoch in range(start_epoch + 1, max_epoch + 1):
            model.train()
            for i_batch, sampled_batch in enumerate(trainloader):
                batch_images, batch_labels = sampled_batch['image'], sampled_batch['label']
                batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda().squeeze(dim=1)

                out = model(batch_images)
                ce_loss = ce_loss_func(out, batch_labels.long())
                dice_loss = muti_dice_loss_func(out, batch_labels.long())
                loss = 0.5*ce_loss+0.5*dice_loss

                print(f"Fold{fold}| Epoch[{epoch}|{max_epoch}] Batch{iter_num}: {loss.detach().cpu().numpy()} | Dice Loss: {dice_loss.detach().cpu().numpy()} |CE Loss: {ce_loss.detach().cpu().numpy()}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                iter_num = iter_num + 1

            if epoch >= 35:
                model.eval()
                val_loss = []
                total_correct = 0
                total_pixels = 0
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(valloader):
                        batch_images, batch_labels = sampled_batch['image'], sampled_batch['label']
                        batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda().squeeze(dim=1)
                        
                        pred = model(batch_images)
                        pred_softmax = torch.softmax(pred, dim=1)
                        loss_dice = dice_loss_func(pred_softmax, batch_labels.long())
                        val_loss.append(loss_dice.detach().cpu().numpy())
                        pred_label = torch.argmax(pred, dim=1)
                        total_correct += (pred_label == batch_labels).sum().item()
                        total_pixels += batch_labels.numel()
                    val_loss_dice_mean = np.mean(val_loss)
                    val_accuracy = total_correct / total_pixels
                    print(f"Fold{fold} | Epoch[{epoch}/{max_epoch}] | "f"Val Dice Loss: {val_loss_dice_mean:.6f} | "f"Val Accuracy: {val_accuracy:.6f} "
                          f"({val_accuracy * 100:.2f}%)"
                    )
        
    
                    if val_loss_dice_mean < best:
                        best = val_loss_dice_mean
                        best_epoch = epoch
                        best_model_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        print(f"Best model updated: "f"epoch={best_epoch}, "f"dice_loss={best:.6f}")
        save_dir = os.path.join('./newmodelend1', f'fold{fold}')
        os.makedirs(save_dir, exist_ok=True)
        save_mode_path = os.path.join(save_dir,'epoch_' + str(best_epoch) + 'dice_' + str(best) + '.pth')
        torch.save(best_model_state, save_mode_path)
        print(f"Training finished for fold {fold}. "f"Best epoch: {best_epoch}, "f"Best Dice Loss: {best:.6f}")
        print(f"Best model saved to: {save_mode_path}")


if __name__ == '__main__':
    main()