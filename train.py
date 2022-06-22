import sys

import numpy as np
import torch
import wandb
import logging
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import  torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.localization_network import LocalizationNet
import util
from utils.dice_loss import dice_coeff

# from model.unet_model import UNet
from model.vnet import VNet
from model.unet3d import UNet3D
from model.unet import UNet
# from utils.data_loading import BasicDataLoader
from utils.dataloader import ImageDataset
# from utils.loss_dice import dice_loss
import SimpleITK as sitk

img_dir = 'E:\\Data\\spine\\test_4'
dir_checkpoint = './checkpoints/'




def train_net(net,
              device,
              epochs : int = 10,
              batch_size: int = 1,
              learning_rate : float = 1e-5,
              val_percent : float = 0.1,
              save_checkpoint : bool = True,
              img_scale : float = 1.0,
              amp : bool = False):

    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)
    dataset = ImageDataset(img_dir=img_dir)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='3DU-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))
    writer = SummaryWriter(log_dir= 'logs')


    logging.info(f'''Starting training:
                Epochs:          {epochs}
                Batch size:      {batch_size}
                Learning rate:   {learning_rate}
                Training size:   {n_train}
                Validation size: {n_val}
                Checkpoints:     {save_checkpoint}
                Device:          {device.type}
                Images scaling:  {img_scale}
                Mixed Precision: {amp}
            ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=0.0001, eps=1e-08)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = dice_coeff
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    min_val_loss = 1.0
    for epoch in range(epochs):
        # net.train()
        epoch_loss = 0
        evaluate_loss = 0
        batch_num = 0
        evaluate_num = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                batch_num += 1
                im = batch['im'].to(device=device, dtype=torch.float32)
                msk = batch['msk'].to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast():
                    pred_msk = net(im)
                    loss = dice_coeff(pred_msk, msk)


                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()



                pbar.update(im.shape[0])
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})



        with tqdm(total=n_val, desc=f'Eval Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            net.eval()
            for batch in val_loader:
                evaluate_num += 1
                im = batch['im'].to(device=device, dtype=torch.float32)
                msk = batch['msk'].to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast():
                    pred_msk = net(im)
                    loss = dice_coeff(pred_msk, msk)
                pbar.update(im.shape[0])
                evaluate_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

        scheduler.step(evaluate_loss / evaluate_num)
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        print(f'epochs loss : {epoch_loss / batch_num}')
        print(f'evaluate loss : {evaluate_loss / evaluate_num}')
        # logging.info(f'epochs loss : {epoch_loss / batch_num}')
        # logging.info(f'evaluate loss : {evaluate_loss / evaluate_num}')

        if save_checkpoint and evaluate_loss / evaluate_num < min_val_loss:
            min_val_loss = evaluate_loss / evaluate_num
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint + 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, ormat='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # net = VNet()
    # net = UNet(in_channel=1, out_channel=13)
    # net = UNet3D(in_channels=1, out_channels=15, init_features=8)
    # logging.info(f'Network:\n'
    #              f'\t{net.n_channels} input channels\n'
    #              f'\t{net.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    net = LocalizationNet()

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=100,
                  batch_size=1,
                  learning_rate=0.005,
                  device=device,
                  img_scale=1,
                  val_percent=0.2,
                  amp=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)