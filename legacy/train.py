import numpy as np
import os
import argparse
import torch
import torch.optim as optim
import datetime

import torch as tc

from model import ModelWrapper
from utils import create_train_val_cal_loaders

Tensor = torch.cuda.FloatTensor


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_vdvae(config, images, labels):
    device = torch.device(f"cuda:0")
    best_epoch = 0

    # These are the directories to store trained models and vae masks
    if not os.path.exists('pretrained'):
        os.makedirs('pretrained')
    if not os.path.exists('vae_masks'):
        os.makedirs('vae_masks')

    # Get the time stamp
    ft = "%Y_%m_%d_%H_%M_%S"
    timestamp = datetime.datetime.now().strftime(ft)

    model = ModelWrapper(config['input_size'],
                         z_dim=config['z_dim'],
                         start_dims=config['start_dims'],
                         continuous=config['continuous']).to(device)

    num_epochs = config['epochs']

    train_loader, val_loader, cal_loader = create_train_val_cal_loaders(images, labels,
                                                                        batch_size=config['batch_size'],
                                                                        continuous=config['continuous'])

    # Other optimisers work as well, Adamax is quite stable though
    optimizer = optim.Adamax(model.parameters(), weight_decay=config['wd'], lr=config['lr'])

    print('NUM PARAMS: {}'.format(count_parameters(model)))
    print(f'NUM EPOCHS: {num_epochs}')

    best_loss = 1e30
    best_acc = 0
    best_lk = 1e30
    global_step = 0

    for epoch in range(num_epochs):
        model.zero_grad()
        train_acc = 0

        # The trackers for the mean and scale of the inference map
        vae_mask = np.zeros((config['input_size'], config['input_size'], config['input_size']))
        vae_scale = np.zeros((config['input_size'], config['input_size'], config['input_size']))

        for (x, y) in train_loader:
            optimizer.zero_grad()

            x = x.type(Tensor).to(device)
            y = y.type(Tensor).to(device)

            ret_dict = model(x, y)

            loss = ret_dict['loss'].mean()

            loss.backward()
            optimizer.step()

            vae_mask += np.squeeze(ret_dict['mean_mask'].cpu().data.numpy())
            vae_scale += np.squeeze(ret_dict['mask_scale'].cpu().data.numpy())
            train_acc += 1
            global_step += 1

        vae_mask = vae_mask / train_acc
        val_mask = tc.from_numpy(vae_mask).type(Tensor).to(device).view(1, 1,
                                                                        config['input_size'],
                                                                        config['input_size'],
                                                                        config['input_size'])
        vae_scale = vae_scale / train_acc
        val_scale = tc.from_numpy(vae_scale).type(Tensor).to(device).view(1, 1,
                                                                        config['input_size'],
                                                                        config['input_size'],
                                                                        config['input_size'])

        val_acc = 0
        accuracy_acc = 0
        loss_acc = 0
        likelihood_acc = 0
        kld_acc = 0
        recon_acc = 0
        with torch.no_grad():
            for (x, y) in val_loader:
                x = x.type(Tensor).to(device)
                y = y.type(Tensor).to(device)

                ret_dict = model(x, y,
                                 provided_mask=val_mask,
                                 provided_scale=val_scale,
                                 val=True)

                loss_acc += ret_dict['loss'].mean().item()
                val_acc += 1
                likelihood_acc += ret_dict['mask_ll'].item()
                accuracy_acc += ret_dict['acc'].item()
                kld_acc += ret_dict['kl'].item()
                recon_acc += ret_dict['recon_ll'].item()

        loss = loss_acc / val_acc
        lk = likelihood_acc / val_acc
        acc = round(accuracy_acc / val_acc, 4)
        kl = round(kld_acc / val_acc, 3)
        rec = recon_acc / val_acc

        print(f'Epoch: {epoch}, mask likelihood: {lk}, KL: {kl}, accuracy: {acc}, recon likelihood: {rec}')

        if lk < best_lk:
            best_loss = loss
            best_lk = lk
            best_acc = acc
            best_recon = recon_acc
            best_epoch = epoch
            torch.save(model, f"pretrained/{timestamp}.pth")
            np.save(f'vae_masks/{timestamp}.npy', vae_mask)
            np.save(f'vae_masks/{timestamp}_scale.npy', vae_scale)

        if epoch % 10 == 0:
            print(f'Best acc: {best_acc}, likelihood: {best_lk}, epoch: {best_epoch}')

    print(f'Best acc: {best_acc}, likelihood: {best_loss}, epoch: {best_epoch}')
    print('TRAINING DONE, CALIBRATING THE BEST MODEL')

    model = torch.load(f"pretrained/{timestamp}.pth")
    model.eval()
    vae_mask = np.load(f'vae_masks/{timestamp}.npy')

    best_threshold = 0
    best_likelihood = 1e30
    threshold_range = np.linspace(0.95, 0.99, num=20)
    for thresh in threshold_range:
        t = np.quantile(vae_mask, thresh)
        bin_res = (vae_mask > t) * vae_mask
        with torch.no_grad():
            counter = 0
            likelihood = 0
            for (x, y) in cal_loader:
                x = x.type(Tensor).to(device)
                y = y.type(Tensor).to(device)
                ret_dict = model(x, y,
                                 calibrate=True,
                                 t=float(thresh))

                likelihood += ret_dict['mask_ll']
                counter += 1
            likelihood = likelihood / counter
            if likelihood < best_likelihood:
                best_likelihood = likelihood
                best_threshold = thresh

    t = np.quantile(vae_mask, best_threshold)
    thresholded_mask = (vae_mask > t) * vae_mask

    # Save the thresholded mask
    np.save(f'vae_masks/thresholded_{timestamp}.npy', thresholded_mask)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('-d', required=True)
    # parser.add_argument('-c', required=True)
    #
    # args = parser.parse_args()

    # Currently generate fake images and labels, replace with your own data
    images = np.random.uniform(0, 1, (1000, 32, 32, 32))
    labels = np.random.uniform(0, 1, (1000, 1))

    # This config works pretty well, but context dependent
    config = dict(input_size=32,
                  z_dim=128,
                  start_dims=16,
                  continuous=True,
                  epochs=1000,
                  batch_size=500
                  )

    train_vdvae(config, images, labels)
