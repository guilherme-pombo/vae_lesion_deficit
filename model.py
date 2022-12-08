import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

# Define two globals
bce_fn = nn.BCELoss(reduction='none')
Tensor = torch.cuda.FloatTensor


def add_coords(x, just_coords=False):
    '''
    This just the Uber CoordConv method extended to 3D. Definitely use it on the input
    Using it on other layers of the model can be helpful, but it slows down training
    :param x:
    :param just_coords:
    :return:
    '''
    batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = x.shape
    xx_ones = torch.ones([1, 1, 1, 1, dim_x])
    yy_ones = torch.ones([1, 1, 1, 1, dim_y])
    zz_ones = torch.ones([1, 1, 1, 1, dim_z])

    xy_range = torch.arange(dim_y).float()
    xy_range = xy_range[None, None, None, :, None]
    yz_range = torch.arange(dim_z).float()
    yz_range = yz_range[None, None, None, :, None]
    zx_range = torch.arange(dim_x).float()
    zx_range = zx_range[None, None, None, :, None]

    xy_channel = torch.matmul(xy_range, xx_ones)
    xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)
    xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1, 1)

    yz_channel = torch.matmul(yz_range, yy_ones)
    yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
    yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)
    yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1, 1)

    zx_channel = torch.matmul(zx_range, zz_ones)
    zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
    zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)
    zz_channel = zz_channel.repeat(batch_size_shape, 1, 1, 1, 1)

    xx_channel = xx_channel.to(x.device)
    yy_channel = yy_channel.to(x.device)
    zz_channel = zz_channel.to(x.device)

    xx_channel = xx_channel.float() / (dim_x - 1)
    yy_channel = yy_channel.float() / (dim_y - 1)
    zz_channel = zz_channel.float() / (dim_z - 1)

    if just_coords:
        out = torch.cat([xx_channel, yy_channel, zz_channel], dim=1)
    else:
        out = torch.cat([x, xx_channel, yy_channel, zz_channel], dim=1)

    return out


class SBlock(nn.Module):

    def __init__(self, in_planes, planes, downsample=False, ks=3, stride=1, upsample=False, add_coords=False):
        '''
        This is the Convolutional block that constitutes the meat of the Encoder and Decoder
        :param in_planes:
        :param planes:
        :param downsample:
        :param ks:
        :param stride:
        :param upsample:
        :param add_coords:
        '''
        super(SBlock, self).__init__()
        self.downsample = downsample
        self.upsample = upsample

        if ks == 3:
            pad = 1
        elif ks == 5:
            pad = 2
        else:
            pad = 3

        if add_coords:
            in_planes += 3
        self.add_coords = add_coords

        self.c1 = nn.Sequential(nn.Conv3d(in_planes, planes, kernel_size=ks, stride=stride,
                                          padding=pad),
                                nn.BatchNorm3d(planes),
                                nn.GELU())

        self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):

        if self.add_coords:
            x = add_coords(x)

        out = self.c1(x)

        if self.downsample:
            out = F.avg_pool3d(out, kernel_size=2, stride=2)

        if self.upsample:
            out = self.upsample_layer(out)

        return out


class VAE(nn.Module):

    def __init__(self, input_size, sd=16, z_dim=20, out_chans=1, in_chans=1):
        '''
        This is the VAE model that does the lesion deficit mapping inference. It does two tasks with a single latent.
        First it produces the lesion-deficit map. Second it produces a reconstruction of the lesions.
        Both of these are necessary because we are modelling the joint distribution P(X,Y)
        There are many architectural improvements that will probably help get better accuracy, but this is a simple
        architecture that works even with little data. The more data you have, the more you might want to replace
        the Encoder and Decoder with something more complicated. Or even use a VDVAE
        Adding coordinates helps as well, but by default the models doesn't add them
        :param input_size:
        :param sd:
        :param z_dim:
        :param out_chans:
        :param in_chans:
        '''
        super(VAE, self).__init__()

        self.sd = sd
        self.z_dim = z_dim

        '''
        Encoder -- You'll probably need to tweak this to get the best results, GPU memory usage, etc.
        '''
        self.enc1 = SBlock(in_chans, sd, downsample=True)
        self.enc2 = nn.Sequential(SBlock(sd, sd * 2, downsample=True))
        self.enc3 = nn.Sequential(SBlock(sd * 2, sd * 4, downsample=True))
        self.enc4 = nn.Sequential(SBlock(sd * 4, sd * 8, downsample=True))
        self.enc5 = nn.Sequential(SBlock(sd * 8, sd * 16, downsample=True))

        # These are the dimensions of a fully connect latent at the end of the encoder
        self.spatial_dims = input_size // (2 ** 5)
        self.dense_dims = self.spatial_dims ** 3 * (sd * 16)

        '''
        Parameters of the latent space
        '''
        self.mu = nn.Linear(self.dense_dims, z_dim)
        self.logvar = nn.Linear(self.dense_dims, z_dim)

        '''
        Decoder for the inference maps
        '''
        self.back_to_conv = nn.Sequential(nn.Linear(z_dim, self.dense_dims),
                                          nn.GELU())
        self.mu_fc2 = nn.Sequential(SBlock(sd * 16, sd * 8, upsample=True))
        self.mu_fc3 = nn.Sequential(SBlock(sd * 8, sd * 4, upsample=True))
        self.mu_fc4 = nn.Sequential(SBlock(sd * 4, sd * 4, upsample=True))
        self.mu_fc5 = nn.Sequential(SBlock(sd * 4, sd * 2, upsample=True))
        self.mu_fc6 = nn.Sequential(SBlock(sd * 2, sd, upsample=True))
        self.mu_fc7 = nn.Sequential(nn.Conv3d(sd, int(sd / 2), kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv3d(int(sd / 2), out_chans, kernel_size=1, stride=1, padding=0)
                                    )

        '''
        Decoder for the lesion reconstructions
        '''
        self.rback_to_conv = nn.Sequential(nn.Linear(z_dim, self.dense_dims),
                                           nn.GELU())
        self.rmu_fc2 = nn.Sequential(SBlock(sd * 16, sd * 8, upsample=True))
        self.rmu_fc3 = nn.Sequential(SBlock(sd * 8, sd * 4, upsample=True))
        self.rmu_fc4 = nn.Sequential(SBlock(sd * 4, sd * 4, upsample=True))
        self.rmu_fc5 = nn.Sequential(SBlock(sd * 4, sd * 2, upsample=True))
        self.rmu_fc6 = nn.Sequential(SBlock(sd * 2, sd, upsample=True))
        self.rmu_fc7 = nn.Sequential(nn.Conv3d(sd, int(sd / 2), kernel_size=3, stride=1, padding=1),
                                     nn.GELU(),
                                     nn.Conv3d(int(sd / 2), 1, kernel_size=1, stride=1, padding=0)
                                     )

    def sampling(self, mu, log_var):
        '''
        Sample your latent from z ~ N(mean, scale)
        :param mu:
        :param log_var:
        :return:
        '''
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encoder(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        x = x.view(-1, self.dense_dims)
        return self.mu(x), self.logvar(x)

    def decoder(self, x):
        x = self.back_to_conv(x)
        x = x.view(x.size(0), -1, self.spatial_dims, self.spatial_dims, self.spatial_dims)
        x = self.mu_fc2(x)
        x = self.mu_fc3(x)
        x = self.mu_fc4(x)
        x = self.mu_fc5(x)
        x = self.mu_fc6(x)
        x = self.mu_fc7(x)
        return x

    def rdecoder(self, x):
        x = self.rback_to_conv(x)
        x = x.view(x.size(0), -1, self.spatial_dims, self.spatial_dims, self.spatial_dims)
        x = self.rmu_fc2(x)
        x = self.rmu_fc3(x)
        x = self.rmu_fc4(x)
        x = self.rmu_fc5(x)
        x = self.rmu_fc6(x)
        x = self.rmu_fc7(x)
        return x

    def forward(self, x, y):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)

        kl = torch.sum(0.5 * (-log_var + torch.exp(log_var) + mu ** 2 - 1), dim=1)

        return self.decoder(z), self.rdecoder(z), kl


class ModelWrapper(nn.Module):

    def __init__(self, input_size, z_dim=128, start_dims=16, continuous=False):
        '''
        A model wrapper around the VAE
        :param input_size:
        :param z_dim:
        :param start_dims:
        :param continuous:
        '''
        super().__init__()
        self.z_dim = z_dim
        self.start_dims = start_dims

        # 5 input channels - X, the coordinates, and Y
        # 2 output channels - The mean and the variance of the inference maps
        self.mask_model = VAE(input_size, sd=16, z_dim=20, out_chans=2, in_chans=5)

        self.continuous = continuous
        print(f'CONTINUOUS MODEL: {self.continuous}')

    def forward(self, x, y, val=False, provided_mask=None, provided_scale=None, t=0.5, calibrate=False):
        '''
        If doing validation you will want to use the generated inference map to gauge the accuracy of the
        predictions
        :param x:
        :param y:
        :param val:
        :param provided_mask:
        :param provided_scale:
        :param t:
        :param calibrate:
        :return:
        '''

        b, c, h, w, d = x.shape

        # Add coordinates to the lesion
        coord_x = add_coords(x)
        # Add the label as a volume
        my = y.view(-1, 1, 1, 1, 1).repeat(1, 1, h, w, d)
        coord_x = torch.cat([coord_x, my], dim=1)

        if val:
            # If doing validation use the masks calculated from the training data
            # Do a forward pass still so we can evaluate reconstruction quality and KL
            masks, recons, kl_m = self.mask_model(coord_x, y)
            preds_mean = provided_mask
            preds_scale = provided_scale
        else:
            masks, recons, kl_m = self.mask_model(coord_x, y)
            preds_mean = masks[:, 0].view(-1, 1, h, w, d)
            preds_scale = masks[:, 1].view(-1, 1, h, w, d)

        if calibrate:
            # If calibrating predictions, we want to find a thresholding quantile that achieves the best accuracy!
            flat_preds_a = preds_mean.view(x.size(0), -1)
            qt = torch.quantile(flat_preds_a, t, dim=1).view(-1, 1, 1, 1, 1)
            preds_mean = (preds_mean > qt) * preds_mean

        # The three outputs of our network -> Reconstructed lesion, Mean inference map and STD variance map
        recons = torch.sigmoid(recons)
        logits = torch.mean(x * preds_mean, dim=(-4, -3, -2, -1)).view(-1, 1)
        # Standard deviation is currently between 0 and 1, but it can be larger or smaller
        scale = torch.mean(x * preds_scale, dim=(-4, -3, -2, -1)).view(-1, 1).exp()

        '''
        Calculate log P(Y|X,M), i.e. the log-likelihood of our inference objective
        '''
        if self.continuous:
            mask_ll = - D.Normal(logits, scale + 1e-5).log_prob(y).mean()
        else:
            # Don't use STD on binary case because Bernoulli has no variance -> Beta distributions work well
            probabilities = torch.sigmoid(logits)
            mask_ll = bce_fn(probabilities, y).mean()

        '''
        Calculate log P(X|M), i.e. the log likelihood of our lesions 
        '''
        recon_ll = torch.sum(bce_fn(recons, x), dim=(-3, -2, -1)).mean()

        preds = torch.mean(preds_mean, dim=0).view(1, 1, h, w, d)
        mask_scale = torch.mean(preds_scale, dim=0).view(1, 1, h, w, d)

        # Calculate the accuracy of the predictions. If it is continuous, this is just MSE
        if self.continuous:
            acc = mask_ll
        else:
            quant_preds = (probabilities > 0.5).to(torch.float32)
            acc = torch.mean(torch.eq(quant_preds, y).float())

        '''
        The final loss is log P(Y| X, M) + log P(X|M) + D_KL[Q(M|X,Y) || P(M)]
        '''
        loss = mask_ll + recon_ll + kl_m.mean()

        ret_dict = dict(mean_mask=preds,
                        mask_scale=mask_scale,
                        mask_ll=mask_ll.mean(),
                        kl=kl_m.mean(),
                        loss=loss, acc=acc,
                        recon_ll=recon_ll.mean()
                        )

        return ret_dict

    def sample_masks(self, num_samples=400):
        '''
        Use this to sample the mean and STD masks from the latent space
        :param x:
        :param num_samples:
        :return:
        '''
        z = torch.randn(num_samples, self.z_dim).type(Tensor)
        preds = self.mask_model.decoder(z)
        mean_mask = torch.mean(preds[:, 0], dim=(0, 1))
        scale_mask = torch.mean(preds[:, 1], dim=(0, 1))
        return mean_mask, scale_mask
