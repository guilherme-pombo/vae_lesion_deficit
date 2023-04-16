import numpy as np
import random

from torch.utils.data import Dataset, DataLoader
from monai.transforms import *


def resize(volume, target_size):
    resize_transform = Compose([Resize((target_size[0],
                                        target_size[1],
                                        target_size[2]))])
    if len(volume.shape) == 3:
        volume = np.expand_dims(volume, axis=0)

    resized_volume = resize_transform(volume)
    resized_volume = np.squeeze(resized_volume)

    return resized_volume


class DeficitDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]

        return img, np.expand_dims(self.labels[index], axis=0)


def create_train_val_cal_loaders(images, labels, batch_size, continuous=False, seed=42):
    '''
    Create the training, validation and calibration sets given input lesions and associated labels
    If the labels are continuous you need to pass the flag as TRUE otherwise they won't be normalised
    :param images:
    :param labels:
    :param continuous:
    :param seed:
    :return:
    '''

    # Currently chooses training, validation and calibration randomly
    # You will most likely need to proper sampling to ensure Positive/Negative ratios that are decent
    # This is data dependent though, so you have to choose
    np.random.seed(seed)
    random.seed(seed)

    # Shuffle data randomly -- REMEMBER! BE SMART WITH YOUR SAMPLING OF TRAIN, VAL, CAL THIS IS JUST AN EXAMPLE
    indices = [i for i in range(len(images))]
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    if continuous:
        # Put them in the 0-1 range -- OPTIONAL but recommended
        labels = labels - labels.min()
        labels = labels / labels.max()
        # Gotta z-score normalise the labels
        labels = (labels - labels.mean()) / labels.std()

    # 90/5/5 split - no test set, this is inference
    train_l = int(0.9 * len(images))
    val_l = int(0.05 * len(images))
    cal_l = int(0.05 * len(images))

    train_data = images[:train_l]
    train_labels = labels[:train_l]
    val_data = images[train_l:(train_l+val_l)]
    val_labels = labels[train_l:(train_l+val_l)]
    cal_data = images[(train_l + val_l):]
    cal_labels = labels[(train_l + val_l):]

    '''
    Num workers = 0 because this DataLoader you can fit all the images and labels into RAM
    Lesion-deficit datasets are usually small, so this should be possible
    '''
    dataset = DeficitDataset(data=train_data, labels=train_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_dataset = DeficitDataset(data=val_data, labels=val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False,
                            shuffle=True, num_workers=0, pin_memory=True)
    cal_dataset = DeficitDataset(data=cal_data, labels=cal_labels)
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, drop_last=False,
                            shuffle=True, num_workers=0, pin_memory=True)

    return train_loader, val_loader, cal_loader


def viz_functional_parcellation(mask, grid_acc, v_idx, v, template_brain, cmap='autumn',
                                vmax=None, vmin=None):
    if v_idx == 0:
        template_slice = template_brain[:, :, v]
        template_slice = np.pad(template_slice, ((6, 6), (0, 0)),
                                constant_values=0)
    elif v_idx == 1:
        template_slice = template_brain[v, :, :]
        template_slice = np.pad(template_slice, ((0, 0), (6, 6)),
                                constant_values=0)
    else:
        template_slice = template_brain[:, v, :]
        template_slice = np.pad(template_slice, ((6, 6), (6, 6)),
                                constant_values=0)

    template_slice = np.rot90(template_slice, k=1)
    grid_acc.imshow(template_slice, cmap='gray')

    if v_idx == 0:
        mask = mask[:, :, v]
        mask = np.pad(mask, ((6, 6), (0, 0)), constant_values=0)
        mask = np.rot90(mask, k=1)
    elif v_idx == 1:
        mask = mask[v, :, :]
        mask = np.pad(mask, ((0, 0), (6, 6)), constant_values=0)
        mask = np.rot90(mask, k=1)
    else:
        mask = mask[:, v, :]
        mask = np.pad(mask, ((6, 6), (6, 6)), constant_values=0)
        mask = np.rot90(mask, k=1)

    if cmap != 'jet':
        mask = (mask > 0).astype(np.uint8)
        mask = np.ma.masked_where(mask == 0, mask)

    if not vmax:
        grid_acc.imshow(mask, cmap=cmap, alpha=0.6)
    else:
        grid_acc.imshow(mask, cmap=cmap, alpha=0.6, vmin=vmin, vmax=vmax)
