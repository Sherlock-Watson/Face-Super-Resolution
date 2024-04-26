import os

from ffhqsub_dataset import FFHQsubDataset
import yaml
from collections import OrderedDict
from os import path as osp
import math
import cv2
import random
import torchvision.transforms as T
from PIL import Image
from matplotlib import cm
import numpy as np
from degradations import random_mixed_kernels


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for several datasets, e.g., test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def apply_convolution(img: np.array, kernel: np.array):
    # Get the height, width, and number of channels of the image
    height, width, c = img.shape[0], img.shape[1], img.shape[2]

    # Get the height, width, and number of channels of the kernel
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]

    # Create a new image of original img size minus the border
    # where the convolution can't be applied
    new_img = np.zeros((height - kernel_height + 1, width - kernel_width + 1, 3))

    # Loop through each pixel in the image
    # But skip the outer edges of the image
    for i in range(kernel_height // 2, height - kernel_height):
        for j in range(kernel_width // 2, width - kernel_width):
            # Extract a window of pixels around the current pixel
            window = img[i - kernel_height // 2: i + kernel_height // 2 + 1,
                     j - kernel_width // 2: j + kernel_width // 2 + 1]

            # Apply the convolution to the window and set the result as the value of the current pixel in the new image
            new_img[i, j, 0] = int((window[:, :, 0] * kernel).sum())
            new_img[i, j, 1] = int((window[:, :, 1] * kernel).sum())
            new_img[i, j, 2] = int((window[:, :, 2] * kernel).sum())

    # Clip values to the range 0-255
    new_img = np.clip(new_img, 0, 255)
    return new_img.astype(np.uint8)

if __name__ == '__main__':
    # opt_folder = '/home/polybee/Downloads/AI6126-Advanced_Computer_Vision-master/project_2/src/BasicSR/options/train/CARN/train_CARN_x4.yml'
    # opt = parse(opt_folder, is_train=True)
    # transform = T.ToPILImage()
    #
    # dataset = FFHQsubDataset(opt)
    # for i in range(len(os.listdir("/home/polybee/Downloads/AI6126-Advanced_Computer_Vision-master/project_2/data/dataset/datasets-20240424T134923Z-001/datasets/train/GT"))):
    #     choice = random.choice(["kernel1", "kernel2", "sinc_kernel"])
    #     d = dataset.__getitem__(i)
    #     img = transform(d[choice])
    #     img.show()

    for file in os.listdir("E:/msai/ACV/project_2/data/dataset/datasets-20240424T134923Z-001/datasets/train/GT"):
        choice = random.choice(['iso', 'aniso', 'generalized_iso', 'plateau_iso', 'plateau_aniso'])
        img = cv2.imread(f"E:/msai/ACV/project_2/data/dataset/datasets-20240424T134923Z-001/datasets/train/GT/{file}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        kernel = random_mixed_kernels(
            choice,
            0.5,
            kernel_size=21,
            sigma_x_range=(0.6, 5),
            sigma_y_range=(0.6, 5),
            rotation_range=(-math.pi, math.pi),
            betag_range=(0.5, 8),
            betap_range=(0.5, 8),
            noise_range=None)
        # pad_size = (21 - 21) // 2
        # kernel2 = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        new_im = cv2.filter2D(img, -1, kernel)
        new_im = cv2.resize(new_im, (128, 128))
        # new_im = apply_convolution(img, kernel2)
        new_im = Image.fromarray(np.uint8(new_im)).convert('RGB')
        new_im.save(os.path.join("E:/msai/ACV/project_2/data/dataset/datasets-20240424T134923Z-001/datasets/train/LQ2", file))
        # break




