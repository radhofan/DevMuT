import os
import shutil
import time
from collections import deque
from mindspore.nn.cell import Cell
import mindspore
import mindspore.dataset.vision as c_vision
import cv2
import multiprocessing
import mindspore.dataset as ds
from PIL import Image, ImageSequence
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F2
from mindspore.dataset.vision.utils import Inter
import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore.common.initializer import TruncatedNormal
from network.cv.unet.Unetconfig import config
from mindspore import context, ops
from mindspore.nn import CentralCrop


def preprocess_img_mask(img, mask, num_classes, img_size, augment=False, eval_resize=False):
    """
    Preprocess for multi-class dataset.
    Random crop and flip images and masks when augment is True.
    """
    if augment:
        img_size_w = int(np.random.randint(img_size[0], img_size[0] * 1.5, 1))
        img_size_h = int(np.random.randint(img_size[1], img_size[1] * 1.5, 1))
        img = cv2.resize(img, (img_size_w, img_size_h))
        mask = cv2.resize(mask, (img_size_w, img_size_h))
        dw = int(np.random.randint(0, img_size_w - img_size[0] + 1, 1))
        dh = int(np.random.randint(0, img_size_h - img_size[1] + 1, 1))
        img = img[dh:dh + img_size[1], dw:dw + img_size[0], :]
        mask = mask[dh:dh + img_size[1], dw:dw + img_size[0]]
        if np.random.random() > 0.5:
            flip_code = int(np.random.randint(-1, 2, 1))
            img = cv2.flip(img, flip_code)
            mask = cv2.flip(mask, flip_code)
    else:
        img = cv2.resize(img, img_size)
        if not eval_resize:
            mask = cv2.resize(mask, img_size)
    img = (img.astype(np.float32) - 127.5) / 127.5
    img = img.transpose(2, 0, 1)
    if num_classes == 2:
        mask = mask.astype(np.float32) / mask.max()
        mask = (mask > 0.5).astype(np.int64)
    else:
        mask = mask.astype(np.int64)
    mask = (np.arange(num_classes) == mask[..., None]).astype(int)
    mask = mask.transpose(2, 0, 1).astype(np.float32)
    return img, mask


class MultiClassDataset:
    """
    Read image and mask from original images, and split all data into train_dataset and val_dataset by `split`.
    Get image path and mask path from a tree of directories,
    images within one folder is an image, the image file named `"image.png"`, the mask file named `"mask.png"`.
    """

    def __init__(self, data_dir, repeat, is_train=False, split=0.8, shuffle=False):
        self.data_dir = data_dir
        self.is_train = is_train
        self.split = (split != 1.0)
        if self.split:
            self.img_ids = sorted(next(os.walk(self.data_dir))[1])
            self.train_ids = self.img_ids[:int(len(self.img_ids) * split)] * repeat
            self.val_ids = self.img_ids[int(len(self.img_ids) * split):]
        else:
            self.train_ids = sorted(next(os.walk(os.path.join(self.data_dir, "train")))[1])
            self.val_ids = sorted(next(os.walk(os.path.join(self.data_dir, "val")))[1])
        if shuffle:
            np.random.shuffle(self.train_ids)

    def _read_img_mask(self, img_id):
        if self.split:
            path = os.path.join(self.data_dir, img_id)
        elif self.is_train:
            path = os.path.join(self.data_dir, "train", img_id)
        else:
            path = os.path.join(self.data_dir, "val", img_id)
        img = cv2.imread(os.path.join(path, "image.png"))
        mask = cv2.imread(os.path.join(path, "mask.png"), cv2.IMREAD_GRAYSCALE)
        return img, mask

    def __getitem__(self, index):
        if self.is_train:
            return self._read_img_mask(self.train_ids[index])
        return self._read_img_mask(self.val_ids[index])

    @property
    def column_names(self):
        column_names = ['image', 'mask']
        return column_names

    def __len__(self):
        if self.is_train:
            return len(self.train_ids)
        return len(self.val_ids)


def create_multi_class_dataset(data_dir, img_size, repeat, batch_size, num_classes=2, is_train=False, augment=False,
                               eval_resize=False, split=0.8, rank=0, group_size=1, shuffle=True):
    """
    Get generator dataset for multi-class dataset.
    """
    cv2.setNumThreads(0)
    ds.config.set_enable_shared_mem(True)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(4, cores // group_size)
    mc_dataset = MultiClassDataset(data_dir, repeat, is_train, split, shuffle)
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=True,
                                  num_shards=group_size, shard_id=rank,
                                  num_parallel_workers=num_parallel_workers, python_multiprocessing=is_train)
    compose_map_func = (lambda image, mask: preprocess_img_mask(image, mask, num_classes, tuple(img_size),
                                                                augment and is_train, eval_resize))
    dataset = dataset.map(operations=compose_map_func, input_columns=mc_dataset.column_names,
                          output_columns=mc_dataset.column_names,
                          num_parallel_workers=num_parallel_workers)
    dataset = dataset.batch(batch_size, drop_remainder=is_train, num_parallel_workers=num_parallel_workers)
    return dataset


def get_axis(x):
    shape = F2.shape(x)
    length = F2.tuple_len(shape)
    perm = F2.make_range(0, length)
    return perm


def _load_multipage_tiff(path):
    """Load tiff images containing many images in the channel dimension"""
    return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])


def _get_val_train_indices(length, fold, ratio=0.8):
    assert 0 < ratio <= 1, "Train/total data ratio must be in range (0.0, 1.0]"
    np.random.seed(0)
    indices = np.arange(0, length, 1, dtype=np.int64)
    np.random.shuffle(indices)

    if fold is not None:
        indices = deque(indices)
        indices.rotate(fold * round((1.0 - ratio) * length))
        indices = np.array(indices)
        train_indices = indices[:round(ratio * len(indices))]
        val_indices = indices[round(ratio * len(indices)):]
    else:
        train_indices = indices
        val_indices = []
    return train_indices, val_indices


def train_data_augmentation(img, mask):
    h_flip = np.random.random()
    if h_flip > 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)
    v_flip = np.random.random()
    if v_flip > 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)

    left = int(np.random.uniform() * 0.3 * 572)
    right = int((1 - np.random.uniform() * 0.3) * 572)
    top = int(np.random.uniform() * 0.3 * 572)
    bottom = int((1 - np.random.uniform() * 0.3) * 572)

    img = img[top:bottom, left:right]
    mask = mask[top:bottom, left:right]

    # adjust brightness
    brightness = np.random.uniform(-0.2, 0.2)
    img = np.float32(img + brightness * np.ones(img.shape))
    img = np.clip(img, -1.0, 1.0)

    return img, mask


def data_post_process(img, mask):
    img = np.expand_dims(img, axis=0)
    mask = (mask > 0.5).astype(np.int64)
    mask = (np.arange(mask.max() + 1) == mask[..., None]).astype(int)
    mask = mask.transpose(2, 0, 1).astype(np.float32)
    return img, mask


def create_dataset(data_dir, repeat=400, train_batch_size=16, augment=False, cross_val_ind=1, run_distribute=False,
                   do_crop=None, img_size=None):
    images = _load_multipage_tiff(os.path.join(data_dir, 'train-volume.tif'))
    masks = _load_multipage_tiff(os.path.join(data_dir, 'train-labels.tif'))

    train_indices, val_indices = _get_val_train_indices(len(images), cross_val_ind)
    train_images = images[train_indices]
    train_masks = masks[train_indices]
    train_images = np.repeat(train_images, repeat, axis=0)
    train_masks = np.repeat(train_masks, repeat, axis=0)
    val_images = images[val_indices]
    val_masks = masks[val_indices]

    train_image_data = {"image": train_images}
    train_mask_data = {"mask": train_masks}
    valid_image_data = {"image": val_images}
    valid_mask_data = {"mask": val_masks}
    ds_train_images = ds.NumpySlicesDataset(data=train_image_data, sampler=None, shuffle=False)
    ds_train_masks = ds.NumpySlicesDataset(data=train_mask_data, sampler=None, shuffle=False)
    ds_valid_images = ds.NumpySlicesDataset(data=valid_image_data, sampler=None, shuffle=False)
    ds_valid_masks = ds.NumpySlicesDataset(data=valid_mask_data, sampler=None, shuffle=False)

    if do_crop != "None":
        resize_size = [int(img_size[x] * do_crop[x] / 572) for x in range(len(img_size))]
    else:
        resize_size = img_size
    c_resize_op = c_vision.Resize(size=(resize_size[0], resize_size[1]), interpolation=Inter.BILINEAR)
    c_pad = c_vision.Pad(padding=(img_size[0] - resize_size[0]) // 2)
    c_rescale_image = c_vision.Rescale(1.0 / 127.5, -1)
    c_rescale_mask = c_vision.Rescale(1.0 / 255.0, 0)

    c_trans_normalize_img = [c_rescale_image, c_resize_op, c_pad]
    c_trans_normalize_mask = [c_rescale_mask, c_resize_op, c_pad]
    c_center_crop = c_vision.CenterCrop(size=388)

    train_image_ds = ds_train_images.map(input_columns="image", operations=c_trans_normalize_img)
    train_mask_ds = ds_train_masks.map(input_columns="mask", operations=c_trans_normalize_mask)
    train_ds = ds.zip((train_image_ds, train_mask_ds))
    train_ds = train_ds.project(columns=["image", "mask"])
    if augment:
        augment_process = train_data_augmentation
        c_resize_op = c_vision.Resize(size=(img_size[0], img_size[1]), interpolation=Inter.BILINEAR)
        train_ds = train_ds.map(input_columns=["image", "mask"], operations=augment_process)
        train_ds = train_ds.map(input_columns="image", operations=c_resize_op)
        train_ds = train_ds.map(input_columns="mask", operations=c_resize_op)

    if do_crop != "None":
        train_ds = train_ds.map(input_columns="mask", operations=c_center_crop)
    post_process = data_post_process
    train_ds = train_ds.map(input_columns=["image", "mask"], operations=post_process)
    train_ds = train_ds.shuffle(repeat * 24)
    train_ds = train_ds.batch(batch_size=train_batch_size, drop_remainder=True)

    valid_image_ds = ds_valid_images.map(input_columns="image", operations=c_trans_normalize_img)
    valid_mask_ds = ds_valid_masks.map(input_columns="mask", operations=c_trans_normalize_mask)
    valid_ds = ds.zip((valid_image_ds, valid_mask_ds))
    valid_ds = valid_ds.project(columns=["image", "mask"])
    if do_crop != "None":
        valid_ds = valid_ds.map(input_columns="mask", operations=c_center_crop)
    post_process = data_post_process
    valid_ds = valid_ds.map(input_columns=["image", "mask"], operations=post_process)
    valid_ds = valid_ds.batch(batch_size=1, drop_remainder=True)

    return train_ds, valid_ds


def get_loss(x, weights=1.0):
    """
    Computes the weighted loss
    Args:
        weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
            inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
    """
    input_dtype = x.dtype
    x = F.Cast()(x, mstype.float32)
    weights = F.Cast()(weights, mstype.float32)
    x = F.Mul()(weights, x)
    if True and True:
        x = F.ReduceMean()(x, get_axis(x))
    # if True and not True:
    #     x = self.reduce_sum(x, self.get_axis(x))
    x = F.Cast()(x, input_dtype)
    return x


class DoubleConv(nn.Cell):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        init_value_0 = TruncatedNormal(0.06)
        init_value_1 = TruncatedNormal(0.06)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.SequentialCell(
            [nn.Conv2d(in_channels, mid_channels, kernel_size=3, has_bias=True,
                       weight_init="HeUniform", pad_mode="valid", bias_init="Uniform"),
             nn.ReLU(),
             nn.Conv2d(mid_channels, out_channels, kernel_size=3, has_bias=True,
                       pad_mode="valid", weight_init="HeUniform", bias_init="Uniform"),
             nn.ReLU()]
        )

    def construct(self, x):
        # print("double_conv", x.shape)
        return self.double_conv(x)


class Down(nn.Cell):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.SequentialCell(
            [nn.MaxPool2d(kernel_size=2, stride=2),
             DoubleConv(in_channels, out_channels)]
        )

    def construct(self, x):
        return self.maxpool_conv(x)


class Up1(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 56.0 / 64.0
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.print_fn = F.Print()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        # print("x.shape = ", x.shape)
        return self.conv(x)


class Up2(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 104.0 / 136.0
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up3(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 200 / 280
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.print_fn = F.Print()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up4(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 392 / 568
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class OutConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        init_value = TruncatedNormal(0.06)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=True, weight_init="HeUniform"
                              , bias_init="Uniform")

    def construct(self, x):
        x = self.conv(x)
        return x


class UNetMedical(nn.Cell):
    def __init__(self, n_channels, n_classes):
        super(UNetMedical, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up1(1024, 512)
        self.up2 = Up2(512, 256)
        self.up3 = Up3(256, 128)
        self.up4 = Up4(128, 64)
        self.outc = OutConv(64, n_classes)

        self.in_shapes = {
            'INPUT': [-1, 1, 572, 572],
            'inc.double_conv.0': [-1, 1, 572, 572],
            'inc.double_conv.1': [-1, 64, 570, 570],
            'inc.double_conv.2': [-1, 64, 570, 570],
            'inc.double_conv.3': [-1, 64, 568, 568],
            'down1.maxpool_conv.0': [-1, 64, 568, 568],
            'down1.maxpool_conv.1.double_conv.0': [-1, 64, 284, 284],
            'down1.maxpool_conv.1.double_conv.1': [-1, 128, 282, 282],
            'down1.maxpool_conv.1.double_conv.2': [-1, 128, 282, 282],
            'down1.maxpool_conv.1.double_conv.3': [-1, 128, 280, 280],
            'down2.maxpool_conv.0': [-1, 128, 280, 280],
            'down2.maxpool_conv.1.double_conv.0': [-1, 128, 140, 140],
            'down2.maxpool_conv.1.double_conv.1': [-1, 256, 138, 138],
            'down2.maxpool_conv.1.double_conv.2': [-1, 256, 138, 138],
            'down2.maxpool_conv.1.double_conv.3': [-1, 256, 136, 136],
            'down3.maxpool_conv.0': [-1, 256, 136, 136],
            'down3.maxpool_conv.1.double_conv.0': [-1, 256, 68, 68],
            'down3.maxpool_conv.1.double_conv.1': [-1, 512, 66, 66],
            'down3.maxpool_conv.1.double_conv.2': [-1, 512, 66, 66],
            'down3.maxpool_conv.1.double_conv.3': [-1, 512, 64, 64],
            'down4.maxpool_conv.0': [-1, 512, 64, 64],
            'down4.maxpool_conv.1.double_conv.0': [-1, 512, 32, 32],
            'down4.maxpool_conv.1.double_conv.1': [-1, 1024, 30, 30],
            'down4.maxpool_conv.1.double_conv.2': [-1, 1024, 30, 30],
            'down4.maxpool_conv.1.double_conv.3': [-1, 1024, 28, 28],
            'up1.up': [-1, 1024, 28, 28],
            'up1.relu': [-1, 512, 56, 56],
            'up1.center_crop': [-1, 512, 64, 64],
            'up1.conv.double_conv.0': [-1, 1024, 56, 56],
            'up1.conv.double_conv.1': [-1, 512, 54, 54],
            'up1.conv.double_conv.2': [-1, 512, 54, 54],
            'up1.conv.double_conv.3': [-1, 512, 52, 52],
            'up2.up': [-1, 512, 52, 52],
            'up2.relu': [-1, 256, 104, 104],
            'up2.center_crop': [-1, 256, 136, 136],
            'up2.conv.double_conv.0': [-1, 512, 104, 104],
            'up2.conv.double_conv.1': [-1, 256, 102, 102],
            'up2.conv.double_conv.2': [-1, 256, 102, 102],
            'up2.conv.double_conv.3': [-1, 256, 100, 100],
            'up3.up': [-1, 256, 100, 100],
            'up3.relu': [-1, 128, 200, 200],
            'up3.center_crop': [-1, 128, 280, 280],
            'up3.conv.double_conv.0': [-1, 256, 200, 200],
            'up3.conv.double_conv.1': [-1, 128, 198, 198],
            'up3.conv.double_conv.2': [-1, 128, 198, 198],
            'up3.conv.double_conv.3': [-1, 128, 196, 196],
            'up4.up': [-1, 128, 196, 196],
            'up4.relu': [-1, 64, 392, 392],
            'up4.center_crop': [-1, 64, 568, 568],
            'up4.conv.double_conv.0': [-1, 128, 392, 392],
            'up4.conv.double_conv.1': [-1, 64, 390, 390],
            'up4.conv.double_conv.2': [-1, 64, 390, 390],
            'up4.conv.double_conv.3': [-1, 64, 388, 388],
            'outc.conv': [-1, 64, 388, 388],
            'OUTPUT': [-1, 64, 388, 388]
        }

        self.out_shapes = {
            'INPUT': [-1, 1, 572, 572],
            'inc.double_conv.0': [-1, 64, 570, 570],
            'inc.double_conv.1': [-1, 64, 570, 570],
            'inc.double_conv.2': [-1, 64, 568, 568],
            'inc.double_conv.3': [-1, 64, 568, 568],
            'down1.maxpool_conv.0': [-1, 64, 284, 284],
            'down1.maxpool_conv.1.double_conv.0': [-1, 128, 282, 282],
            'down1.maxpool_conv.1.double_conv.1': [-1, 128, 282, 282],
            'down1.maxpool_conv.1.double_conv.2': [-1, 128, 280, 280],
            'down1.maxpool_conv.1.double_conv.3': [-1, 128, 280, 280],
            'down2.maxpool_conv.0': [-1, 128, 140, 140],
            'down2.maxpool_conv.1.double_conv.0': [-1, 256, 138, 138],
            'down2.maxpool_conv.1.double_conv.1': [-1, 256, 138, 138],
            'down2.maxpool_conv.1.double_conv.2': [-1, 256, 136, 136],
            'down2.maxpool_conv.1.double_conv.3': [-1, 256, 136, 136],
            'down3.maxpool_conv.0': [-1, 256, 68, 68],
            'down3.maxpool_conv.1.double_conv.0': [-1, 512, 66, 66],
            'down3.maxpool_conv.1.double_conv.1': [-1, 512, 66, 66],
            'down3.maxpool_conv.1.double_conv.2': [-1, 512, 64, 64],
            'down3.maxpool_conv.1.double_conv.3': [-1, 512, 64, 64],
            'down4.maxpool_conv.0': [-1, 512, 32, 32],
            'down4.maxpool_conv.1.double_conv.0': [-1, 1024, 30, 30],
            'down4.maxpool_conv.1.double_conv.1': [-1, 1024, 30, 30],
            'down4.maxpool_conv.1.double_conv.2': [-1, 1024, 28, 28],
            'down4.maxpool_conv.1.double_conv.3': [-1, 1024, 28, 28],
            'up1.up': [-1, 512, 56, 56],
            'up1.relu': [-1, 512, 56, 56],
            'up1.center_crop': [-1, 512, 56, 56],
            'up1.conv.double_conv.0': [-1, 512, 54, 54],
            'up1.conv.double_conv.1': [-1, 512, 54, 54],
            'up1.conv.double_conv.2': [-1, 512, 52, 52],
            'up1.conv.double_conv.3': [-1, 512, 52, 52],
            'up2.up': [-1, 256, 104, 104],
            'up2.relu': [-1, 256, 104, 104],
            'up2.center_crop': [-1, 256, 104, 104],
            'up2.conv.double_conv.0': [-1, 256, 102, 102],
            'up2.conv.double_conv.1': [-1, 256, 102, 102],
            'up2.conv.double_conv.2': [-1, 256, 100, 100],
            'up2.conv.double_conv.3': [-1, 256, 100, 100],
            'up3.up': [-1, 128, 200, 200],
            'up3.relu': [-1, 128, 200, 200],
            'up3.center_crop': [-1, 128, 200, 200],
            'up3.conv.double_conv.0': [-1, 128, 198, 198],
            'up3.conv.double_conv.1': [-1, 128, 198, 198],
            'up3.conv.double_conv.2': [-1, 128, 196, 196],
            'up3.conv.double_conv.3': [-1, 128, 196, 196],
            'up4.up': [-1, 64, 392, 392],
            'up4.relu': [-1, 64, 392, 392],
            'up4.center_crop': [-1, 64, 392, 392],
            'up4.conv.double_conv.0': [-1, 64, 390, 390],
            'up4.conv.double_conv.1': [-1, 64, 390, 390],
            'up4.conv.double_conv.2': [-1, 64, 388, 388],
            'up4.conv.double_conv.3': [-1, 64, 388, 388],
            'outc.conv': [-1, 2, 388, 388],
            'OUTPUT': [-1, 2, 388, 388]
        }

        self.orders = {
            "inc.double_conv.0": ["INPUT", "inc.double_conv.1"],
            "inc.double_conv.1": ["inc.double_conv.0", "inc.double_conv.2"],
            "inc.double_conv.2": ["inc.double_conv.1", "inc.double_conv.3"],
            "inc.double_conv.3": ["inc.double_conv.2", ["down1.maxpool_conv.0", "up4.center_crop"]],

            "down1.maxpool_conv.0": ["inc.double_conv.3", "down1.maxpool_conv.1.double_conv.0"],
            "down1.maxpool_conv.1.double_conv.0": ["down1.maxpool_conv.0", "down1.maxpool_conv.1.double_conv.1"],
            "down1.maxpool_conv.1.double_conv.1": ["down1.maxpool_conv.1.double_conv.0",
                                                   "down1.maxpool_conv.1.double_conv.2"],
            "down1.maxpool_conv.1.double_conv.2": ["down1.maxpool_conv.1.double_conv.1",
                                                   "down1.maxpool_conv.1.double_conv.3"],
            "down1.maxpool_conv.1.double_conv.3": ["down1.maxpool_conv.1.double_conv.2",
                                                   ["down2.maxpool_conv.0", "up3.center_crop"]],

            "down2.maxpool_conv.0": ["down1.maxpool_conv.1.double_conv.3", "down2.maxpool_conv.1.double_conv.0"],
            "down2.maxpool_conv.1.double_conv.0": ["down2.maxpool_conv.0", "down2.maxpool_conv.1.double_conv.1"],
            "down2.maxpool_conv.1.double_conv.1": ["down2.maxpool_conv.1.double_conv.0",
                                                   "down2.maxpool_conv.1.double_conv.2"],
            "down2.maxpool_conv.1.double_conv.2": ["down2.maxpool_conv.1.double_conv.1",
                                                   "down2.maxpool_conv.1.double_conv.3"],
            "down2.maxpool_conv.1.double_conv.3": ["down2.maxpool_conv.1.double_conv.2",
                                                   ["down3.maxpool_conv.0", "up2.center_crop"]],

            "down3.maxpool_conv.0": ["down2.maxpool_conv.1.double_conv.3", "down3.maxpool_conv.1.double_conv.0"],
            "down3.maxpool_conv.1.double_conv.0": ["down3.maxpool_conv.0", "down3.maxpool_conv.1.double_conv.1"],
            "down3.maxpool_conv.1.double_conv.1": ["down3.maxpool_conv.1.double_conv.0",
                                                   "down3.maxpool_conv.1.double_conv.2"],
            "down3.maxpool_conv.1.double_conv.2": ["down3.maxpool_conv.1.double_conv.1",
                                                   "down3.maxpool_conv.1.double_conv.3"],
            "down3.maxpool_conv.1.double_conv.3": ["down3.maxpool_conv.1.double_conv.2",
                                                   ["down4.maxpool_conv.0", "up1.center_crop"]],

            "down4.maxpool_conv.0": ["down3.maxpool_conv.1.double_conv.3", "down4.maxpool_conv.1.double_conv.0"],
            "down4.maxpool_conv.1.double_conv.0": ["down4.maxpool_conv.0", "down4.maxpool_conv.1.double_conv.1"],
            "down4.maxpool_conv.1.double_conv.1": ["down4.maxpool_conv.1.double_conv.0",
                                                   "down4.maxpool_conv.1.double_conv.2"],
            "down4.maxpool_conv.1.double_conv.2": ["down4.maxpool_conv.1.double_conv.1",
                                                   "down4.maxpool_conv.1.double_conv.3"],
            "down4.maxpool_conv.1.double_conv.3": ["down4.maxpool_conv.1.double_conv.2", "up1.up"],

            "up1.up": ["down4.maxpool_conv.1.double_conv.3", "up1.relu"],
            "up1.relu": ["up1.up", "up1.conv.double_conv.0"],
            "up1.center_crop": ["down3.maxpool_conv.1.double_conv.3", "up1.conv.double_conv.0"],
            "up1.conv.double_conv.0": [["up1.relu", "up1.center_crop"], "up1.conv.double_conv.1"],
            "up1.conv.double_conv.1": ["up1.conv.double_conv.0", "up1.conv.double_conv.2"],
            "up1.conv.double_conv.2": ["up1.conv.double_conv.1", "up1.conv.double_conv.3"],
            "up1.conv.double_conv.3": ["up1.conv.double_conv.2", "up2.up"],

            "up2.up": ["up1.conv.double_conv.3", "up2.relu"],
            "up2.relu": ["up2.up", "up2.conv.double_conv.0"],
            "up2.center_crop": ["down2.maxpool_conv.1.double_conv.3", "up2.conv.double_conv.0"],
            "up2.conv.double_conv.0": [["up2.relu", "up2.center_crop"], "up2.conv.double_conv.1"],
            "up2.conv.double_conv.1": ["up2.conv.double_conv.0", "up2.conv.double_conv.2"],
            "up2.conv.double_conv.2": ["up2.conv.double_conv.1", "up2.conv.double_conv.3"],
            "up2.conv.double_conv.3": ["up2.conv.double_conv.2", "up3.up"],

            "up3.up": ["up2.conv.double_conv.3", "up3.relu"],
            "up3.relu": ["up3.up", "up3.conv.double_conv.0"],
            "up3.center_crop": ["down1.maxpool_conv.1.double_conv.3", "up3.conv.double_conv.0"],
            "up3.conv.double_conv.0": [["up3.relu", "up3.center_crop"], "up3.conv.double_conv.1"],
            "up3.conv.double_conv.1": ["up3.conv.double_conv.0", "up3.conv.double_conv.2"],
            "up3.conv.double_conv.2": ["up3.conv.double_conv.1", "up3.conv.double_conv.3"],
            "up3.conv.double_conv.3": ["up3.conv.double_conv.2", "up4.up"],

            "up4.up": ["up3.conv.double_conv.3", "up4.relu"],
            "up4.relu": ["up4.up", "up4.conv.double_conv.0"],
            "up4.center_crop": ["inc.double_conv.3", "up4.conv.double_conv.0"],
            "up4.conv.double_conv.0": [["up4.relu", "up4.center_crop"], "up4.conv.double_conv.1"],
            "up4.conv.double_conv.1": ["up4.conv.double_conv.0", "up4.conv.double_conv.2"],
            "up4.conv.double_conv.2": ["up4.conv.double_conv.1", "up4.conv.double_conv.3"],
            "up4.conv.double_conv.3": ["up4.conv.double_conv.2", "outc.conv"],

            "outc.conv": ["up4.conv.double_conv.3", "OUTPUT"],
        }

        self.layer_names = {
            "inc": self.inc,
            "inc.double_conv": self.inc.double_conv,
            "inc.double_conv.0": self.inc.double_conv[0],
            "inc.double_conv.1": self.inc.double_conv[1],
            "inc.double_conv.2": self.inc.double_conv[2],
            "inc.double_conv.3": self.inc.double_conv[3],
            "down1": self.down1,
            "down1.maxpool_conv": self.down1.maxpool_conv,
            "down1.maxpool_conv.0": self.down1.maxpool_conv[0],
            "down1.maxpool_conv.1": self.down1.maxpool_conv[1],
            "down1.maxpool_conv.1.double_conv": self.down1.maxpool_conv[1].double_conv,
            "down1.maxpool_conv.1.double_conv.0": self.down1.maxpool_conv[1].double_conv[0],
            "down1.maxpool_conv.1.double_conv.1": self.down1.maxpool_conv[1].double_conv[1],
            "down1.maxpool_conv.1.double_conv.2": self.down1.maxpool_conv[1].double_conv[2],
            "down1.maxpool_conv.1.double_conv.3": self.down1.maxpool_conv[1].double_conv[3],
            "down2": self.down2,
            "down2.maxpool_conv": self.down2.maxpool_conv,
            "down2.maxpool_conv.0": self.down2.maxpool_conv[0],
            "down2.maxpool_conv.1": self.down2.maxpool_conv[1],
            "down2.maxpool_conv.1.double_conv": self.down2.maxpool_conv[1].double_conv,
            "down2.maxpool_conv.1.double_conv.0": self.down2.maxpool_conv[1].double_conv[0],
            "down2.maxpool_conv.1.double_conv.1": self.down2.maxpool_conv[1].double_conv[1],
            "down2.maxpool_conv.1.double_conv.2": self.down2.maxpool_conv[1].double_conv[2],
            "down2.maxpool_conv.1.double_conv.3": self.down2.maxpool_conv[1].double_conv[3],
            "down3": self.down3,
            "down3.maxpool_conv": self.down3.maxpool_conv,
            "down3.maxpool_conv.0": self.down3.maxpool_conv[0],
            "down3.maxpool_conv.1": self.down3.maxpool_conv[1],
            "down3.maxpool_conv.1.double_conv": self.down3.maxpool_conv[1].double_conv,
            "down3.maxpool_conv.1.double_conv.0": self.down3.maxpool_conv[1].double_conv[0],
            "down3.maxpool_conv.1.double_conv.1": self.down3.maxpool_conv[1].double_conv[1],
            "down3.maxpool_conv.1.double_conv.2": self.down3.maxpool_conv[1].double_conv[2],
            "down3.maxpool_conv.1.double_conv.3": self.down3.maxpool_conv[1].double_conv[3],
            "down4": self.down4,
            "down4.maxpool_conv": self.down4.maxpool_conv,
            "down4.maxpool_conv.0": self.down4.maxpool_conv[0],
            "down4.maxpool_conv.1": self.down4.maxpool_conv[1],
            "down4.maxpool_conv.1.double_conv": self.down4.maxpool_conv[1].double_conv,
            "down4.maxpool_conv.1.double_conv.0": self.down4.maxpool_conv[1].double_conv[0],
            "down4.maxpool_conv.1.double_conv.1": self.down4.maxpool_conv[1].double_conv[1],
            "down4.maxpool_conv.1.double_conv.2": self.down4.maxpool_conv[1].double_conv[2],
            "down4.maxpool_conv.1.double_conv.3": self.down4.maxpool_conv[1].double_conv[3],
            "up1": self.up1,
            "up1.center_crop": self.up1.center_crop,
            "up1.conv": self.up1.conv,
            "up1.conv.double_conv": self.up1.conv.double_conv,
            "up1.conv.double_conv.0": self.up1.conv.double_conv[0],
            "up1.conv.double_conv.1": self.up1.conv.double_conv[1],
            "up1.conv.double_conv.2": self.up1.conv.double_conv[2],
            "up1.conv.double_conv.3": self.up1.conv.double_conv[3],
            "up1.up": self.up1.up,
            "up1.relu": self.up1.relu,
            "up2": self.up2,
            "up2.center_crop": self.up2.center_crop,
            "up2.conv": self.up2.conv,
            "up2.conv.double_conv": self.up2.conv.double_conv,
            "up2.conv.double_conv.0": self.up2.conv.double_conv[0],
            "up2.conv.double_conv.1": self.up2.conv.double_conv[1],
            "up2.conv.double_conv.2": self.up2.conv.double_conv[2],
            "up2.conv.double_conv.3": self.up2.conv.double_conv[3],
            "up2.up": self.up2.up,
            "up2.relu": self.up2.relu,
            "up3": self.up3,
            "up3.center_crop": self.up3.center_crop,
            "up3.conv": self.up3.conv,
            "up3.conv.double_conv": self.up3.conv.double_conv,
            "up3.conv.double_conv.0": self.up3.conv.double_conv[0],
            "up3.conv.double_conv.1": self.up3.conv.double_conv[1],
            "up3.conv.double_conv.2": self.up3.conv.double_conv[2],
            "up3.conv.double_conv.3": self.up3.conv.double_conv[3],
            "up3.up": self.up3.up,
            "up3.relu": self.up3.relu,
            "up4": self.up4,
            "up4.center_crop": self.up4.center_crop,
            "up4.conv": self.up4.conv,
            "up4.conv.double_conv": self.up4.conv.double_conv,
            "up4.conv.double_conv.0": self.up4.conv.double_conv[0],
            "up4.conv.double_conv.1": self.up4.conv.double_conv[1],
            "up4.conv.double_conv.2": self.up4.conv.double_conv[2],
            "up4.conv.double_conv.3": self.up4.conv.double_conv[3],
            "up4.up": self.up4.up,
            "up4.relu": self.up4.relu,
            "outc": self.outc,
            "outc.conv": self.outc.conv,
        }

        self.origin_layer_names = {
            "inc": self.inc,
            "inc.double_conv": self.inc.double_conv,
            "inc.double_conv.0": self.inc.double_conv[0],
            "inc.double_conv.1": self.inc.double_conv[1],
            "inc.double_conv.2": self.inc.double_conv[2],
            "inc.double_conv.3": self.inc.double_conv[3],
            "down1": self.down1,
            "down1.maxpool_conv": self.down1.maxpool_conv,
            "down1.maxpool_conv.0": self.down1.maxpool_conv[0],
            "down1.maxpool_conv.1": self.down1.maxpool_conv[1],
            "down1.maxpool_conv.1.double_conv": self.down1.maxpool_conv[1].double_conv,
            "down1.maxpool_conv.1.double_conv.0": self.down1.maxpool_conv[1].double_conv[0],
            "down1.maxpool_conv.1.double_conv.1": self.down1.maxpool_conv[1].double_conv[1],
            "down1.maxpool_conv.1.double_conv.2": self.down1.maxpool_conv[1].double_conv[2],
            "down1.maxpool_conv.1.double_conv.3": self.down1.maxpool_conv[1].double_conv[3],
            "down2": self.down2,
            "down2.maxpool_conv": self.down2.maxpool_conv,
            "down2.maxpool_conv.0": self.down2.maxpool_conv[0],
            "down2.maxpool_conv.1": self.down2.maxpool_conv[1],
            "down2.maxpool_conv.1.double_conv": self.down2.maxpool_conv[1].double_conv,
            "down2.maxpool_conv.1.double_conv.0": self.down2.maxpool_conv[1].double_conv[0],
            "down2.maxpool_conv.1.double_conv.1": self.down2.maxpool_conv[1].double_conv[1],
            "down2.maxpool_conv.1.double_conv.2": self.down2.maxpool_conv[1].double_conv[2],
            "down2.maxpool_conv.1.double_conv.3": self.down2.maxpool_conv[1].double_conv[3],
            "down3": self.down3,
            "down3.maxpool_conv": self.down3.maxpool_conv,
            "down3.maxpool_conv.0": self.down3.maxpool_conv[0],
            "down3.maxpool_conv.1": self.down3.maxpool_conv[1],
            "down3.maxpool_conv.1.double_conv": self.down3.maxpool_conv[1].double_conv,
            "down3.maxpool_conv.1.double_conv.0": self.down3.maxpool_conv[1].double_conv[0],
            "down3.maxpool_conv.1.double_conv.1": self.down3.maxpool_conv[1].double_conv[1],
            "down3.maxpool_conv.1.double_conv.2": self.down3.maxpool_conv[1].double_conv[2],
            "down3.maxpool_conv.1.double_conv.3": self.down3.maxpool_conv[1].double_conv[3],
            "down4": self.down4,
            "down4.maxpool_conv": self.down4.maxpool_conv,
            "down4.maxpool_conv.0": self.down4.maxpool_conv[0],
            "down4.maxpool_conv.1": self.down4.maxpool_conv[1],
            "down4.maxpool_conv.1.double_conv": self.down4.maxpool_conv[1].double_conv,
            "down4.maxpool_conv.1.double_conv.0": self.down4.maxpool_conv[1].double_conv[0],
            "down4.maxpool_conv.1.double_conv.1": self.down4.maxpool_conv[1].double_conv[1],
            "down4.maxpool_conv.1.double_conv.2": self.down4.maxpool_conv[1].double_conv[2],
            "down4.maxpool_conv.1.double_conv.3": self.down4.maxpool_conv[1].double_conv[3],
            "up1": self.up1,
            "up1.center_crop": self.up1.center_crop,
            "up1.conv": self.up1.conv,
            "up1.conv.double_conv": self.up1.conv.double_conv,
            "up1.conv.double_conv.0": self.up1.conv.double_conv[0],
            "up1.conv.double_conv.1": self.up1.conv.double_conv[1],
            "up1.conv.double_conv.2": self.up1.conv.double_conv[2],
            "up1.conv.double_conv.3": self.up1.conv.double_conv[3],
            "up1.up": self.up1.up,
            "up1.relu": self.up1.relu,
            "up2": self.up2,
            "up2.center_crop": self.up2.center_crop,
            "up2.conv": self.up2.conv,
            "up2.conv.double_conv": self.up2.conv.double_conv,
            "up2.conv.double_conv.0": self.up2.conv.double_conv[0],
            "up2.conv.double_conv.1": self.up2.conv.double_conv[1],
            "up2.conv.double_conv.2": self.up2.conv.double_conv[2],
            "up2.conv.double_conv.3": self.up2.conv.double_conv[3],
            "up2.up": self.up2.up,
            "up2.relu": self.up2.relu,
            "up3": self.up3,
            "up3.center_crop": self.up3.center_crop,
            "up3.conv": self.up3.conv,
            "up3.conv.double_conv": self.up3.conv.double_conv,
            "up3.conv.double_conv.0": self.up3.conv.double_conv[0],
            "up3.conv.double_conv.1": self.up3.conv.double_conv[1],
            "up3.conv.double_conv.2": self.up3.conv.double_conv[2],
            "up3.conv.double_conv.3": self.up3.conv.double_conv[3],
            "up3.up": self.up3.up,
            "up3.relu": self.up3.relu,
            "up4": self.up4,
            "up4.center_crop": self.up4.center_crop,
            "up4.conv": self.up4.conv,
            "up4.conv.double_conv": self.up4.conv.double_conv,
            "up4.conv.double_conv.0": self.up4.conv.double_conv[0],
            "up4.conv.double_conv.1": self.up4.conv.double_conv[1],
            "up4.conv.double_conv.2": self.up4.conv.double_conv[2],
            "up4.conv.double_conv.3": self.up4.conv.double_conv[3],
            "up4.up": self.up4.up,
            "up4.relu": self.up4.relu,
            "outc": self.outc,
            "outc.conv": self.outc.conv,
        }

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []



    def set_layers(self, layer_name, new_layer):
        if 'inc' == layer_name:
            self.inc = new_layer
            self.layer_names["inc"] = new_layer
            self.origin_layer_names["inc"] = new_layer
        elif 'inc.double_conv' == layer_name:
            self.inc.double_conv = new_layer
            self.layer_names["inc.double_conv"] = new_layer
            self.origin_layer_names["inc.double_conv"] = new_layer
        elif 'inc.double_conv.0' == layer_name:
            self.inc.double_conv[0] = new_layer
            self.layer_names["inc.double_conv.0"] = new_layer
            self.origin_layer_names["inc.double_conv.0"] = new_layer
        elif 'inc.double_conv.1' == layer_name:
            self.inc.double_conv[1] = new_layer
            self.layer_names["inc.double_conv.1"] = new_layer
            self.origin_layer_names["inc.double_conv.1"] = new_layer
        elif 'inc.double_conv.2' == layer_name:
            self.inc.double_conv[2] = new_layer
            self.layer_names["inc.double_conv.2"] = new_layer
            self.origin_layer_names["inc.double_conv.2"] = new_layer
        elif 'inc.double_conv.3' == layer_name:
            self.inc.double_conv[3] = new_layer
            self.layer_names["inc.double_conv.3"] = new_layer
            self.origin_layer_names["inc.double_conv.3"] = new_layer
        elif 'down1' == layer_name:
            self.down1 = new_layer
            self.layer_names["down1"] = new_layer
            self.origin_layer_names["down1"] = new_layer
        elif 'down1.maxpool_conv' == layer_name:
            self.down1.maxpool_conv = new_layer
            self.layer_names["down1.maxpool_conv"] = new_layer
            self.origin_layer_names["down1.maxpool_conv"] = new_layer
        elif 'down1.maxpool_conv.0' == layer_name:
            self.down1.maxpool_conv[0] = new_layer
            self.layer_names["down1.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.0"] = new_layer
        elif 'down1.maxpool_conv.1' == layer_name:
            self.down1.maxpool_conv[1] = new_layer
            self.layer_names["down1.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv' == layer_name:
            self.down1.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.0' == layer_name:
            self.down1.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.1' == layer_name:
            self.down1.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.2' == layer_name:
            self.down1.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.3' == layer_name:
            self.down1.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'down2' == layer_name:
            self.down2 = new_layer
            self.layer_names["down2"] = new_layer
            self.origin_layer_names["down2"] = new_layer
        elif 'down2.maxpool_conv' == layer_name:
            self.down2.maxpool_conv = new_layer
            self.layer_names["down2.maxpool_conv"] = new_layer
            self.origin_layer_names["down2.maxpool_conv"] = new_layer
        elif 'down2.maxpool_conv.0' == layer_name:
            self.down2.maxpool_conv[0] = new_layer
            self.layer_names["down2.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.0"] = new_layer
        elif 'down2.maxpool_conv.1' == layer_name:
            self.down2.maxpool_conv[1] = new_layer
            self.layer_names["down2.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv' == layer_name:
            self.down2.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.0' == layer_name:
            self.down2.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.1' == layer_name:
            self.down2.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.2' == layer_name:
            self.down2.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.3' == layer_name:
            self.down2.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'down3' == layer_name:
            self.down3 = new_layer
            self.layer_names["down3"] = new_layer
            self.origin_layer_names["down3"] = new_layer
        elif 'down3.maxpool_conv' == layer_name:
            self.down3.maxpool_conv = new_layer
            self.layer_names["down3.maxpool_conv"] = new_layer
            self.origin_layer_names["down3.maxpool_conv"] = new_layer
        elif 'down3.maxpool_conv.0' == layer_name:
            self.down3.maxpool_conv[0] = new_layer
            self.layer_names["down3.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.0"] = new_layer
        elif 'down3.maxpool_conv.1' == layer_name:
            self.down3.maxpool_conv[1] = new_layer
            self.layer_names["down3.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv' == layer_name:
            self.down3.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.0' == layer_name:
            self.down3.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.1' == layer_name:
            self.down3.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.2' == layer_name:
            self.down3.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.3' == layer_name:
            self.down3.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'down4' == layer_name:
            self.down4 = new_layer
            self.layer_names["down4"] = new_layer
            self.origin_layer_names["down4"] = new_layer
        elif 'down4.maxpool_conv' == layer_name:
            self.down4.maxpool_conv = new_layer
            self.layer_names["down4.maxpool_conv"] = new_layer
            self.origin_layer_names["down4.maxpool_conv"] = new_layer
        elif 'down4.maxpool_conv.0' == layer_name:
            self.down4.maxpool_conv[0] = new_layer
            self.layer_names["down4.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.0"] = new_layer
        elif 'down4.maxpool_conv.1' == layer_name:
            self.down4.maxpool_conv[1] = new_layer
            self.layer_names["down4.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv' == layer_name:
            self.down4.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.0' == layer_name:
            self.down4.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.1' == layer_name:
            self.down4.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.2' == layer_name:
            self.down4.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.3' == layer_name:
            self.down4.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'up1' == layer_name:
            self.up1 = new_layer
            self.layer_names["up1"] = new_layer
            self.origin_layer_names["up1"] = new_layer
        elif 'up1.center_crop' == layer_name:
            self.up1.center_crop = new_layer
            self.layer_names["up1.center_crop"] = new_layer
            self.origin_layer_names["up1.center_crop"] = new_layer
        elif 'up1.conv' == layer_name:
            self.up1.conv = new_layer
            self.layer_names["up1.conv"] = new_layer
            self.origin_layer_names["up1.conv"] = new_layer
        elif 'up1.conv.double_conv' == layer_name:
            self.up1.conv.double_conv = new_layer
            self.layer_names["up1.conv.double_conv"] = new_layer
            self.origin_layer_names["up1.conv.double_conv"] = new_layer
        elif 'up1.conv.double_conv.0' == layer_name:
            self.up1.conv.double_conv[0] = new_layer
            self.layer_names["up1.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.0"] = new_layer
        elif 'up1.conv.double_conv.1' == layer_name:
            self.up1.conv.double_conv[1] = new_layer
            self.layer_names["up1.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.1"] = new_layer
        elif 'up1.conv.double_conv.2' == layer_name:
            self.up1.conv.double_conv[2] = new_layer
            self.layer_names["up1.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.2"] = new_layer
        elif 'up1.conv.double_conv.3' == layer_name:
            self.up1.conv.double_conv[3] = new_layer
            self.layer_names["up1.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.3"] = new_layer
        elif 'up1.up' == layer_name:
            self.up1.up = new_layer
            self.layer_names["up1.up"] = new_layer
            self.origin_layer_names["up1.up"] = new_layer
        elif 'up1.relu' == layer_name:
            self.up1.relu = new_layer
            self.layer_names["up1.relu"] = new_layer
            self.origin_layer_names["up1.relu"] = new_layer
        elif 'up2' == layer_name:
            self.up2 = new_layer
            self.layer_names["up2"] = new_layer
            self.origin_layer_names["up2"] = new_layer
        elif 'up2.center_crop' == layer_name:
            self.up2.center_crop = new_layer
            self.layer_names["up2.center_crop"] = new_layer
            self.origin_layer_names["up2.center_crop"] = new_layer
        elif 'up2.conv' == layer_name:
            self.up2.conv = new_layer
            self.layer_names["up2.conv"] = new_layer
            self.origin_layer_names["up2.conv"] = new_layer
        elif 'up2.conv.double_conv' == layer_name:
            self.up2.conv.double_conv = new_layer
            self.layer_names["up2.conv.double_conv"] = new_layer
            self.origin_layer_names["up2.conv.double_conv"] = new_layer
        elif 'up2.conv.double_conv.0' == layer_name:
            self.up2.conv.double_conv[0] = new_layer
            self.layer_names["up2.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.0"] = new_layer
        elif 'up2.conv.double_conv.1' == layer_name:
            self.up2.conv.double_conv[1] = new_layer
            self.layer_names["up2.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.1"] = new_layer
        elif 'up2.conv.double_conv.2' == layer_name:
            self.up2.conv.double_conv[2] = new_layer
            self.layer_names["up2.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.2"] = new_layer
        elif 'up2.conv.double_conv.3' == layer_name:
            self.up2.conv.double_conv[3] = new_layer
            self.layer_names["up2.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.3"] = new_layer
        elif 'up2.up' == layer_name:
            self.up2.up = new_layer
            self.layer_names["up2.up"] = new_layer
            self.origin_layer_names["up2.up"] = new_layer
        elif 'up2.relu' == layer_name:
            self.up2.relu = new_layer
            self.layer_names["up2.relu"] = new_layer
            self.origin_layer_names["up2.relu"] = new_layer
        elif 'up3' == layer_name:
            self.up3 = new_layer
            self.layer_names["up3"] = new_layer
            self.origin_layer_names["up3"] = new_layer
        elif 'up3.center_crop' == layer_name:
            self.up3.center_crop = new_layer
            self.layer_names["up3.center_crop"] = new_layer
            self.origin_layer_names["up3.center_crop"] = new_layer
        elif 'up3.conv' == layer_name:
            self.up3.conv = new_layer
            self.layer_names["up3.conv"] = new_layer
            self.origin_layer_names["up3.conv"] = new_layer
        elif 'up3.conv.double_conv' == layer_name:
            self.up3.conv.double_conv = new_layer
            self.layer_names["up3.conv.double_conv"] = new_layer
            self.origin_layer_names["up3.conv.double_conv"] = new_layer
        elif 'up3.conv.double_conv.0' == layer_name:
            self.up3.conv.double_conv[0] = new_layer
            self.layer_names["up3.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.0"] = new_layer
        elif 'up3.conv.double_conv.1' == layer_name:
            self.up3.conv.double_conv[1] = new_layer
            self.layer_names["up3.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.1"] = new_layer
        elif 'up3.conv.double_conv.2' == layer_name:
            self.up3.conv.double_conv[2] = new_layer
            self.layer_names["up3.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.2"] = new_layer
        elif 'up3.conv.double_conv.3' == layer_name:
            self.up3.conv.double_conv[3] = new_layer
            self.layer_names["up3.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.3"] = new_layer
        elif 'up3.up' == layer_name:
            self.up3.up = new_layer
            self.layer_names["up3.up"] = new_layer
            self.origin_layer_names["up3.up"] = new_layer
        elif 'up3.relu' == layer_name:
            self.up3.relu = new_layer
            self.layer_names["up3.relu"] = new_layer
            self.origin_layer_names["up3.relu"] = new_layer
        elif 'up4' == layer_name:
            self.up4 = new_layer
            self.layer_names["up4"] = new_layer
            self.origin_layer_names["up4"] = new_layer
        elif 'up4.center_crop' == layer_name:
            self.up4.center_crop = new_layer
            self.layer_names["up4.center_crop"] = new_layer
            self.origin_layer_names["up4.center_crop"] = new_layer
        elif 'up4.conv' == layer_name:
            self.up4.conv = new_layer
            self.layer_names["up4.conv"] = new_layer
            self.origin_layer_names["up4.conv"] = new_layer
        elif 'up4.conv.double_conv' == layer_name:
            self.up4.conv.double_conv = new_layer
            self.layer_names["up4.conv.double_conv"] = new_layer
            self.origin_layer_names["up4.conv.double_conv"] = new_layer
        elif 'up4.conv.double_conv.0' == layer_name:
            self.up4.conv.double_conv[0] = new_layer
            self.layer_names["up4.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.0"] = new_layer
        elif 'up4.conv.double_conv.1' == layer_name:
            self.up4.conv.double_conv[1] = new_layer
            self.layer_names["up4.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.1"] = new_layer
        elif 'up4.conv.double_conv.2' == layer_name:
            self.up4.conv.double_conv[2] = new_layer
            self.layer_names["up4.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.2"] = new_layer
        elif 'up4.conv.double_conv.3' == layer_name:
            self.up4.conv.double_conv[3] = new_layer
            self.layer_names["up4.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.3"] = new_layer
        elif 'up4.up' == layer_name:
            self.up4.up = new_layer
            self.layer_names["up4.up"] = new_layer
            self.origin_layer_names["up4.up"] = new_layer
        elif 'up4.relu' == layer_name:
            self.up4.relu = new_layer
            self.layer_names["up4.relu"] = new_layer
            self.origin_layer_names["up4.relu"] = new_layer
        elif 'outc' == layer_name:
            self.outc = new_layer
            self.layer_names["outc"] = new_layer
            self.origin_layer_names["outc"] = new_layer
        elif 'outc.conv' == layer_name:
            self.outc.conv = new_layer
            self.layer_names["outc.conv"] = new_layer
            self.origin_layer_names["outc.conv"] = new_layer

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def get_order(self, layer_name):
        if layer_name not in self.orders.keys():
            return False
        return self.orders[layer_name]

    def set_order(self, layer_name, order):
        if layer_name not in self.orders.keys():
            return False
        self.orders[layer_name] = order

    def get_outshape(self, layer_name):

        if layer_name not in self.out_shapes.keys():
            return False

        return self.out_shapes[layer_name]

    def set_outshape(self, layer_name, out):

        if layer_name not in self.out_shapes.keys():
            return False

        self.out_shapes[layer_name] = out

    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name, out):
        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name] = out

    def set_Basic_OPS(self, b):
        self.Basic_OPS = b

    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self, c):
        self.Cascade_OPs = c

    def construct(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)

        x = self.up4(x, x1)


        logits = self.outc(x)

        return logits


class MyLoss(Cell):
    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__()
        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = F.ReduceMean()
        self.reduce_sum = F.ReduceSum()
        self.mul = F.Mul()
        self.cast = F.Cast()

    def get_axis(self, x):
        shape = F2.shape(x)
        length = F2.tuple_len(shape)
        perm = F2.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        """
        Computes the weighted loss
        Args:
            weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
                inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
        """
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, base, target):
        raise NotImplementedError


class CrossEntropyWithLogits(MyLoss):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.softmax_cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.cast = F.Cast()

    def construct(self, logits, label):
        # NCHW->NHWC
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, mindspore.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))
        _, _, _, c = F.Shape()(label)
        loss = self.reduce_mean(
            self.softmax_cross_entropy_loss(self.reshape_fn(logits, (-1, c)), self.reshape_fn(label, (-1, c))))
        return self.get_loss(loss)


class Losser(nn.Cell):  # deprecated since we are no longer needing to use this for gradient descent
    def __init__(self, network, criterion):
        super(Losser, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss


def lose(logits, label, network):
    logits = network(logits)
    logits = F.Transpose()(logits, (0, 2, 3, 1))
    logits = F.Cast()(logits, mindspore.float32)
    label = F.Transpose()(label, (0, 2, 3, 1))
    _, _, _, c = F.Shape()(label)
    loss = F.ReduceMean()(
        nn.SoftmaxCrossEntropyWithLogits()(F.Reshape()(logits, (-1, c)), F.Reshape()(label, (-1, c))))
    return get_loss(loss)


class UnetEval(nn.Cell):
    """
    Add Unet evaluation activation.
    """

    def __init__(self, net, need_slice=False, eval_activate="softmax"):
        super(UnetEval, self).__init__()
        self.net = net
        self.need_slice = need_slice
        self.transpose = ops.Transpose()
        self.softmax = ops.Softmax(axis=-1)
        self.argmax = ops.Argmax(axis=-1)
        self.squeeze = ops.Squeeze(axis=0)
        if eval_activate.lower() not in ("softmax", "argmax"):
            raise ValueError("eval_activate only support 'softmax' or 'argmax'")
        self.is_softmax = True
        if eval_activate == "argmax":
            self.is_softmax = False

    def construct(self, x):
        # print("x", x.shape)
        out = self.net(x)
        if self.need_slice:
            out = self.squeeze(out[-1:])
        out = self.transpose(out, (0, 2, 3, 1))
        if self.is_softmax:
            softmax_out = self.softmax(out)
            return softmax_out
        argmax_out = self.argmax(out)
        return argmax_out


class dice_coeff(nn.Metric):
    """Unet Metric, return dice coefficient and IOU."""

    def __init__(self, print_res=False, show_eval=False):
        super(dice_coeff, self).__init__()
        self.show_eval = show_eval
        self.print_res = print_res
        self.img_num = 0
        self.clear()

    def clear(self):
        self._dice_coeff_sum = 0
        self._iou_sum = 0
        self._samples_num = 0
        self.img_num = 0
        if self.show_eval:
            self.eval_images_path = "./draw_eval"
            if os.path.exists(self.eval_images_path):
                shutil.rmtree(self.eval_images_path)
            os.mkdir(self.eval_images_path)

    def draw_img(self, gray, index):
        """
        black：rgb(0,0,0)
        red：rgb(255,0,0)
        green：rgb(0,255,0)
        blue：rgb(0,0,255)
        cyan：rgb(0,255,255)
        cyan purple：rgb(255,0,255)
        white：rgb(255,255,255)
        """
        color = config.color
        color = np.array(color)
        np_draw = np.uint8(color[gray.astype(int)])
        return np_draw

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Need 2 inputs (y_predict, y), but got {}'.format(len(inputs)))
        y = self._convert_data(inputs[1])
        self._samples_num += y.shape[0]
        y = y.transpose(0, 2, 3, 1)
        b, h, w, c = y.shape
        if b != 1:
            raise ValueError('Batch size should be 1 when in evaluation.')
        y = y.reshape((h, w, c))
        start_index = 0
        if not config.include_background:
            y = y[:, :, 1:]
            start_index = 1

        if config.eval_activate.lower() == "softmax":
            y_softmax = np.squeeze(self._convert_data(inputs[0]), axis=0)
            if config.eval_resize:
                y_pred = []
                for i in range(start_index, config.num_classes):
                    y_pred.append(cv2.resize(np.uint8(y_softmax[:, :, i] * 255), (w, h)) / 255)
                y_pred = np.stack(y_pred, axis=-1)
            else:
                y_pred = y_softmax
                if not config.include_background:
                    y_pred = y_softmax[:, :, start_index:]

        elif config.eval_activate.lower() == "argmax":
            y_argmax = np.squeeze(self._convert_data(inputs[0]), axis=0)
            y_pred = []
            for i in range(start_index, config.num_classes):
                if config.eval_resize:
                    y_pred.append(cv2.resize(np.uint8(y_argmax == i), (w, h), interpolation=cv2.INTER_NEAREST))
                else:
                    y_pred.append(np.float32(y_argmax == i))
            y_pred = np.stack(y_pred, axis=-1)
        else:
            raise ValueError('config eval_activate should be softmax or argmax.')

        if self.show_eval:
            self.img_num += 1
            if not config.include_background:
                y_pred_draw = np.ones((h, w, c)) * 0.5
                y_pred_draw[:, :, 1:] = y_pred
                y_draw = np.ones((h, w, c)) * 0.5
                y_draw[:, :, 1:] = y
            else:
                y_pred_draw = y_pred
                y_draw = y
            y_pred_draw = y_pred_draw.argmax(-1)
            y_draw = y_draw.argmax(-1)
            cv2.imwrite(os.path.join(self.eval_images_path, "predict-" + str(self.img_num) + ".png"),
                        self.draw_img(y_pred_draw, 2))
            cv2.imwrite(os.path.join(self.eval_images_path, "mask-" + str(self.img_num) + ".png"),
                        self.draw_img(y_draw, 2))

        y_pred = y_pred.astype(np.float32)
        inter = np.dot(y_pred.flatten(), y.flatten())
        union = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2 * float(inter) / float(union + 1e-6)
        single_iou = single_dice_coeff / (2 - single_dice_coeff)
        if self.print_res:
            print("single dice coeff is: {}, IOU is: {}".format(single_dice_coeff, single_iou))
        self._dice_coeff_sum += single_dice_coeff
        self._iou_sum += single_iou

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._dice_coeff_sum / float(self._samples_num), self._iou_sum / float(self._samples_num)

