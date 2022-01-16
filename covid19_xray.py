# from tensorflow.keras.models import load_model
import pdb
import numpy as np
from skimage import morphology, color, io, exposure, img_as_float, transform
import random
# import torch.nn.parallel
import torch.utils.data
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torchsummary import summary
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.metrics import classification_report
import pdb
# from augmentation import random_rotation, random_shear, random_shift, flip_axis
import torchvision
import sys
from sklearn.preprocessing import MultiLabelBinarizer
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
import torch.nn.functional as F
torch.manual_seed(0)
random.seed(0)
import re
import itertools
from std_arch import get_vgg16, get_resnet50, get_squeezenet, get_resnext, get_densenet, count_parameters
# from final_model import SepGroupBlock, ShuffleGroupedConv2d, GroupedConv2d, DepthwiseSeparableConv2d, ChannelShuffledDualBranchedCNN, SevenLayerBaseline, TrackResponse, dice_sim, dice_dissim
# file = '/Users/hmanikan/PycharmProjects/covid19_project/xray_samples/lung_seg.jpg'
# file = '/Users/hmanikan/PycharmProjects/covid19_project/covid19/1-s2.0-S0929664620300449-gr2_lrg-a.jpg'
# file = '/Users/hmanikan/PycharmProjects/covid19_project/xray_samples/jpeg_testing.jpg'
from sklearn.metrics import confusion_matrix, accuracy_score
file = '/xray_samples/covid19/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-000-fig1a.png'
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import random
import pdb
import itertools

'''
Utils
'''
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv =  nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.one_by_one_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, input):
        x = self.depthwise_conv(input)
        x = self.one_by_one_conv(x)
        return x

class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GroupedConv2d, self).__init__()
        self.grouped_conv =  nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups = 4)

    def forward(self, input):
        return self.grouped_conv(input)

class ShuffleGroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ShuffleGroupedConv2d, self).__init__()
        self.groups = 4
        self.in_channels = in_channels
        self.depthwise_conv =  nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups = self.groups)

    def forward(self, input):
        channels_per_grp = self.in_channels // self.groups
        shuffled_channels = []
        for i in range(0, channels_per_grp):
            shuffled_channels.extend([i, i+channels_per_grp, i+ 2*channels_per_grp, i+ 3*channels_per_grp])
        x = input[:, shuffled_channels]
        return self.depthwise_conv(x)

# input h x w x c to h/4 x w/4 x 4c   (128x128x64 to (32x32x256) and (32,32,256) to (8,8,1024)
class SepGroupBlock(nn.Module):
    def __init__(self, in_channels):
        super(SepGroupBlock, self).__init__()
        self.sepgroupblock_depthwise_separable = DepthwiseSeparableConv2d(in_channels, in_channels * 4, 3, 2, 1)
        self.sepgroupblock_avg_pool1 = nn.AvgPool2d(4, 2, 1)

        self.sepgroupblock_conv1 = nn.Conv2d(in_channels, in_channels * 2, 3, 2, 1, bias=False)
        self.sepgroupblock_grouped_conv = GroupedConv2d(in_channels * 2, in_channels * 4, 3, 2, 1)

        self.sepgroupblock_conv2 = nn.Conv2d(in_channels * 2, in_channels * 4, 3, 2, 1, bias=False)

        self.sepgroupblock_batchnorm_1 = nn.BatchNorm2d(in_channels * 4)
        self.sepgroupblock_batchnorm_2 = nn.BatchNorm2d(in_channels * 2)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, input):
        depthwise_separable_out = self.leaky_relu(self.sepgroupblock_batchnorm_1(self.sepgroupblock_depthwise_separable(input)))
        # gives (ndf*4)x64x64
        avgpool1_out = self.sepgroupblock_avg_pool1(depthwise_separable_out)  # gives (ndf*4)x32x32

        # state size. (ndf) x 128 x 128 from layer1_out
        layer2_out = self.leaky_relu(self.sepgroupblock_batchnorm_2(self.sepgroupblock_conv1(input)))  # gives (ndf*2)x64x64
        grouped_conv1_out = self.leaky_relu(self.sepgroupblock_batchnorm_1(self.sepgroupblock_grouped_conv(layer2_out)))  # gives (ndf*4)x32x32

        # state size. (ndf*2) x 64 x 64 from layer2_out
        layer3_out = self.leaky_relu(self.sepgroupblock_batchnorm_1(self.sepgroupblock_conv2(layer2_out)))  # gives (ndf*4)x32x32

        # the bigger out gives (ndf*4)x32x32
        layer3_big_out = (avgpool1_out + grouped_conv1_out + layer3_out) / 3

        return layer3_big_out


def dice_dissim(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def dice_sim(input, target):
    return 1 - dice_dissim(input, target)

class TrackResponse:
    def __init__(self, conv_layer, fm_layer):
        self.conv_layer = conv_layer
        self.fm_layer = fm_layer

        self.fm_layer.register_forward_hook(self.fm_track)

        self.percentile = lambda t, q: t.view(-1).kthvalue(1 + round(.01 * float(q) * (t.numel() - 1))).values.item()

    def fm_track(self, module, input, output):
        self.fm_out = output

    def conv_response(self, labels, cnn_outs):
        conv_weights = self.conv_layer.weight
        rep_vectors = []
        for i, label in enumerate(labels):
            # pdb.set_trace()
            # get the prediction value
            pred_prob = cnn_outs[i, label]
            # get the gradient of class predicted prob wrt feature map
            grads = torch.autograd.grad(pred_prob, self.fm_out, retain_graph=True, create_graph=True)[0][i]
            # perform softmax weighing on this.
            softmax_weighed_grads = nn.functional.softmax(grads, dim=0)
            # do a grad of fm output wrt conv layer weights by passing softmax weighed vector as grad_outputs
            desired = torch.autograd.grad(outputs=self.fm_out[i], inputs = conv_weights, grad_outputs=softmax_weighed_grads, retain_graph=True, create_graph=True)[0]
            # perform mean on desired spatial axes.
            filter_response_scores = torch.mean(desired, dim=(-2,-1)).view(-1)
            # perform norm and save to rep vectors list
            filter_grads = (filter_response_scores - torch.min(filter_response_scores)) / (torch.max(filter_response_scores) - torch.min(filter_response_scores)) * 1.0 + 0.0
            # 90th percentile value
            percentile_val = self.percentile(filter_grads, 90.0)
            # filter grads thresholded
            filter_grads[filter_grads > percentile_val] = 1.0 ; filter_grads[filter_grads <= percentile_val] = 0.0
            rep_vectors.append(filter_grads)
        return rep_vectors


class SevenLayerBaseline(nn.Module):
    def __init__(self):
        super(SevenLayerBaseline, self).__init__()
        ndf = 64
        nc = 1
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 0, bias=False)
        self.conv6 = nn.Conv2d(ndf * 8, ndf * 16, 3, 1, 0, bias=False)
        self.conv_out = nn.Conv2d(ndf * 16, 4, 5, 1, 0, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm2 = nn.BatchNorm2d(ndf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ndf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ndf * 8)
        self.batch_norm16 = nn.BatchNorm2d(ndf * 16)

    def forward(self, input):
        # input is (nc) x 256 x 256
        layer1_out = self.leaky_relu(self.conv1(input))
        # state size. (ndf) x 128 x 128
        layer2_out = self.leaky_relu(self.batch_norm2(self.conv2(layer1_out)))
        # state size. (ndf*2) x 64 x 64
        layer3_out = self.leaky_relu(self.batch_norm4(self.conv3(layer2_out)))
        # state size. (ndf*4) x 32 x 32
        layer4_out = self.leaky_relu(self.batch_norm8(self.conv4(layer3_out)))
        # state size. (ndf*8) x 16 x 16
        layer5_out = self.leaky_relu(self.batch_norm8(self.conv5(layer4_out)))
        # state size. (ndf*8) x 7 x 7
        layer6_out = self.leaky_relu(self.batch_norm16(self.conv6(layer4_out)))
        # state size. (ndf*16) x 5 x 5
        out = self.conv_out(layer6_out)
        # state size. (ndf*8) x 1 x 1
        return out

class ChannelShuffledDualBranchedCNN(nn.Module):
    def __init__(self):
        super(ChannelShuffledDualBranchedCNN, self).__init__()
        self.spatial_conv7x7 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.spatial_conv5x5 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False)

        self.depthwise_separable_link1 = DepthwiseSeparableConv2d(in_channels=128,  out_channels= 128 * 4, kernel_size=3, stride=2, padding=1)
        self.depthwise_separable_link2 = DepthwiseSeparableConv2d(in_channels=256, out_channels= 256 * 2, kernel_size=3, stride=2, padding=1)

        self.avg_pool = nn.AvgPool2d(4, 2, 1)

        self.spatial_conv3x3_block1_1 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.spatial_conv3x3_block1_2 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.spatial_conv3x3_block1_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)

        self.shuffle_blk1 = ShuffleGroupedConv2d(256, 256, 3, 1, 1) # gives 16x16 again
        self.shuffle_blk2 = ShuffleGroupedConv2d(512, 512, 3, 1, 1)

        self.grouped_conv_link1 = GroupedConv2d(64, 256, 3, 2, 1)
        self.grouped_conv_link2 = GroupedConv2d(512, 512, 3, 1, 1)

        self.dropout_5percent = nn.Dropout2d(0.01)
        self.dropout_10percent = nn.Dropout2d(0.01)

        self.spatial_conv3x3_block2_1 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.spatial_conv3x3_block2_2 = nn.Conv2d(512, 512, 3, 2, 1, bias=False)
        self.spatial_conv3x3_block2_3 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)

        self.spatial_conv3x3_final_1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.spatial_conv3x3_final_2 = nn.Conv2d(512, 1024, 3, 2, 1, bias=False)

        self.classifier = nn.Conv2d(1024, 4, 4, 1, 0, bias=False)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.relu_final_out = nn.ReLU()

        # batch norm layers
        self.batch_norm32 = nn.BatchNorm2d(32, track_running_stats = False)
        self.batch_norm64 = nn.BatchNorm2d(64, track_running_stats=False)
        self.batch_norm128 = nn.BatchNorm2d(128, track_running_stats = False)
        self.batch_norm256 = nn.BatchNorm2d(256, track_running_stats = False)
        self.batch_norm512 = nn.BatchNorm2d(512, track_running_stats = False)
        self.batch_norm1024 = nn.BatchNorm2d(1024, track_running_stats=False)

    def forward(self, input):
        # input is (nc) x 256 x 256

        spatial_conv7x7_out = self.relu(self.batch_norm32(self.spatial_conv7x7(input)))
        spatial_conv5x5_out = self.relu(self.batch_norm64(self.spatial_conv5x5(spatial_conv7x7_out)))

        # BLOCK 1
        # side 1
        grouped_conv_link1_out = self.relu(self.batch_norm256(self.grouped_conv_link1(self.dropout_5percent(spatial_conv5x5_out))))
        blk1_side1_out = self.avg_pool(grouped_conv_link1_out)
        # main
        spatial_block1_1_out = self.relu(self.batch_norm128(self.spatial_conv3x3_block1_1(spatial_conv5x5_out)))
        spatial_block1_2_out = self.relu(self.batch_norm256(self.spatial_conv3x3_block1_2(spatial_block1_1_out))) + blk1_side1_out
        spatial_block1_3_out = self.relu(self.batch_norm256(self.spatial_conv3x3_block1_3(spatial_block1_2_out)))
        shuffled_out1 = self.relu(self.batch_norm256(self.shuffle_blk1(spatial_block1_3_out)))
        # side 2
        depthwise_separable_link1_out = self.relu(self.batch_norm512(self.depthwise_separable_link1(self.dropout_10percent(spatial_block1_1_out))))
        blk1_side2_out = self.avg_pool(depthwise_separable_link1_out)

        # print('blk1 side1:',blk1_side1_out.size())
        # print('shuffled1:', shuffled_out1.size())
        # print('blk1 side2:', blk1_side2_out.size())
        # BLOCK 2
        # side 1
        depthwise_separable_link2_out = self.relu(self.batch_norm512(self.depthwise_separable_link2(self.dropout_10percent(shuffled_out1))))
        blk2_side1_out = self.avg_pool(depthwise_separable_link2_out)
        # main
        spatial_block2_1_out = self.relu(self.batch_norm512(self.spatial_conv3x3_block2_1(shuffled_out1))) + blk1_side2_out
        spatial_block2_2_out = self.relu(self.batch_norm512(self.spatial_conv3x3_block2_2(spatial_block2_1_out))) + blk2_side1_out
        spatial_block2_3_out = self.relu(self.batch_norm512(self.spatial_conv3x3_block2_3(spatial_block2_2_out)))
        shuffled_out2 = self.relu(self.batch_norm512(self.shuffle_blk2(spatial_block2_3_out)))
        # side 2
        grouped_conv_link2_out = self.relu(self.batch_norm512(self.grouped_conv_link2(self.dropout_5percent(spatial_block2_1_out))))
        blk2_side2_out = self.avg_pool(grouped_conv_link2_out)

        # print('blk2 side1:', blk2_side1_out.size())
        # print('shuffled2:', shuffled_out2.size())
        # print('blk2 side2:', blk2_side2_out.size())

        # classification
        final1_out = self.relu(self.batch_norm512(self.spatial_conv3x3_final_1(shuffled_out2))) + blk2_side2_out
        final2_out = self.relu_final_out(self.batch_norm1024(self.spatial_conv3x3_final_2(final1_out)))

        return self.classifier(final2_out).view(-1, 4)

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    img = morphology.binary_dilation(img)
    return img

# model_name = 'xray_samples/lung_segmentation.hdf5'
# UNet = load_model(model_name)

def generate_masks(paths):
    inp_shape = [256, 256]
    images_stack = []
    for path in paths:
        img = read_img(path)
        images_stack.append(img)
    images_stack = np.stack(images_stack)

    # do prediction
    pred = UNet.predict(images_stack)[..., 0]
    pr = pred > 0.5

    # Remove regions smaller than 2% of the image
    convert_bool = lambda x: np.uint8(np.where(x == True, 1.0, 0.0) * 255)
    pr = [ convert_bool(remove_small_regions(pr_slice, 0.05 * np.prod(inp_shape))) for pr_slice in pr]
    return pr

def read_img(path, mask = False):
    inp_shape = [256, 256]
    img = img_as_float(io.imread(path, as_gray=True))
    if mask:
        return img
    img = transform.resize(img, inp_shape)
    # img = exposure.equalize_hist(img)
    img = np.expand_dims(img, -1)
    # img = img - img.mean()
    # std = img.std()
    # if std == 0.0:
    #     std += 10e-5
    # img /= std
    return img

def read_pil_image(path, mask = False):
    inp_shape = [256, 256]
    return pil_image.open(path).resize(inp_shape).convert('L')


def get_segmented_img(path):
    img = read_img(path)
    path_comps = path.split('/')
    image_name = '.'.join(path_comps[-1].split('.')[:-1])
    mask_path = 'lung_masks/' + path_comps[-2] + '/' + image_name + '.jpg'
    mask = read_img(mask_path, mask = True)[...,np.newaxis]
    return np.multiply(img, mask)

class COVID19_dataset():

    def __init__(self, samples, transforms = None):
        self.samples = []
        self.labels = []
        self.weight_per_sample = []
        self.classes = ['normal', 'covid19', 'bacterial', 'viral']
        total_samples = sum([len(items) for items in samples.values()])
        for label, cls in enumerate(self.classes):
            cls_items = samples[cls]
            self.samples.extend(cls_items)
            self.labels.extend([label] * len(cls_items))
            self.weight_per_sample.extend([total_samples / len(cls_items)] * len(cls_items))

        self.weight_per_sample = torch.DoubleTensor(self.weight_per_sample)
        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(self.weight_per_sample, len(self.samples))
        self.transforms = transforms
        self.toTensorTransform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        # segmented_img = get_segmented_img(self.samples[item]).astype(np.float32)
        img = read_pil_image(self.samples[item])
        # pdb.set_trace()
        if self.transforms is not None:
            segmented_img = self.transforms(img)
        else:
            segmented_img = self.toTensorTransform(img)

        return segmented_img, torch.tensor([self.labels[item]], dtype = torch.long)

nc = 1
ndf = 64
lr = 0.002
beta1 = 0.5
ngpu = 2

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv =  nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)


    def forward(self, input):
        pass

'''
Enhanced Model
'''
# image to (256, 256, 64), then conv to (128,128,64) --1 blk -- shuffle block --1 block -- 2 convs done.
class BlockedModel(nn.Module):
    def __init__(self, ngpu):
        super(BlockedModel, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Conv2d(nc, ndf, 3, 2, 1, bias=False)
        self.sepgroupconv1 = SepGroupBlock(ndf)
        # gives ndf * 4
        self.shuffle = ShuffleGroupedConv2d(ndf * 4, ndf * 4, 3, 1, 1)  # gives 16x16 again
        self.conv2 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False)  # gives 8x8
        # gives ndf * 8
        self.sepgroupconv2 = SepGroupBlock(ndf * 8)
        # gives ndf * 32
        self.conv3 = nn.Conv2d(ndf * 32, ndf*64, 3, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(ndf * 64, 4, 4, 1, 0, bias=False)

        # activation function
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # batch norm layers
        self.batch_norm1 = nn.BatchNorm2d(ndf)
        self.batch_norm4 = nn.BatchNorm2d(ndf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ndf * 8)
        self.batch_norm64 = nn.BatchNorm2d(ndf * 64)

    def forward(self, input):
        # input is (nc) x 256 x 256
        print('input:', input.shape)
        conv1_out = self.leaky_relu(self.batch_norm1(self.conv1(input))) # gives ndfx128x128
        print('conv1:', conv1_out.shape)
        block1_out = self.sepgroupconv1(conv1_out) # is of size (32x32x256)
        print('block1-out:', block1_out.shape)

        shuffle_out = self.leaky_relu(self.batch_norm4(self.shuffle(block1_out)))   # give 32x32x256
        print('shuffle-out:', shuffle_out.shape)
        conv2_out = self.leaky_relu(self.batch_norm8(self.conv2(shuffle_out)))  # gives (512) x 32 x 32
        print('conv2-out:', conv2_out.shape)

        block2_out = self.sepgroupconv2(conv2_out)  # 8x8x2048 which is 8 x8 x (ndf*32)
        print('block2-out:', block2_out.shape)
        # last 2 layers
        # penultimate layer
        penultimate =  self.leaky_relu(self.batch_norm64(self.conv3(block2_out))) # gives (ndf*64) x 4 x 4
        print('penultimate-out:', block2_out.shape)
        # last layer
        last_out = self.conv4(penultimate) # gives 4 x 1 x 1
        print('last-out:', block2_out.shape)
        return last_out

'''
import torch
from prepare_dataset import BlockedModel
input = torch.randn(32, 1, 256, 256)
model = BlockedModel(2)
model(input)
'''

def save_bool_mask_2d(bool_mask, path):
    array_to_img(bool_mask[...,np.newaxis]).save(path)

def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])

class ModelWithMoreLayers(nn.Module):
    def __init__(self, ngpu):
        super(ModelWithMoreLayers, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Conv2d(nc, ndf, 3, 2, 1, bias=False)
        self.depthwise_separable1 = DepthwiseSeparableConv2d(ndf, ndf * 4, 3, 2, 1)
        self.avg_pool1 = nn.AvgPool2d(4, 2, 1)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False)
        self.grouped_conv = GroupedConv2d(ndf * 2, ndf * 4, 3, 2, 1)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False) # gives 16x16 till here
        self.shuffle = ShuffleGroupedConv2d(ndf * 8, ndf * 8, 3, 1, 1) # gives 16x16 again

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1, bias=False) # gives 8x8
        self.conv6 = nn.Conv2d(ndf * 16, ndf*16, 3, 2, 1, bias=False)
        self.conv7 = nn.Conv2d(ndf * 16, 4, 4, 1, 0, bias=False)

        # activation function
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)  # nn.ReLU

        # batch norm layers
        self.batch_norm1 = nn.BatchNorm2d(ndf * 1, track_running_stats = True)
        self.batch_norm2 = nn.BatchNorm2d(ndf * 2, track_running_stats = True)
        self.batch_norm4 = nn.BatchNorm2d(ndf * 4, track_running_stats = True)
        self.batch_norm8 = nn.BatchNorm2d(ndf * 8, track_running_stats = True)
        self.batch_norm16 = nn.BatchNorm2d(ndf * 16, track_running_stats = True)

    def forward(self, input):
        # input is (nc) x 256 x 256
        layer1_out = self.leaky_relu(self.batch_norm1(self.conv1(input))) # gives ndfx128x128
        depthwise_separable1_out = self.leaky_relu(self.batch_norm4(self.depthwise_separable1(layer1_out)))
        # gives (ndf*4)x64x64
        avgpool1_out = self.avg_pool1(depthwise_separable1_out) # gives (ndf*4)x32x32

        # state size. (ndf) x 128 x 128 from layer1_out
        layer2_out = self.leaky_relu(self.batch_norm2(self.conv2(layer1_out))) # gives (ndf*2)x64x64
        grouped_conv1_out = self.leaky_relu(self.batch_norm4(self.grouped_conv(layer2_out))) # gives (ndf*4)x32x32

        # state size. (ndf*2) x 64 x 64 from layer2_out
        layer3_out = self.leaky_relu(self.batch_norm4(self.conv3(layer2_out))) # gives (ndf*4)x32x32

        # the bigger out gives (ndf*4)x32x32
        layer3_big_out = (avgpool1_out + grouped_conv1_out + layer3_out)/3

        # state size. (ndf*4) x 32 x 32
        layer4_out = self.leaky_relu(self.batch_norm8(self.conv4(layer3_big_out))) # gives (ndf * 8) x 16 x 16
        # add a shuffle conv
        shuffle_out = self.leaky_relu(self.batch_norm8(self.shuffle(layer4_out))) # gives (ndf * 8)x16x16

        # state size. (ndf*8) x 16 x 16 - Coming to the last 3 here.
        layer5_out = self.leaky_relu(self.batch_norm16(self.conv5(shuffle_out)))  # gives (ndf*16) x 8 x 8
        # penultimate layer
        layer6_out =  self.leaky_relu(self.batch_norm16(self.conv6(layer5_out))) # gives (ndf*16) x 4 x 4
        # last layer
        layer7_out = self.conv7(layer6_out) # gives 4 x 1 x 1
        return layer7_out.view(-1, 4) #, layer6_out


def novelty_component(model, layer_out, conv_layer_name):
    dice_vectors = []
    for sample in range(layer_out.size(0)):
        layer_out[sample].backward(F.softmax(layer_out[sample], dim=0), retain_graph = True)
        filter_grads = [layer for name, layer in model.named_children() if name==conv_layer_name][0].weight.requires_grad
        filter_grads = (filter_grads - torch.min(filter_grads))/(torch.max(filter_grads) - torch.min(filter_grads)) * 1.0 + 0.0
        dice_vectors.append(torch.where(filter_grads > 0.9, 1.0, 0.0).view(-1))
    return dice_vectors

def ensure_exists(paths):
    for path in paths:
        Path.mkdir(path, exist_ok=True, parents=True)

def obtain_latest_checkpoint(dir_path):
    files = sorted([(path, path.name) for path in dir_path.iterdir()], key = lambda x: int(re.search('\d+', x[1]).group()), reverse = True)
    return files[0][0] if len(files) else None

def load_training_stats(path):
    return json.load(open(path, 'r'))

def model_train(dataloader_train, dataloader_val, model, epochs = 50, path_to_save= Path('xray_outputs/'), model_name = 'baseline', fold = None, response_tracker = None):
    # pdb.set_trace()
    # Initialize BCELoss function
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()

    if fold is None:
        model_path_to_save = path_to_save / model_name / 'xray_models' / 'epoch%d.pth'
        metrics_path_to_save = path_to_save / model_name / 'xray_metrics' / 'xray_metrics.json'
    else:
        model_path_to_save = path_to_save / model_name / fold / 'xray_models' / 'epoch%d.pth'
        metrics_path_to_save = path_to_save / model_name / fold / 'xray_metrics' / 'xray_metrics.json'
    # makes sure these directories are created.
    ensure_exists([model_path_to_save.parent, metrics_path_to_save.parent])

    model = model.to(device)

    start_epoch = 1
    best_val_acc = 0.0
    training_stats = {}

    # Setup Adam optimizers for both G and D
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))


    latest_checkpoint_file = obtain_latest_checkpoint(model_path_to_save.parent)
    if latest_checkpoint_file is not None:
        checkpoint = torch.load(str(latest_checkpoint_file))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        start_epoch = checkpoint['epoch']
        # best_val_acc = checkpoint['best_val_acc']
        training_stats = load_training_stats(str(metrics_path_to_save))

    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use ", torch.cuda.device_count(), " GPUs for the Model!")
        model = nn.DataParallel(model, list(range(ngpu)))

    # epochs
    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            epoch_loss = epoch_accuracy = 0
            model.train()
            for batch_index, data in enumerate(dataloader_train):
                # if batch_index == 18:
                #     break
                cts, class_labels = data
                cts = cts.to(device)
                class_labels = class_labels.to(device)

                batch_size = class_labels.size(0)
                cls_labels_flat = class_labels.view(-1)

                # onehot_labels = torch.zeros(batch_size, 4)
                # onehot_labels.scatter_(1, class_labels, 1.0)

                model.zero_grad()

                output = model(cts)
                output =output.view(batch_size, 4)
                error = criterion(output, cls_labels_flat)

                # Do DFL
                dfl_loss = 0.0
                if response_tracker is not None:
                    rep_vectors = response_tracker.conv_response(class_labels, output)
                    dfl_loss = 0.0; combos = 0
                    for i1, i2 in itertools.combinations(range(len(rep_vectors)), 2):
                        dfl_loss += dice_dissim(rep_vectors[i1], rep_vectors[i2]) if cls_labels_flat[i1] == cls_labels_flat[i2] else dice_sim(rep_vectors[i1], rep_vectors[i2])
                        combos += 1
                    dfl_loss /= combos

                tot_loss = error + dfl_loss

                # acc gradients
                tot_loss.backward()

                # perform grad update
                optimizer.step()

                output = output.detach().cpu()
                cls_labels_flat = cls_labels_flat.cpu()
                # model accuracy
                _, model_output_indices = torch.max(output, dim = -1)

                model_accuracy = torch.sum(model_output_indices == cls_labels_flat).float() / torch.numel(class_labels)

                # Training Stats:
                print('[%d/%d] [%d/%d]\tTRAIN Loss %.4f,\tAccuracy: %.4f' % (epoch, start_epoch + epochs, batch_index+1, len(dataloader_train), tot_loss.item(), model_accuracy.item()))

                epoch_loss += tot_loss.item(); epoch_accuracy += model_accuracy.item()

            with torch.no_grad():
                val_acc = 0.0; val_loss = 0.0
                model.eval()
                val_labels, val_preds = [], []
                for data, labels in dataloader_val:
                    data, labels, batch_size = data.to(device), labels.view(-1), labels.size(0)
                    output = model(data)
                    output = output.view(batch_size, 4).detach().cpu()
                    _, model_output_indices = torch.max(output, dim=-1)
                    # pdb.set_trace()
                    val_acc += (torch.sum(model_output_indices == labels).float() / torch.numel(labels)).item()
                    val_loss += criterion(output, labels).item()
                    val_preds.extend(model_output_indices.numpy().tolist())
                    val_labels.extend(labels.numpy().tolist())

            epoch_stat = {
                'loss': epoch_loss / len(dataloader_train),
                'acc': epoch_accuracy/len(dataloader_train),
                'val_loss': val_loss/len(dataloader_val),
                'val_acc': val_acc/len(dataloader_val),
                'val_labels': val_labels,
                'val_preds': val_preds
            }

            if epoch_stat['val_acc'] > best_val_acc:
                best_val_acc = epoch_stat['val_acc']
                checkpoint_model(epoch, model, optimizer, best_val_acc, str(model_path_to_save) % epoch)

            print('\nEpoch Loss:%.4f,\tEpoch Accuracy:%.4f' % (epoch_stat['loss'], epoch_stat['acc']))
            print('Epoch VAL Loss:%.4f,\tEpoch VAL Accuracy:%.4f\n' % (epoch_stat['val_loss'], epoch_stat['val_acc']))
            # save stat to global stats.
            training_stats['epoch%d' % epoch] = epoch_stat
    except KeyboardInterrupt:
        print('Oh its an interrupt.')
    finally:
        print('Saving checkpoints and metrics')
        # pdb.set_trace()
        checkpoint_metrics(training_stats, metrics_path= metrics_path_to_save)
        checkpoint_model(epoch, model, optimizer, best_val_acc, str(model_path_to_save) % epoch)
        print('DONE saving')
    return model

def checkpoint_metrics(object, metrics_path):
    json.dump(object, open(str(metrics_path), 'w'), indent=2)

def checkpoint_model(epoch, model, opt, best_val_acc, model_path):
    model_state_dict = model.module.state_dict() if (device.type == 'cuda') else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'opt_state_dict': opt.state_dict(),
        'best_val_acc': best_val_acc
    }, model_path)


# Code to generate masks
should_generate_masks = False

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_train_val_test_splits(classes, paths, ratios, all_samples = False, folds= None):
    train_val_test_samples = {
        'train': {},
        'val': {},
        'test': {},
        'all_samples': {}
    }

    if folds is not None:
        train_val_test_samples_by_folds = { 'fold' + str(i): {'val': {}, 'train': {}} for i in range(folds) }

    cls_paths = {}
    for cls in classes:
        cls_paths[cls] = sorted([str(path).strip() for path in paths[cls].iterdir()])[:500]
        random.seed(0)
        random.shuffle(cls_paths[cls])

    if folds is None:
        for cls in classes:
            cls_sample_paths = cls_paths[cls]
            if all_samples:
                train_val_test_samples['all_samples'][cls] = cls_sample_paths
                continue
            first_split = round(len(cls_sample_paths) * ratios[0])
            second_split = first_split + round(len(cls_sample_paths) * ratios[1])
            # assign splits to paths
            train_val_test_samples['train'][cls] = cls_sample_paths[:first_split]
            train_val_test_samples['val'][cls] = cls_sample_paths[first_split:second_split]
            train_val_test_samples['test'][cls] = cls_sample_paths[second_split:]

        return train_val_test_samples

    else:
        for fold in range(folds):
            for cls in classes:
                cls_sample_paths = np.array(cls_paths[cls])
                val_indices = list(range(round(100/folds * fold), round(100/folds * (fold + 1))+1))
                train_indices = list(set(range(0, len(cls_sample_paths))) - set(val_indices))
                train_val_test_samples_by_folds['fold' + str(fold)]['val'][cls] =  cls_sample_paths[val_indices].tolist()
                train_val_test_samples_by_folds['fold' + str(fold)]['train'][cls] =  cls_sample_paths[train_indices].tolist()
        return train_val_test_samples_by_folds

def save_test_samples(samples):
    for key, items in samples.items():
        samples[key] = list(map(lambda x: str(x) + '\n', items))
    json.dump(samples, open('test_samples_xray_withTestSplits_AllSamples_NoAug_NoNovelty.txt', 'w'), indent=2)

def get_test_set_perf(model, dataloader_test, model_path_to_save):
    latest_checkpoint_file = obtain_latest_checkpoint(model_path_to_save.parent)
    if latest_checkpoint_file is not None:
        checkpoint = torch.load(str(latest_checkpoint_file))
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
    print('best val acc: ', best_val_acc)

    model= model.to(device)

    model.eval()
    model_pred, test_labels, model_output= [], [], []
    for data, labels in dataloader_test:
        with torch.no_grad():
            data, labels, batch_size = data.to(device), labels.view(-1), labels.size(0)
            output= model(data)
            output = output.view(batch_size, 4).detach().cpu()
            output = output.view(batch_size, 4).detach().cpu()
            softmax_output = F.softmax(output, dim=-1)
            _, model_output_indices = torch.max(softmax_output, dim=-1)
            model_pred.extend(model_output_indices.numpy().tolist())
            model_output.extend(softmax_output.cpu().numpy().tolist())
            test_labels.extend(labels.numpy().tolist())
        # pdb.set_trace()
    return test_labels, model_pred, compute_all_metrics(test_labels, model_pred, model_output = model_output)

def compute_all_metrics(test_labels, model_pred, model_output = None):
    conf_matrix = confusion_matrix(test_labels, model_pred)
    classwise_metrics = {}
    print(conf_matrix)
    mlb = MultiLabelBinarizer()
    mlb.fit([[0], [1], [2], [3]])
    labs = mlb.transform(np.array(test_labels)[..., np.newaxis])
    model_preds = mlb.transform(np.array(model_pred)[..., np.newaxis]) if model_output is None else np.array(model_output)

    classes = ['normal', 'covid19', 'bacterial', 'viral']
    for cls_index, cls in enumerate(classes):
        tp = conf_matrix[cls_index, cls_index]
        fp = np.sum(conf_matrix[:, cls_index]) - tp
        fn = np.sum(conf_matrix[cls_index, :]) - tp
        tn = np.sum(conf_matrix) - (tp + fp + fn)
        accuracy = (tp + tn)/(tp + fp + tn + fn)
        precision = tp/(tp + fp)
        recall = tp / (tp + fn)
        specificity = tn/(tn + fp)
        f1_score = 2 * precision * recall / (precision + recall)

        fpr, tpr, _ = roc_curve(labs[:, cls_index], model_preds[:, cls_index])
        roc_auc = auc(fpr, tpr)

        classwise_metrics[cls] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'fpr': fpr, 'tpr': tpr
        }
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([classwise_metrics[cls]['fpr'] for cls in classes]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for cls in classes:
        mean_tpr += np.interp(all_fpr, classwise_metrics[cls]['fpr'], classwise_metrics[cls]['tpr'])
    mean_tpr /= len(classes)

    tpr_covid, fpr_covid = None, None
    for cls in classes:
        tpr_vals = classwise_metrics[cls].pop('tpr')
        fpr_vals = classwise_metrics[cls].pop('fpr')
        if cls == 'covid19':
            tpr_covid = tpr_vals; fpr_covid = fpr_vals

    classwise_metrics['macro_roc'] = auc(all_fpr, mean_tpr)
    classwise_metrics['macro_f1'] = np.mean([classwise_metrics[cls]['f1_score'] for cls in classes])
    classwise_metrics['accuracy'] = accuracy_score(test_labels, model_pred)
    classwise_metrics['macro_precision'] =np.mean([classwise_metrics[cls]['precision'] for cls in classes])
    classwise_metrics['macro_recall'] = np.mean([classwise_metrics[cls]['recall'] for cls in classes])
    return classwise_metrics, tpr_covid, fpr_covid, all_fpr, mean_tpr

test_transform = torchvision.transforms.Compose([
        # torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(p=1.0),
        torchvision.transforms.RandomAffine(10, translate=(0.1, 0.1), scale=None, shear=5, resample=False, fillcolor=0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.499010], [0.24]),
    ])

# pdb.set_trace()

root_dir = 'xray_samples' if len(sys.argv)== 1 else sys.argv[1]
classes, paths = ['normal', 'covid19', 'bacterial', 'viral'], {
    'normal': Path(root_dir) / 'normal',
    'covid19': Path(root_dir)  / 'covid19',
    'bacterial': Path(root_dir) / 'bacterial_pneumonia',
    'viral': Path(root_dir) / 'viral_pneumonia'
}

validation_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.499010], [0.24]),
])

if __name__ == '__main__':
    data_dir = '/home/vshshv3/Downloads/covid19_xray_processed' #'xray_samples' # '/home/vshshv3/Downloads/covid19_xray_processed'
    classes, paths = ['normal', 'covid19', 'bacterial', 'viral'], {
        'normal': Path(data_dir) / 'normal',
        'covid19': Path(data_dir) / 'covid19',
        'bacterial': Path(data_dir) / 'bacterial_pneumonia',
        'viral': Path(data_dir) / 'viral_pneumonia'
    }

    train_val_test_samples_folds = get_train_val_test_splits(classes, paths, None, all_samples=False, folds=5)
    for fold, fold_dataset in train_val_test_samples_folds.items():
        print('Fold: ', fold)
        # covid19 dataset objects
        covid19_dataset_train = COVID19_dataset(fold_dataset['train'], transforms=validation_transform)
        covid19_dataset_val = COVID19_dataset(fold_dataset['val'], transforms=validation_transform)
        # covid19 dataloader objects
        covid19_sampler = covid19_dataset_train.sampler
        dataloader_train = torch.utils.data.DataLoader(covid19_dataset_train, sampler=covid19_sampler,
                                                                batch_size=224,
                                                                num_workers=50)
        dataloader_val = torch.utils.data.DataLoader(covid19_dataset_val, batch_size=128,
                                                              num_workers=50)
        path = 'xray_models/fold/' + str(fold)
        model_name = 'csdb_folds'
        model = ChannelShuffledDualBranchedCNN()

        model_train(dataloader_train, dataloader_val, model, epochs = 73, model_name=model_name, response_tracker = None, fold = fold)

        path_to_save = Path('xray_outputs/')

        labels, pred, metrics = get_test_set_perf(model, dataloader_val, model_path_to_save=path_to_save / model_name / fold / 'xray_models' / 'epoch%d.pth')
        classwise_metrics, tpr_covid, fpr_covid, all_fpr, mean_tpr = metrics
        json.dump(classwise_metrics, open(str(path_to_save / model_name / fold / 'xray_metrics' / 'computed_metrics.json'), 'w'), indent=4)


''' Done: resnet50, resnext_50_32x4d, all'''
if __name__ == '__main__2':
    # pdb.set_trace()
    data_dir = '/home/vshshv3/Downloads/covid19_xray_processed' # #'xray_samples' #
    classes, paths = ['normal', 'covid19', 'bacterial', 'viral'], {
        'normal': Path(data_dir) / 'normal',
        'covid19': Path(data_dir) / 'covid19',
        'bacterial': Path(data_dir) / 'bacterial_pneumonia',
        'viral': Path(data_dir) / 'viral_pneumonia'
    }

    train_val_test_samples = get_train_val_test_splits(classes, paths, [0.85, 0.14, 0.01])
    all_samples = get_train_val_test_splits(classes, paths, None, True)

    covid19_dataset = COVID19_dataset(all_samples['all_samples'], transforms= validation_transform)
    covid19_dataset_train = COVID19_dataset(train_val_test_samples['train'], transforms= validation_transform)
    covid19_dataset_val = COVID19_dataset(train_val_test_samples['val'], transforms=validation_transform)
    covid19_dataset_test = COVID19_dataset(train_val_test_samples['test'], transforms=validation_transform)
    # save_test_samples(train_val_test_samples['test'])

    covid19_sampler = covid19_dataset_train.sampler
    balanced_dataloader_train = torch.utils.data.DataLoader(covid19_dataset_train, sampler=covid19_sampler, batch_size=50,
                                                      num_workers=50)

    balanced_dataloader_val = torch.utils.data.DataLoader(covid19_dataset_val, batch_size=25,
                                                            num_workers=50)

    balanced_dataloader_test = torch.utils.data.DataLoader(covid19_dataset_test, batch_size=128,
                                                          num_workers=50)

    print('total: ', len(covid19_dataset))
    if should_generate_masks:
        for batch in range(0, len(covid19_dataset), 64):
            batch_paths = covid19_dataset.samples[batch: batch + 64]
            masks = generate_masks(batch_paths)
            print('batch:', batch)
            for path, mask in zip(batch_paths, masks):
                path_comps = path.split('/')
                image_name = '.'.join(path_comps[-1].split('.')[:-1])
                save_bool_mask_2d(mask, 'lung_masks/' + path_comps[-2] + '/' + image_name + '.jpg')

    # means, stds = [], []
    # for batch in balanced_dataloader_train:
    #     batch_data, labels = batch
    #     print('cls0: ', torch.sum(labels == 0).item(), 'cls1: ', torch.sum(labels == 1).item(), 'cls2:', torch.sum(labels ==2).item(), 'cls3:', torch.sum(labels ==3).item())
    #     mean, std = torch.mean(batch_data).item(), torch.std(batch_data).item()
    #     print('mean: ', mean, 'std: ', std)
    #     means.append(mean); stds.append(std);
    # print('Average mean %f, average std: %.2f' % (np.mean(means), np.mean(stds)))

    vgg16_model = get_vgg16()
    seven_layer_net = SevenLayerBaseline()
    squeezenet = get_squeezenet()
    densenet = get_densenet()
    resnext = get_resnext()
    resnet = get_resnet50()

    print(resnext.__class__.__name__, ', params:', count_parameters(resnext))
    # print(squeezenet.__class__.__name__, ', params:', count_parameters(squeezenet))
    pdb.set_trace()
    # our_model = ModelWithMoreLayers(2)

    # our_model = ChannelShuffledDualBranchedCNN()
    model_name = 'seven_baseline' #'channel_shuffled_dual_branched' #'csdb_dfl'
    path_to_save = Path('xray_outputs/')
    # response tracker.
    # tr = TrackResponse(our_model.spatial_conv3x3_final_2, our_model.relu_final_out)

    # model_train(balanced_dataloader_train, balanced_dataloader_val, seven_layer_net, epochs = 40, model_name=model_name, response_tracker = None)

    # for data, labels in balanced_dataloader_train:
    #     preds = model(data)
    #     preds = preds.view(-1, 4)
    #     novelty_component(model, layer6, 'conv6')
    # pdb.set_trace()
    labels, pred, metrics = get_test_set_perf(seven_layer_net, balanced_dataloader_val, model_path_to_save = path_to_save / model_name / 'xray_models' / 'epoch%d.pth')
    print(metrics)
    json.dump(metrics, open(str(path_to_save / model_name / 'xray_metrics' / 'computed_metrics.json'), 'w'), indent=4)