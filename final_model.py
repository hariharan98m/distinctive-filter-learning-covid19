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


if __name__ == '__main__':
    final_model = ChannelShuffledDualBranchedCNN()
    # final_model.relu_final_out.register_forward_hook(conv_response)
    x = torch.randn(10, 1, 256, 256)
    labels = torch.tensor([[random.randint(0,3)] for i in range(10)])
    tr = TrackResponse(final_model.spatial_conv3x3_final_2, final_model.relu_final_out)
    out = final_model(x)
    rep_vectors = tr.conv_response(labels, out)
    dfl_loss = 0.0
    combos = 0
    for i1, i2 in itertools.combinations(range(len(rep_vectors)), 2):
        dfl_loss+= dice_dissim(rep_vectors[i1], rep_vectors[i2]) if labels[i1] == labels[i2] else dice_sim(rep_vectors[i1], rep_vectors[i2])
        combos+=1
    dfl_loss/=combos
    print(dfl_loss)
    pdb.set_trace()