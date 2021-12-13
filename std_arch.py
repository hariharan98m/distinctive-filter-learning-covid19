import torch
import torchvision
import torch.nn as nn

def get_vgg16():
    vgg16 = torchvision.models.vgg16(pretrained=False)
    weight = vgg16.features[0].weight.clone()
    vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
    with torch.no_grad():
        vgg16.features[0].weight[:, 0] = weight[:, 0]
    vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=4, bias=True)
    return vgg16

def get_resnet50():
    resnet50 = torchvision.models.resnet50(pretrained=False)
    weight = resnet50.conv1.weight.clone()
    resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
    with torch.no_grad():
        resnet50.conv1.weight[:, 0] = weight[:, 0]
    resnet50.fc = nn.Linear(in_features=2048, out_features=4, bias=True)
    return resnet50

def get_squeezenet():
    squeezenet = torchvision.models.squeezenet1_0(pretrained=False)
    weight = squeezenet.features[0].weight.clone()
    squeezenet.features[0] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(1, 1), padding=1)
    with torch.no_grad():
        squeezenet.features[0].weight[:, 0] = weight[:, 0]
    squeezenet.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
    return squeezenet

def get_resnext():
    resnext = torchvision.models.resnext50_32x4d(pretrained=False)
    weight = resnext.conv1.weight.clone()
    resnext.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
    with torch.no_grad():
        resnext.conv1.weight[:, 0] = weight[:, 0]
    resnext.fc = nn.Linear(2048, 4, bias=True)
    return resnext

def get_densenet():
    densenet = torchvision.models.densenet161(pretrained=False)
    weight = densenet.features.conv0.weight.clone()
    densenet.features.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=3)
    with torch.no_grad():
        densenet.features.conv0.weight[:, 0] = weight[:, 0]
    densenet.classifier = nn.Linear(2208, 4, bias=True)
    return densenet

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)