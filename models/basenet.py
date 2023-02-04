from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
from models.googlenet import googlenet


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambd

        return output, None


def grad_reverse(x, lambd=1.0):
    return ReverseLayerF.apply(x)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class Efficientnet_B0_Base(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(Efficientnet_B0_Base, self).__init__()
        efficientnet = models.efficientnet_b0(pretrained=pret)
        self.features = nn.Sequential(*list(efficientnet.
                                            features._modules.values())[:])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return x

class Efficientnet_B4_Base(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(Efficientnet_B4_Base, self).__init__()
        efficientnet = models.efficientnet_b4(pretrained=pret)
        self.features = nn.Sequential(*list(efficientnet.
                                            features._modules.values())[:])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return x

class GoogLeNet_Base(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(GoogLeNet_Base, self).__init__()
        self.features = googlenet(pretrained=pret)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return x

class VGG16_Base(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGG16_Base, self).__init__()
        self.features = models.vgg16(pretrained=pret)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return x

class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        pred = self.fc(x) / self.temp
        return x, pred


class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=1.0):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x, x_out


class Predictor_DANN(nn.Module):
    def __init__(self, num_class=64, inc=512, temp=0.05):
        super(Predictor_DANN, self).__init__()
        self.fc_cls = nn.Linear(inc, num_class, bias=False)
        self.fc_dmn = nn.Linear(inc, 2, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=True, eta=1.0):
        x = F.normalize(x)
        reverse_x = grad_reverse(x, eta)
        out_pred = self.fc_cls(x) / self.temp
        out_domn = self.fc_dmn(reverse_x)
        return out_pred, out_domn


class Predictor_deep_DANN(nn.Module):
    def __init__(self, num_class=64, inc=512, temp=0.05):
        super(Predictor_deep_DANN, self).__init__()
        self.fc = nn.Linear(inc, 512)
        self.fc_cls = nn.Linear(512, num_class, bias=False)
        self.fc_dmn = nn.Linear(512, 2, bias=False)
        self.num_class = num_class
        self.temp = temp


class Predictor_APE(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_APE, self).__init__()
        self.fc= nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x, x_out


class Predictor_deep_APE(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep_APE, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=1.0):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x, x_out
