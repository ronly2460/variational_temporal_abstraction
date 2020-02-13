import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, kl_divergence


class Flatten(nn.Module):
    def forward(self, input_data):
        if len(input_data.size()) == 4:
            return input_data.view(input_data.size(0), -1)
        else:
            return input_data.view(input_data.size(0), input_data.size(1), -1)


class LinearLayer(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 nonlinear=nn.ELU(inplace=True)):
        super(LinearLayer, self).__init__()
        # linear
        self.linear = nn.Linear(in_features=input_size,
                                out_features=output_size)

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.linear(input_data))


class ConvLayer1D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvLayer1D, self).__init__()
        # linear
        self.linear = nn.Conv1d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm1d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class ConvLayer2D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvLayer2D, self).__init__()
        # linear
        self.linear = nn.Conv2d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm2d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class ConvTransLayer2D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvTransLayer2D, self).__init__()
        # linear
        self.linear = nn.ConvTranspose2d(in_channels=input_size,
                                         out_channels=output_size,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm2d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class RecurrentLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size):
        super(RecurrentLayer, self).__init__()
        # rnn cell
        self.rnn_cell = nn.GRUCell(input_size=input_size,
                                   hidden_size=hidden_size)

    def forward(self, input_data, prev_state):
        return self.rnn_cell(input_data, prev_state)


class LatentDistribution(nn.Module):
    def __init__(self,
                 input_size,
                 latent_size,
                 feat_size=None):
        super(LatentDistribution, self).__init__()
        if feat_size is None:
            self.feat = nn.Identity()
            feat_size = input_size
        else:
            self.feat = LinearLayer(input_size=input_size,
                                    output_size=feat_size)

        self.mean = LinearLayer(input_size=feat_size,
                                output_size=latent_size,
                                nonlinear=nn.Identity())

        self.std = LinearLayer(input_size=feat_size,
                               output_size=latent_size,
                               nonlinear=nn.Sigmoid())

    def forward(self, input_data):
        feat = self.feat(input_data)
        return Normal(loc=self.mean(feat), scale=self.std(feat))


class Encoder(nn.Module):
    def __init__(self,
                 output_size=None,
                 feat_size=64):
        super(Encoder, self).__init__()
        network_list = [ConvLayer2D(input_size=3,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 16 x 16
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 8 x 8
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 4 x 4
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0),  # 1 x 1
                        Flatten()]
        if output_size is not None:
            network_list.append(LinearLayer(input_size=feat_size,
                                            output_size=output_size))
            self.output_size = output_size
        else:
            self.output_size = feat_size

        self.network = nn.Sequential(*network_list)

    def forward(self, input_data):
        return self.network(input_data)


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 feat_size=64):
        super(Decoder, self).__init__()
        if input_size == feat_size:
            self.linear = nn.Identity()
        else:
            self.linear = LinearLayer(input_size=input_size,
                                      output_size=feat_size,
                                      nonlinear=nn.Identity())
        self.network = nn.Sequential(ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size,
                                                      kernel_size=4,
                                                      stride=1,
                                                      padding=0),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=3,
                                                      normalize=False,
                                                      nonlinear=nn.Tanh()))

    def forward(self, input_data):
        return self.network(self.linear(input_data).unsqueeze(-1).unsqueeze(-1))


class PriorBoundaryDetector(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=2):
        super(PriorBoundaryDetector, self).__init__()
        self.network = LinearLayer(input_size=input_size,
                                   output_size=output_size,
                                   nonlinear=nn.Identity())

    def forward(self, input_data):
        logit_data = self.network(input_data)
        return logit_data


class PostBoundaryDetector(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=2,
                 num_layers=1):
        super(PostBoundaryDetector, self).__init__()
        network = list()
        for l in range(num_layers):
            network.append(ConvLayer1D(input_size=input_size,
                                       output_size=input_size))
        network.append(ConvLayer1D(input_size=input_size,
                                   output_size=output_size,
                                   normalize=False,
                                   nonlinear=nn.Identity()))
        self.network = nn.Sequential(*network)

    def forward(self, input_data_list):
        input_data = input_data_list.permute(0, 2, 1)
        return self.network(input_data).permute(0, 2, 1)
