import torch
from torch import nn
import numpy as np
import src.gabor_functions as gf


class MFNet(nn.Module):
    """
    MFNet model definition
    """

    def __init__(self):
        super(MFNet, self).__init__()
        self.features = nn.Sequential(
            # Apply log-Gabor filter in each channel
            MappingLayer2d(),
            nn.InstanceNorm2d(1, affine=False),
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )
        self.descr = nn.Sequential(
            nn.Linear(64 * 8 * 8 * 9, 256),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.descr(x)
        return x


class MappingLayer2d(nn.Module):
    def __init__(self):
        super(MappingLayer2d, self).__init__()

    def forward(self, input_x):
        kshape_i = input_x.size()[2]
        kshape_j = input_x.size()[3]

        filters = self._initialize_filter_bank((kshape_j, kshape_i))

        filters = np.fft.fftshift(filters)
        images_edge = np.fft.ifft2(np.fft.fft2(input_x) * filters)
        images_edge = np.abs(images_edge)

        maxp = images_edge.argmax(1)
        maxp += 1
        maxp = torch.from_numpy(maxp).unsqueeze(1)

        return maxp.type(torch.FloatTensor)

    def _initialize_filter_bank(self, ksize):
        # Build a filter bank with a Log-Gabor filters
        n_scales = 2
        n_orient = 5

        # The values of the parameters for the log-Gabor
        # filters were defined as suggested in a previous
        # study [2], because they demonstrated good results
        # in texture extraction when log-Gabor filters were
        # used for image descriptions.
        min_wavelen = 3
        scale_factor = 2
        sigma_over_f = 0.65
        sigma_theta = 1

        filters = gf.get_log_gabor_filterbank(
            ksize, n_scales, n_orient, min_wavelen, scale_factor,
            sigma_over_f, sigma_theta)
        return filters
