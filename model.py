#!/usr/bin/env python3

from torch import nn


def Conv(filter_size: int, filters: int, channels: int):
    return nn.Conv2d(channels, filters, filter_size, padding=filter_size//2)


class FSRCNN(nn.Module):
    def __init__(self, *, channels=1, d, s, m, scale):
        super(FSRCNN, self).__init__()
        self.feature_extraction = nn.Sequential(
            Conv(5, d, 1),
            nn.PReLU(),
        )

        self.shrinking = nn.Sequential(
            Conv(1, s, d),
            nn.PReLU(),
        )

        self.non_linear_mapping = nn.Sequential(*(
            Conv(3, s, s) for _ in range(m)
        ), nn.PReLU())

        self.expanding = nn.Sequential(
            Conv(1, d, s),
            nn.PReLU()
        )

        self.deconvolution = nn.ConvTranspose2d(
            d, channels, 9, stride=scale, padding=9//2, output_padding=scale-1)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.non_linear_mapping(x)
        x = self.expanding(x)
        return self.deconvolution(x)
