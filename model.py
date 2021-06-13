#!/usr/bin/env python3

from torch import nn


def Conv(filter_size: int, filters: int, channels: int):
    return nn.Conv2d(channels, filters, filter_size, padding=filter_size//2)


class FSRCNN(nn.Model):
    def __init__(self, *, channels=1, d, s, m, n):
        super(FSRCNN, self).__init__()
        self.feature_extraction = nn.Sequential(
            Conv(5, d, 1),
            nn.PReLu(d),
        )

        self.shrinking = nn.Sequential(
            Conv(1, s, d),
            nn.PReLu(s),
        )

        self.non_linear_mapping = nn.Sequential(*(
            Conv(3, s, s) for _ in range(m)
        ), nn.PReLu(s))

        self.expanding = nn.Sequential(
            Conv(1, d, s),
            nn.PReLu(d)
        )

        self.deconvolution = nn.ConvTranspose2D(d, 1, 9, stride=n)
