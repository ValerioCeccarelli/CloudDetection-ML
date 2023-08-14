import torch
import torch.nn as nn


class DSC(nn.Module):
    '''
    implementation of depthwise separable convolution (same as SeparableConv2d of TensorFlow).
    First, apply a depthwise convolution (which acts on each input channel separately); 
    Second, apply a pointwise convolution (which acts on the output of the depthwise convolution).
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False) -> None:
        super(DSC, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation, groups=in_channels, bias=bias)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SharedConv(nn.Module):
    '''
    Implementtion of shared convolution.
    Each channel of the input is convolved with the same filter.
    '''

    def __init__(self, kernel_size=3, stride=1, padding=0, dilation=1, bias=False) -> None:
        super(SharedConv, self).__init__()

        # Shared filter
        self.shared = nn.Conv2d(1, 1, kernel_size,
                                stride, padding, dilation, groups=1, bias=bias)

    def forward(self, x):
        outputs = []

        # Applay the same filter to each channel
        for i in range(self.in_channels):
            channel_data = x[:, i:i+1, :, :]
            channel_output = self.conv(channel_data)
            outputs.append(channel_output)

        return torch.cat(outputs, dim=1)


class SDC(nn.Module):
    '''
    Implementation of shared dilated convolution residual block.
    '''

    def __init__(self, in_out_channels, kernel_size=3, stride=1, padding=0, dilation=1) -> None:
        super(SDC, self).__init__()

        self.shared_conv = SharedConv(
            kernel_size, stride, padding, bias=False)
        self.dilated_conv = nn.Conv2d(
            in_out_channels, in_out_channels, kernel_size, stride, padding, dilation, bias=True)

    def forward(self, x):
        shared_output = self.shared_conv(x)
        sum = shared_output + x
        dilated_output = self.dilated_conv(sum)
        return dilated_output + x


class SDRB(nn.Module):
    '''
    Shared and dilated convolution residual block layer
    '''

    def __init__(self, in_out_channels, kernel_size=3, stride=1, dilation=2, biased=True) -> None:
        super(SDRB, self).__init__()

        self.shared_dilated_conv_1 = SDC(
            in_out_channels, kernel_size, stride, dilation=dilation)
        self.shared_dilated_conv_2 = SDC(
            in_out_channels, kernel_size, stride, dilation=dilation)

        self.batch_norm_1 = nn.BatchNorm2d(in_out_channels)
        self.batch_norm_2 = nn.BatchNorm2d(in_out_channels)

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        output_1 = self.shared_dilated_conv_1(x)
        output_1 = self.batch_norm_1(output_1)
        output_1 = self.relu_1(output_1)

        output_2 = self.shared_dilated_conv_2(output_1)
        output_2 = self.batch_norm_2(output_2)
        output_2 = self.relu_2(output_2)

        return output_2 + x
