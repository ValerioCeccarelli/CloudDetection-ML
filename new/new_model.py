import torch
import torch.nn as nn


class DSC(nn.Module):
    '''
    implementation of depthwise separable convolution (same as SeparableConv2d of TensorFlow).
    First, apply a depthwise convolution (which acts on each input channel separately); 
    Second, apply a pointwise convolution (which acts on the output of the depthwise convolution).
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False) -> None:
        super(DSC, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=1, groups=in_channels, bias=bias)

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

    def __init__(self, in_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False) -> None:
        super(SharedConv, self).__init__()

        self.in_channels = in_channels

        # Shared filter
        self.shared = nn.Conv2d(1, 1, kernel_size,
                                stride, padding, dilation, groups=1, bias=bias)

    def forward(self, x):
        outputs = []

        # Applay the same filter to each channel
        for i in range(self.in_channels):
            channel_data = x[:, i:i+1, :, :]
            channel_output = self.shared(channel_data)
            outputs.append(channel_output)

        return torch.cat(outputs, dim=1)


class SDC(nn.Module):
    '''
    Implementation of shared dilated convolution residual block.
    '''

    def __init__(self, in_out_channels, kernel_size=3, stride=1, padding=1, dilation=1, biased=True) -> None:
        super(SDC, self).__init__()

        self.shared_conv = SharedConv(
            in_out_channels, kernel_size, stride, padding=padding, bias=False)
        self.dilated_conv = nn.Conv2d(
            in_out_channels, in_out_channels, 3, stride, padding=1, bias=biased)

    def forward(self, x):
        shared_output = self.shared_conv(x)
        sum = shared_output + x
        dilated_output = self.dilated_conv(sum)
        return dilated_output + x


class SDRB(nn.Module):
    '''
    Shared and dilated convolution residual block layer
    '''

    def __init__(self, in_out_channels, kernel_size=3, stride=1, dilation=2, biased=True, padding=1) -> None:
        super(SDRB, self).__init__()

        self.shared_dilated_conv_1 = SDC(
            in_out_channels, kernel_size, stride, dilation=dilation, biased=biased, padding=padding)
        self.shared_dilated_conv_2 = SDC(
            in_out_channels, kernel_size, stride, dilation=dilation, biased=biased, padding=padding)

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


class MDSC(nn.Module):

    # TODO: da aggiustare con la kernel list
    def __init__(self, input_dim, output_dim):
        super(MDSC, self).__init__()

        self.separable_conv_1 = nn.Conv2d(
            input_dim, input_dim, 3, padding=1, groups=input_dim, bias=False)
        self.separable_conv_2 = nn.Conv2d(
            input_dim, input_dim, 5, padding=2, groups=input_dim, bias=False)

        self.conv = nn.Conv2d(input_dim*2, output_dim,
                              1, bias=False, padding=0)

    def forward(self, x):
        output_1 = self.separable_conv_1(x)
        output_2 = self.separable_conv_2(x)

        output = torch.cat([output_1, output_2], dim=1)
        output = self.conv(output)

        return output


class CDFM3SF(nn.Module):
    def __init__(self, input_dim, gf_dim=64):
        super(CDFM3SF, self).__init__()

        self.conv1 = nn.Conv2d(
            input_dim[0], gf_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            input_dim[1], gf_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            input_dim[2], gf_dim, kernel_size=3, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(gf_dim)
        self.batch_norm2 = nn.BatchNorm2d(gf_dim)
        self.batch_norm3 = nn.BatchNorm2d(gf_dim)

        self.MDSC1 = MDSC(gf_dim, gf_dim)
        self.MDSC2 = MDSC(gf_dim * 2, gf_dim)
        self.MDSC3 = MDSC(gf_dim * 2, gf_dim)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.MDSC4 = MDSC(gf_dim, gf_dim)

        self.deconv1 = nn.ConvTranspose2d(
            gf_dim * 2, gf_dim, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            gf_dim, gf_dim, kernel_size=9, stride=3, padding=3)
        self.deconv3 = nn.ConvTranspose2d(
            gf_dim, gf_dim, kernel_size=4, stride=2, padding=1)

        self.DSC1 = DSC(gf_dim*4, gf_dim*2, stride=1)
        self.DSC2 = DSC(gf_dim * 2, gf_dim, stride=1)
        self.DSC3 = DSC(gf_dim*2, gf_dim, stride=1)
        self.DSC4 = DSC(gf_dim*2, gf_dim, stride=1)

        self.SDRB1 = SDRB(gf_dim, kernel_size=5, dilation=2, padding=2)
        self.SDRB2 = SDRB(gf_dim, kernel_size=5, dilation=2, padding=2)
        self.SDRB3 = SDRB(gf_dim, kernel_size=7, dilation=3, padding=3)
        self.SDRB4 = SDRB(gf_dim, kernel_size=7, dilation=3, padding=3)
        self.SDRB5 = SDRB(gf_dim, kernel_size=9, dilation=4, padding=4)
        self.SDRB6 = SDRB(gf_dim, kernel_size=9, dilation=4, padding=4)

        self.relu = nn.ReLU(inplace=True)

        self.output1 = nn.Conv2d(gf_dim, 1, kernel_size=1, stride=1)
        self.output2 = nn.Conv2d(gf_dim, 1, kernel_size=1, stride=1)
        self.output3 = nn.Conv2d(gf_dim, 1, kernel_size=1, stride=1)

    def forward(self, input_, input_1, input_2):
        # TODO: dividere in sezioni sta funzione
        # TODO: aggiungere commenti
        mom = self.conv1(input_)
        e10 = self.relu(mom)
        e1 = self.relu(self.batch_norm1(self.MDSC1(e10)))
        e1 = e10 + e1
        p1 = self.pool1(e1)
        mom1 = self.conv2(input_1)
        e20 = self.relu(mom1)
        c120 = torch.cat([p1, e20], dim=1)
        e2 = self.relu(self.batch_norm2(self.MDSC2(c120)))
        e2 = p1 + e20 + e2
        p2 = self.pool2(e2)
        e30 = self.relu(self.conv3(input_2))
        c230 = torch.cat([p2, e30], dim=1)

        e3 = self.relu(self.batch_norm3(self.MDSC3(c230)))
        e3 = p2 + e30 + e3
        p3 = self.pool3(e3)
        e4 = self.relu(self.MDSC4(p3))
        e4 = p3 + e4

        r1 = self.SDRB1(e4)
        r2 = self.SDRB2(r1)
        r3 = self.SDRB3(r2)
        r4 = self.SDRB4(r3)
        r5 = self.SDRB5(r4)
        r6 = self.SDRB6(r5)

        d1 = torch.cat([e4, r2, r4, r6], dim=1)
        d1 = self.relu(self.DSC1(d1))
        d1 = self.deconv1(d1)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.relu(self.DSC2(d1))
        output3 = self.output3(d1)  # rinominare sta funzione in modo sensato

        d2 = self.deconv2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.relu(self.DSC3(d2))
        output2 = self.output2(d2)

        d3 = self.deconv3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.relu(self.DSC4(d3))
        output1 = self.output1(d3)

        return output1, output2, output3
