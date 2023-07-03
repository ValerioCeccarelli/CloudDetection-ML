import torch
import torch.nn as nn


class MDSC(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_list=[3, 5], stride=1, padding="same", scale=1, biased=True):
        super(MDSC, self).__init__()

        self.depthwise_convs = nn.ModuleList()
        for kernel_size in kernel_list:
            self.depthwise_convs.append(nn.Conv2d(
                input_dim, input_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=input_dim))

        self.output_dim = output_dim
        self.scale = scale
        self.biased = biased
        self.conv = nn.Conv2d(input_dim * len(kernel_list), output_dim,
                              kernel_size=1, stride=1, padding=padding, bias=biased)

    def forward(self, input_):
        depthoutput_list = []
        for depth_conv in self.depthwise_convs:
            depthoutput_list.append(depth_conv(input_))
        output = torch.cat(depthoutput_list, dim=1)
        output = self.conv(output)
        return output


class DSC(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, dilation=2, padding="same", biased=True):
        super(DSC, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.biased = biased

        self.fix_w_size = self.dilation * 2 - 1
        self.conv1 = nn.Conv3d(input_dim, 1, kernel_size=[
                               self.fix_w_size, self.fix_w_size, 1], stride=stride, padding=padding, bias=biased)
        self.conv2 = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=biased)

    def forward(self, input_):
        eo = torch.unsqueeze(input_, -1)
        o = self.conv1(eo)
        o = eo + o
        o = torch.squeeze(o, -1)
        o = self.conv2(o)
        return o


class SDRB(nn.Module):
    def __init__(self, input_dim, kernel_size=3, stride=1, dilation=2, training=False, biased=True):
        super(SDRB, self).__init__()

        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.training = training
        self.biased = biased

        self.sconv1 = DSC(input_dim, input_dim, kernel_size,
                          stride, dilation, biased=biased)
        self.batch_norm1 = nn.BatchNorm2d(input_dim)
        self.relu1 = nn.ReLU()
        self.sconv2 = DSC(input_dim, input_dim, kernel_size,
                          stride, dilation, biased=biased)
        self.batch_norm2 = nn.BatchNorm2d(input_dim)

    def forward(self, input_):
        sconv1 = self.sconv1(input_)
        sconv1 = self.batch_norm1(sconv1)
        sconv1 = self.relu1(sconv1)
        sconv2 = self.sconv2(sconv1)
        sconv2 = self.batch_norm2(sconv2)
        output = self.relu1(sconv2 + input_)
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

        self.MDSC1 = MDSC(gf_dim, gf_dim, stride=1)
        self.MDSC2 = MDSC(gf_dim * 2, gf_dim, stride=1)
        self.MDSC3 = MDSC(gf_dim * 2, gf_dim, stride=1)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.MDSC4 = MDSC(gf_dim, gf_dim, stride=1)

        self.pad = nn.ZeroPad2d(1)

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

        self.SDRB1 = SDRB(gf_dim, kernel_size=3, stride=1, dilation=2)
        self.SDRB2 = SDRB(gf_dim, kernel_size=3, stride=1, dilation=2)
        self.SDRB3 = SDRB(gf_dim, kernel_size=3, stride=1, dilation=3)
        self.SDRB4 = SDRB(gf_dim, kernel_size=3, stride=1, dilation=3)
        self.SDRB5 = SDRB(gf_dim, kernel_size=3, stride=1, dilation=4)
        self.SDRB6 = SDRB(gf_dim, kernel_size=3, stride=1, dilation=4)

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
        mom2 = self.MDSC3(c230)
        mom3 = self.batch_norm3(mom2)
        e3 = self.relu(mom3)
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
        output3 = self.output3(d1)

        d2 = self.deconv2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.relu(self.DSC3(d2))
        output2 = self.output2(d2)

        d3 = self.deconv3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.relu(self.DSC4(d3))
        output1 = self.output1(d3)

        return output1, output2, output3
