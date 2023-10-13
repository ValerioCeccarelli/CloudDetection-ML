import torch.nn as nn
import torch
from torch.optim import Adam, Optimizer


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation, groups=in_channels, bias=bias)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


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


# m1 = SeparableConv2d(3, 64, kernel_size=3, stride=1,
#                      dilation=2, padding="same")
# m2 = DSC(3, 64, kernel_size=3, stride=1, dilation=2, padding="same")


# # generate fake data 4x32x32x3
# x = torch.rand(4, 3, 32, 32)

# print(x.shape)

# y1 = m1(x)
# y2 = m2(x)

# print(y1.shape)
# print(y2.shape)

# p = nn.Conv3d(2, 2, kernel_size=[2, 2, 1], groups=2, bias=True)
# print(p.weight.shape)
# # print weight count
# print(p.weight.numel())


class CustomCNN1(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(CustomCNN1, self).__init__()

        # Create a single convolutional filter
        self.conv = nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=kernel_size, bias=False)

        # Keep track of the number of channels
        self.in_channels = in_channels

    def forward(self, x):
        # List to hold the output from each channel
        outputs = []

        # Apply the same filter to each channel independently
        for i in range(self.in_channels):
            channel_data = x[:, i:i+1, :, :]
            channel_output = self.conv(channel_data)
            outputs.append(channel_output)

        # Stack the results along the channel dimension
        return torch.cat(outputs, dim=1)


class CustomCNN(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(CustomCNN, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=kernel_size, groups=in_channels, bias=False)

    def forward(self, x):
        return self.conv(x)


model = CustomCNN1(3)

x = torch.rand(4, 3, 32, 32, requires_grad=True)

optimizer = Adam(model.parameters(), lr=0.00025, betas=(0.5, 0.9))

optimizer.zero_grad()

y = model(x)

loss: torch.Tensor = y.sum()

loss.backward()

print(loss)
