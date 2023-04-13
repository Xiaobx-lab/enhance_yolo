from torch import nn, sigmoid, softmax
import torch
# from utils.utils_control import conv_nd, zero_module



class AFS(nn.Module):
    def __init__(self, in_channel, out_channel, in_channel2 = None, out_channel2 = None):
        super(AFS, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        if in_channel2 == None and out_channel2 == None:
            in_channel2 = in_channel
            out_channel2 = out_channel
        self.cat_dims = out_channel+out_channel2

        self.in_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=self.kernel_size, stride=1,
                      padding=self.padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.in_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel2, out_channels=out_channel2, kernel_size=self.kernel_size, stride=1,
                      padding=self.padding),
            nn.BatchNorm2d(out_channel2),
            nn.ReLU()
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels=out_channel+out_channel2, out_channels= out_channel + out_channel2, kernel_size=self.kernel_size, stride=1,
                   padding=self.padding),
            nn.BatchNorm2d(out_channel+out_channel2),
            nn.ReLU()
        )

        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels=out_channel+out_channel2, out_channels=out_channel+out_channel2, kernel_size=self.kernel_size, stride=1,
                   padding=self.padding),
            nn.ReLU(),
            # out_channels=in_channel if in_channel > in_channel2  else in_channel2
            nn.Conv2d(in_channels=out_channel+out_channel2, out_channels=out_channel+out_channel2, kernel_size=self.kernel_size,stride=1, padding=self.padding ),
            nn.BatchNorm2d(out_channel+out_channel2),  #之后可以试一下
            nn.Sigmoid(),
            # CALayer(in_channel if in_channel > in_channel2 else in_channel2),
            # nn.Softmax(dim = 1)
        )

        self.conv_low = nn.Sequential(
            # in_channel if in_channel > in_channel2  else in_channel2, out_channels=in_channel if in_channel > in_channel2  else in_channel2
            nn.Conv2d(in_channels = out_channel+out_channel2, out_channels=out_channel+out_channel2, kernel_size=3, stride=1, padding=self.padding),
            nn.ReLU(),
            # in_channels = in_channel if in_channel > in_channel2  else in_channel2, out_channels= in_channel if in_channel < in_channel2  else in_channel2
            nn.Conv2d(in_channels=out_channel+out_channel2, out_channels=out_channel+out_channel2, kernel_size=3, stride=1, padding=self.padding),
            nn.BatchNorm2d(out_channel+out_channel2), 
            nn.Sigmoid(),
            # CALayer(in_channel if in_channel < in_channel2 else in_channel2),
            # nn.Softmax(dim = 1)
        )

        self.zero_convs_high = self.make_zero_conv(in_channel if in_channel > in_channel2  else in_channel2)
        self.zero_convs_low = self.make_zero_conv(in_channel if in_channel < in_channel2  else in_channel2)
    
    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()
        return module
    
    def make_zero_conv(self, channels):
        return self.zero_module(nn.Conv2d(self.cat_dims, channels, 1, padding=0))



    def forward(self, feature1, feature2):
        x1 = self.in_conv_1(feature1)
        x2 = self.in_conv_2(feature2)
        x = torch.cat([x1,x2],1)
        x = self.conv_cat(x)
        high_feature = self.conv_high(x)
        low_feature = self.conv_low(high_feature)
        high_feature = self.zero_convs_high(high_feature)
        low_feature = self.zero_convs_low(low_feature)

        return high_feature, low_feature

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

if __name__ == "__main__":
    # from utils.utils_control import conv_nd, zero_module

    yolo_feature = torch.Tensor(2,75,52,52)
    unet_feature = torch.Tensor(2,512,52,52)
    net = AFS(75,75,in_channel2=512, out_channel2=512)
    high_map , low_map = net(yolo_feature, unet_feature)
    print(high_map.shape, low_map.shape)