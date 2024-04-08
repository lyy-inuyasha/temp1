import torch
import torch.nn as nn

class BraNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(BraNet, self).__init__()
        self.ec0_0 = self.black_block(self.in_channel, 8,kernel_size=[3,3,3])
        self.ec0_1 = self.yellow_block(8,8,[3,3,3])

        self.enc1_1 = self.yellow_block(8, 16, [3,3,3])
        self.enc1_2 = self.yellow_block(16,16,[3,3,3])

        self.enc2_1 = self.yellow_block(16, 32, [3, 3, 3])
        self.enc2_2 = self.yellow_block(32, 32, [3, 3, 3])

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3_1 = self.yellow_block(32, 64, [3, 3, 3])
        self.enc3_2 = self.yellow_block(64, 64, [3, 3, 3])
        self.pool3 = nn.MaxPool3d(2)

        self.enc4_1 = self.yellow_block(64,128,[1,2,1])  # kernal_size 从[7,9,7]变为[1,2,1]
        self.enc4_2 = self.orange_block(128,256,[1,1,1])
        self.enc4_3 = self.orange_block(256,64,[1,1,1])
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Conv3d(64, n_classes, 1)


    def black_block(self, in_channels, out_channels,kernel_size):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.LeakyReLU())
        return layer

    def yellow_block(self, in_channels, out_channels,kernel_size):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.GroupNorm(2, out_channels) )
        return layer

    def orange_block(self, in_channels, out_channels, kernel_size):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5))
        return layer

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        bs = x.shape[0]
        x = x.float()
        e0 = self.ec0_0(x)
        syn0 = self.ec0_1(e0)
        e0 = self.pool0(e0+syn0)

        e1 = self.enc1_1(e0)
        syn1 = self.enc1_2(e1)
        e1 = self.pool1(e1+syn1)

        e2 = self.enc2_1(e1)
        syn2 = self.enc2_2(e2)
        e2 = self.pool2(e2 + syn2)

        e3 = self.enc3_1(e2)
        syn3 = self.enc3_2(e3)
        e3 = self.pool2(e3 + syn3)

        e4 = self.enc4_1(e3)
        e4 = self.enc4_2(e4)
        e4 = self.enc4_3(e4)
        e4 = self.classifier(self.gap(e4))

        return e4.reshape(bs, -1)

if __name__ == '__main__':
    model = BraNet(2,1)
    x = torch.randn(2, 2, 26,33, 23)
    y = model(x)
    print(y.shape)
