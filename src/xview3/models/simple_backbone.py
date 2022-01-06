import torch

class SimpleBackbone(torch.nn.Module):
    def __init__(self, num_channels):
        super(SimpleBackbone, self).__init__()

        def down_layer(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
                torch.nn.ReLU(inplace=True),
                #torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                #torch.nn.ReLU(inplace=True),
            )

        def up_layer(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
                torch.nn.ReLU(inplace=True),
            )

        self.down1 = down_layer(num_channels, 32) # -> 400x400
        self.down2 = down_layer(32, 64) # -> 200x200
        self.down3 = down_layer(64, 128) # -> 100x100
        self.down4 = down_layer(128, 256) # -> 50x50
        self.down5 = down_layer(256, 512) # -> 25x25
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.up1 = up_layer(512, 256) # -> 50x50
        self.up2 = up_layer(256+256, 128) # -> 100x100
        self.out_channels = 128
        print(self)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        features = self.features(down5)
        up1 = self.up1(features)
        up2 = self.up2(torch.cat([up1, down4], dim=1))
        return {'0': up2}
