import torch
import torch.nn as nn
import torch.nn.functional as F
import CBAM as cbam
import SAL as sal
import torch.optim as optim
import Loss as ls
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNet, self).__init__()

        # Encoder
        self.enc0 = ResidualBlock(in_channels, 16)
        self.enc1 = ResidualBlock(16, 32, stride=2)
        self.enc2 = ResidualBlock(32, 64, stride=2)
        self.enc3 = ResidualBlock(64, 128, stride=2)
        self.enc4 = ResidualBlock(128, 256, stride=2)

        # Decoder
        self.dec4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(256, 128)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)
        self.dec0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decm1 = ResidualBlock(64, 32)
        self.decm2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc0 = self.enc0(x)
        cbam0 = cbam.CBAM(16)(enc0)
        #SAL
        sal0 = sal.SAL(dim=16, input_resolution=(128, 128), num_heads=8)
        input = sal.to_3d(cbam0)
        output = sal0(input)
        sal00 = sal.to_4d(output, 128, 128)

        enc1 = self.enc1(enc0)
        cbam1 = cbam.CBAM(32)(enc1)
        #SAL
        sal1 = sal.SAL(dim=32, input_resolution=(64, 64), num_heads=8)
        input = sal.to_3d(cbam1)
        output = sal1(input)
        sal11 = sal.to_4d(output, 64, 64)

        enc2 = self.enc2(enc1)
        cbam2 = cbam.CBAM(64)(enc2)
        #SAL
        sal2 = sal.SAL(dim=64, input_resolution=(32, 32), num_heads=8)
        input = sal.to_3d(cbam2)
        output = sal2(input)
        sal22 = sal.to_4d(output, 32, 32)

        enc3 = self.enc3(enc2)
        cbam3 = cbam.CBAM(128)(enc3)
        #SAL
        sal3 = sal.SAL(dim=128, input_resolution=(16, 16), num_heads=8)
        input = sal.to_3d(cbam3)
        output = sal3(input)
        sal33 = sal.to_4d(output, 16, 16)

        enc4 = self.enc4(enc3)
        cbam4 = cbam.CBAM(256)(enc4)
        #SAL
        sal4 = sal.SAL(dim=256, input_resolution=(8, 8), num_heads=8)
        input = sal.to_3d(cbam4)
        output = sal4(input)
        sal44 = sal.to_4d(output, 8, 8)

        # Decoder
        dec4 = self.dec4(sal44)
        dec3 = self.dec3(torch.cat([dec4, sal33], dim=1))
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(torch.cat([dec2, sal22], dim=1))
        dec0 = self.dec0(dec1)
        decm1 = self.decm1(torch.cat([dec0, sal11], dim=1))
        decm2 = self.decm2(decm1)
        final = self.final(torch.cat([decm2, sal00], dim=1))

        return final

# sample training function
def training(model,train_loader):
    bce_loss = nn.BCELoss()
    dice_loss = ls.DiceLoss()
    focal_loss = ls.FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, momentum=0.5)

    for epoch in range(100):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            bce = bce_loss(output, target)
            dice = dice_loss(output, target)
            focal = focal_loss(output, target)
            total_loss = bce + dice + focal
            total_loss.backward()
            optimizer.step()

    print("Training complete!")

# Example usage
if __name__ == "__main__":
    model = ResUNet(in_channels=1, out_channels=1)
    x = torch.randn(4, 1, 128, 128)
    output = model(x)
    print(output.shape)