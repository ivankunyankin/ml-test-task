import torch.nn as nn


class QuartzNet(nn.Module):

    def __init__(self, in_channels=64, out_channels=11):

        super(QuartzNet, self).__init__()

        block_channels = [64, 128, 128, 128]
        block_k = [17, 19, 25]

        self.C1 = nn.Sequential(nn.Conv1d(in_channels, 64, kernel_size=17, padding=8, bias=False),
                                nn.BatchNorm1d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(),
                                nn.Dropout(p=0.2, inplace=False))

        self.B = nn.ModuleList([])

        for i in range(3):
            pad = block_k[i] // 2
            self.B.append(JasperBlock(block_channels[i], block_channels[i+1], block_k[i], pad))

        self.C2 = nn.Conv1d(128, out_channels, kernel_size=1)

    def forward(self, x):

        x = self.C1(x)

        for block in self.B:
            x = block(x)

        x = self.C2(x)

        return x


class JasperBlock(nn.Module):

    def __init__(self, in_channels, out_channels, k, padding):

        super(JasperBlock, self).__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, out_channels, k, padding),
            ConvBlock(out_channels, out_channels, k, padding),
            ConvBlock(out_channels, out_channels, k, padding))

        self.last = nn.ModuleList([
            nn.Conv1d(out_channels, out_channels, kernel_size=k, stride=[1], padding=(padding,), dilation=[1], groups=out_channels, bias=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        ])

        self.res = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,), stride=[1], bias=False),
                                 nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):

        y = self.res(x)
        x = self.blocks(x)

        for idx, layer in enumerate(self.last):
            x = layer(x)
            if idx == 2:
                x += y
                x = self.relu(x)
                x = self.dropout(x)

        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, k, padding):

        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=k, stride=[1], padding=(padding,), dilation=[1], groups=in_channels, bias=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False)
        )

    def forward(self, x):

        return self.layers(x)
