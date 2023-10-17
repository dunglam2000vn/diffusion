import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, device):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.device = device

    def forward(self, t):
        #return embedding t_size * embed_dim
        
        pe = torch.zeros(t.shape[0], self.embed_dim)
        #print(pe.shape)
        position = torch.arange(0, t.shape[0], dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=self.device).float() * (-math.log(10000.0) / self.embed_dim))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(self.device)


class DownBlock(nn.Module):
    def __init__(self, input_channel, output_channel, embed_dim):
        super(DownBlock, self).__init__()
        #self.input_channel = input_channel
        #self.output_channel = output_channel

        self.embed_layer = nn.Linear(embed_dim, output_channel)

        self.model = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(input_channel, output_channel, 3, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(output_channel),
            nn.Conv2d(output_channel, output_channel, 3, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(output_channel)
        )

    def forward(self, inputs, t):
        #t is tensor t after positional encoding with shape batch_size * embed_dim
        return self.model(inputs) + self.embed_layer(t)[:, :, None, None] # [B, output_channel, H, W] + [B, output_channel, 1, 1]


class UpBlock(nn.Module):
    def __init__(self, input_channel, output_channel, embed_dim):
        super(UpBlock, self).__init__()
        #self.input_channel = input_channel
        #self.output_channel = output_channel

        self.embed_layer = nn.Linear(embed_dim, output_channel)


        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, 2, 2),
            nn.ReLU(),
            #nn.BatchNorm2d(output_channel)
        )
        #assuming in_ch = dim of conv_transpose + extra
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(output_channel),
            nn.Conv2d(output_channel, output_channel, 3, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(output_channel)
        )
        
    def forward(self, inputs, t, residual=None):
        inputs = self.conv_transpose(inputs)
        #print(inputs.shape)
        #print(residual.shape)
        inputs = torch.cat((inputs, residual), dim=1)
        return self.model(inputs) + self.embed_layer(t)[:, :, None, None] # [B, output_channel, H, W] + [B, output_channel, 1, 1]


class UNet(nn.Module):
    def __init__(self, embed_dim, device, image_channels=3, output_channels=3):
        super(UNet, self).__init__()
        self.image_channels = image_channels
        self.output_channels = output_channels
        self.embed_dim = embed_dim
        self.device = device
        down_channels = [64, 128, 256, 512, 1024]
        up_channels = down_channels[::-1]
        self.positional_encoding = nn.Sequential(
            PositionalEncoding(self.embed_dim, self.device),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.image_channels, down_channels[0], 3, padding=1),
            nn.Conv2d(down_channels[0], down_channels[0], 3, padding=1)
        )
        self.downblocks = nn.ModuleList([DownBlock(down_channels[i], down_channels[i+1], self.embed_dim) for i in range(len(down_channels) - 1)])
        self.upblocks = nn.ModuleList([UpBlock(up_channels[i], up_channels[i+1], self.embed_dim) for i in range(len(up_channels) - 1)])
        self.convfinal = nn.Conv2d(up_channels[-1], self.output_channels, 1)

    def forward(self, inputs, t):
        embed = self.positional_encoding(t)
        inputs = self.conv1(inputs)
        residuals = []
        for block in self.downblocks:
            residuals.append(inputs)
            inputs = block(inputs, embed)

        for block in self.upblocks:
            residual = residuals.pop()
            inputs = block(inputs, embed, residual)

        return self.convfinal(inputs)

        
