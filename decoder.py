import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import Conv2dReLU, SCSEModule
from base_model import Model


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)

        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.block(x)
        x = self.attention2(x)
        return x


class CenterBlock(DecoderBlock):

    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            attention_type=None,
            conv_coarse_channels=(32, 16)
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)

        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))
        self.final_conv_combined = nn.Conv2d(out_channels[4], 1, kernel_size=(1, 1))
        self.final_conv_coarse = nn.Sequential(
            Conv2dReLU(out_channels[2], 
                       conv_coarse_channels[0],
                       kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(conv_coarse_channels[0], 
                       conv_coarse_channels[1], 
                       kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            nn.Conv2d(conv_coarse_channels[1], 
                       final_channels, 
                       kernel_size=(1, 1))
        )
        self.final_conv_coarse_combined = nn.Sequential(
            Conv2dReLU(out_channels[2], 
                       conv_coarse_channels[0],
                       kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(conv_coarse_channels[0], 
                       conv_coarse_channels[1], 
                       kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            nn.Conv2d(conv_coarse_channels[1], 
                       1, 
                       kernel_size=(1, 1))
        )
        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x_1 = self.layer1([encoder_head, skips[0]])
        x_2 = self.layer2([x_1, skips[1]])
        x_3 = self.layer3([x_2, skips[2]])
        x_4 = self.layer4([x_3, skips[3]])
        x_5 = self.layer5([x_4, None])
        x_final = self.final_conv(x_5)
        x_final_combined = self.final_conv_combined(x_5)
        x_coarse = self.final_conv_coarse(x_3)
        x_coarse_combined = self.final_conv_coarse_combined(x_3)
        #x_classification =  #a
        output = {'logits':x_final, 
                  'logits_coarse':x_coarse, 
                  'logits_combined':x_final_combined,
                  'logits_coarse_combined':x_coarse_combined}
        return output