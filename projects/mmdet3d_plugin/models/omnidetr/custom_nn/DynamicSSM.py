import torch
import torch.nn as nn
import torchvision
from .ss2d import SS2D

class DeformableMambaProjEmbed(nn.Module):  # Deformable Patch Embedding
    """ feature map to Projected Embedding
    """
    def __init__(self, in_chans=512, emb_chans=128, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.proj = nn.Conv2d(in_chans, emb_chans, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        # --- deformable offset and modulator
        self.offset_conv = nn.Conv2d(in_chans, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_chans, 1 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.norm = nn.BatchNorm2d(emb_chans)
        self.act = nn.GELU()
        self.conv2d_1 = nn.Conv2d(in_chans, emb_chans, kernel_size=1, stride=1, padding=0)
        self.conv2d_2 = nn.Conv2d(in_chans, emb_chans, kernel_size=1, stride=1, padding=0)
        self.ss2d = SS2D(d_model = emb_chans)

    # def deform_proj(self, x):
    #     # h, w = x.shape[2:]
    #     max_offset = min(x.shape[-2], x.shape[-1]) // 4
    #     offset = self.offset_conv(x).clamp(-max_offset, max_offset)
    #     modulator = 2. * torch.sigmoid(self.modulator_conv(x))
    #     dx = torchvision.ops.deform_conv2d(input=x,
    #                                       offset=offset,
    #                                       weight=self.proj.weight,
    #                                       bias=self.proj.bias,
    #                                       padding=self.padding,
    #                                       mask=modulator,
    #                                       stride=self.stride,
    #                                       )  # deformable conv
    #     rx = self.act(self.conv2d_1(x))
    #     ss2d_out = self.ss2d(rx)
    #     x   = rx + ss2d_out
         

    #     return x
    

    def deform_proj(self, x):
        # h, w = x.shape[2:]
        max_offset = min(x.shape[-2], x.shape[-1]) // 4
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        dx = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )  # deformable conv
        rx = self.act(self.conv2d_1(x))
        ss2d_out = self.ss2d(rx)
        x   = rx + ss2d_out
         

        return x

    def forward(self, x):
        x = self.deform_proj(x)
        x = self.act(self.norm(x))
        return x

if __name__ == '__main__':
    model = DeformableMambaProjEmbed(in_chans=512, emb_chans=512, kernel_size=3, stride=1, padding=1).cuda()
    x = torch.randn(1, 512, 224, 224)
    x = x.cuda()
    y = model(x)
    print(y.shape, x.shape)