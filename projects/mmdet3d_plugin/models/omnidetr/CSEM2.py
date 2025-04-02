from mmdet.models import NECKS
import numpy as np
import torch
import torch.nn as nn



# import sys
# sys.path.append('/home/lk/workspase/python/dection/Sparse4D/projects/mmdet3d_plugin/models/rtdetr/custom_nn')
# import custom_nn.custom_models as custom_nn
# import custom_nn.ring_mamba as ring_mamba
from .custom_nn import custom_models as custom_nn
from .custom_nn import DynamicSSM as DynamicSSM
@NECKS.register_module()
class CSEM2(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        output_layer=[14, 17, 20],
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.output_layer = output_layer

        model = [   [-1, 1, "Conv", [256, 1, 1, None, 1, 1, False]], # 5   --->3
                    [-1, 1, "AIFI", [1024, 8]],                           #---->4
                    [-1, 1, "Conv", [256, 1, 1]], # 7                      ---->5

                    [-1, 1, "Upsample", [None, 2, "nearest"]],         #---->6
                    [1, 1, "Conv", [256, 1, 1, None, 1, 1, False]], # 9   #---->7
                    [[-2, -1], 1, "Concat", [1]],
                    [-1, 3, "RepC3", [256]], # 11
                    [-1, 1, "Conv", [256, 1, 1]], # 12                    #---->10

                    [-1, 1, "Upsample", [None, 2, "nearest"]],
                    [0, 1, "Conv", [256, 1, 1, None, 1, 1, False]], # 14
                    [[-2, -1], 1, "Concat", [1]], # cat backbone P4
                    [-1, 3, "RepC3", [256]], # X3 (16), fpn_blocks.1

                    [-1, 1, "Conv", [256, 3, 2]], # 17, downsample_convs.0
                    [[-1, 10], 1, "Concat", [1]], # cat Y4
                    [-1, 3, "RepC3", [256]], # F4 (19), pan_blocks.0

                    [-1, 1, "Conv", [256, 3, 2]], # 20, downsample_convs.1
                    [[-1, 5], 1, "Concat", [1]], # cat Y5
                    [-1, 3, "RepC3", [256]] # F5 (22), pan_blocks.1]
        ]

        self.parse_model(model)
        self.emd_1 = DynamicSSM.DeformableMambaProjEmbed(in_chans=in_channels[0], emb_chans=in_channels[0])
        self.emd_2 = DynamicSSM.DeformableMambaProjEmbed(in_chans=in_channels[1], emb_chans=in_channels[1])
        self.emd_3 = DynamicSSM.DeformableMambaProjEmbed(in_chans=in_channels[2], emb_chans=in_channels[2])

        
    

    def forward(self, x, batch=None):
        y = [self.emd_1(x[0]), self.emd_2(x[1]), self.emd_3(x[2])]
        # y = [x[0], x[1], x[2]]
        outputs_features = []
        for layer_i, m in enumerate(self.model):  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if layer_i == 0:
                x = m(y[-1])  # run
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            outputs_features.append(x) if m.i in self.output_layer else None  # save output features

        return outputs_features
            

    def parse_model(self, model):
        ch = self.in_channels
        number_input_layer = len(self.in_channels)
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(model):
            i += number_input_layer
            n = max(round(n), 1) if n > 1 else n
            if m in {"Conv", "RepC3"}:
                c1, c2 = ch[f], args[0]
                args = [c1, c2, *args[1:]]
                if m in {"RepC3"}:
                    args.insert(2, n)  # number of repeats
                    n = 1
                m = getattr(custom_nn, m) if hasattr(custom_nn, m) else getattr(nn, m)
            elif m in {"AIFI"}:
                args = [ch[f], *args]
                m = getattr(custom_nn, m) if hasattr(custom_nn, m) else getattr(nn, m)
            elif m in {"Concat"}:
                c2 = sum(ch[x] for x in f)
                m = getattr(custom_nn, m) if hasattr(custom_nn, m) else getattr(nn, m)
            else:
                c2 = ch[f]
                m = getattr(custom_nn, m) if hasattr(custom_nn, m) else getattr(nn, m)

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            m_.i , m_.f = i, f
            layers.append(m_)
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            ch.append(c2)
            
        save.sort()
        self.model = nn.Sequential(*layers)
        self.save = save


if __name__ == "__main__":
    neck = MambaHybridEncoder().cuda()
    x = [torch.randn(1, 512, 8, 16), torch.randn(1, 1024, 4, 8), torch.randn(1, 2048, 2, 4)]
    x = [x_.cuda() for x_ in x]
    y = neck(x)
    print(y)
