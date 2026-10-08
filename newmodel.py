from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch



class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.se = SEBlock(out_ch)


    def forward(self, x):

        x = self.conv(x)
        x = self.se(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
              


class AMCA(nn.Module):
    def __init__(self, channels, reduction=4):
        super(AMCA, self).__init__()

        hidden = max(channels // reduction, 32)

        # 3×3 local
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False)
        )

        # 3×3 dilation=2
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False)
        )

        # 5×5 large receptive field
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False)
        )

        # scale selector
        self.scale_selector = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1)
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)

        multi_scale = torch.cat([f1, f2, f3], dim=1)

        # 每个空间位置自适应决定三个尺度的贡献
        weights = self.scale_selector(multi_scale)
        weights = F.softmax(weights, dim=1)

        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]
        w3 = weights[:, 2:3]

        out = (
            w1 * f1 +
            w2 * f2 +
            w3 * f3
        )

        out = self.out_proj(out)

        return x + out




class PARM(nn.Module):


    def __init__(self, channels, reduction=4):
        super(PARM, self).__init__()

        # ==================================================
        # 1. PS-oriented representation
        # ==================================================

        reduced_channels = channels // reduction

        self.ps_proj = nn.Sequential(
            nn.Conv2d(
                channels,
                reduced_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        # ==================================================
        # 2. FH-oriented representation
        # ==================================================

        self.fh_proj = nn.Sequential(
            nn.Conv2d(
                channels,
                reduced_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        # ==================================================
        # 3. Pairwise relation encoder
     
        # ==================================================

        self.relation = nn.Sequential(
            nn.Conv2d(
                reduced_channels * 3,
                reduced_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                reduced_channels,
                reduced_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        # ==================================================
        # 4. PS-directed relation gate
        # ==================================================

        self.ps_gate = nn.Sequential(
            nn.Conv2d(
                reduced_channels,
                reduced_channels,
                kernel_size=1,
                bias=False
            ),
            nn.Sigmoid()
        )

        # ==================================================
        # 5. FH-directed relation gate
        # ==================================================

        self.fh_gate = nn.Sequential(
            nn.Conv2d(
                reduced_channels,
                reduced_channels,
                kernel_size=1,
                bias=False
            ),
            nn.Sigmoid()
        )

        # ==================================================
        # 6. Reconstruct PS feature
        # ==================================================

        self.ps_refine = nn.Sequential(
            nn.Conv2d(
                reduced_channels * 2,
                reduced_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        # ==================================================
        # 7. Reconstruct FH feature
        # ==================================================

        self.fh_refine = nn.Sequential(
            nn.Conv2d(
                reduced_channels * 2,
                reduced_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        # ==================================================
        # 8. Final fusion
        #
        # ==================================================

        self.out_proj = nn.Sequential(
            nn.Conv2d(
                reduced_channels * 3,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # ==================================================
        # 9. Residual scaling
        # ==================================================

        self.gamma = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, x):

        # ==================================================
        # Step 1. PS / FH oriented representations
        # ==================================================

        f_ps = self.ps_proj(x)

        f_fh = self.fh_proj(x)

        # ==================================================
        # Step 2. Pairwise relation construction
        # ==================================================

        difference = torch.abs(
            f_ps - f_fh
        )

        correlation = (
            f_ps * f_fh
        )

        joint = (
            f_ps + f_fh
        )

        relation_input = torch.cat(
            [
                difference,
                correlation,
                joint
            ],
            dim=1
        )

        # ==================================================
        # Step 3. Explicit pairwise relation representation
        # ==================================================

        relation = self.relation(
            relation_input
        )

        # ==================================================
        # Step 4. Asymmetric relation gates
        # ==================================================

        gate_ps = self.ps_gate(
            relation
        )

        gate_fh = self.fh_gate(
            relation
        )

        # ==================================================
        # Step 5. PS refinement
        # ==================================================

        ps_relation = (
            gate_ps * relation
        )

        f_ps_refined = self.ps_refine(
            torch.cat(
                [
                    f_ps,
                    ps_relation
                ],
                dim=1
            )
        )

        f_ps_refined = (
            f_ps +
            f_ps_refined
        )

        # ==================================================
        # Step 6. FH refinement
        # ==================================================

        fh_relation = (
            gate_fh * relation
        )

        f_fh_refined = self.fh_refine(
            torch.cat(
                [
                    f_fh,
                    fh_relation
                ],
                dim=1
            )
        )

        f_fh_refined = (
            f_fh +
            f_fh_refined
        )

        # ==================================================
        # Step 7. Joint reconstruction
        # ==================================================

        fused = self.out_proj(
            torch.cat(
                [
                    f_ps_refined,
                    f_fh_refined,
                    relation
                ],
                dim=1
            )
        )

        # ==================================================
        # Step 8. Residual output
        # ==================================================

        out = (
            x +
            self.gamma * fused
        )

        return out


        
        



class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=3):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        
        
        self.AMCA = AMCA(filters[4])
        self.PARM = PARM(
        channels=filters[4],
        reduction=4
        )

        

        
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)


       # self.active = torch.nn.Sigmoid()

    def forward(self, x,return_aux=False):

        e1 = self.Conv1(x)


        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)


        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)


        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)


        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        e5 = self.AMCA(e5)
        e5 = self.PARM(e5)


        d5 = self.Up5(e5)
        
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)


        d4 = self.Up4(d5)
        
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)


        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)


        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)


        out = self.Conv(d2)
        
    

        return out


