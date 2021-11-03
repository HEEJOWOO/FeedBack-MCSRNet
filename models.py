from torch import nn
import torch
import torch.nn.functional as F
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
    
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
class RDB(nn.Module):

    def __init__(self, in_channels,wn):
        super(RDB, self).__init__()

        self.distilled_channels = 16
        self.remaining_channels = 48
        gc = 64 
        fc = 48

        self.layer1 = nn.Sequential(wn(nn.Conv2d(in_channels + 0 * gc, gc, 1)),nn.ReLU(True), wn(nn.Conv2d(gc, gc, 3, padding=1, bias=True)))
        self.layer2 = nn.Sequential(wn(nn.Conv2d(in_channels + 1 * gc, gc, 1)),nn.ReLU(True), wn(nn.Conv2d(gc, gc, 3, padding=1, bias=True)))
        self.layer3 = nn.Sequential(wn(nn.Conv2d(in_channels + 2 * gc, gc, 1)),nn.ReLU(True), wn(nn.Conv2d(gc, gc, 3, padding=1, bias=True)))
        self.layer4 = nn.Sequential(wn(nn.Conv2d(in_channels + 3 * gc, gc, 1)),nn.ReLU(True), wn(nn.Conv2d(gc, gc, 3, padding=1, bias=True)))
        self.layer5 = nn.Sequential(wn(nn.Conv2d(in_channels + 4 * gc, gc, 1)),nn.ReLU(True), wn(nn.Conv2d(gc, gc, 3, padding=1, bias=True)))
        self.layer6 = nn.Sequential(wn(nn.Conv2d(in_channels + 5 * gc, gc, 1)),nn.ReLU(True), wn(nn.Conv2d(gc, gc, 3, padding=1, bias=True)))
        self.layer7 = nn.Sequential(wn(nn.Conv2d(in_channels + 6 * gc, gc, 1)),nn.ReLU(True), wn(nn.Conv2d(gc, gc, 3, padding=1, bias=True)))
        self.layer8 = nn.Sequential(wn(nn.Conv2d(in_channels + 7 * gc, gc, 1)),nn.ReLU(True), wn(nn.Conv2d(gc, gc, 3, padding=1, bias=True)))

        self.idle_layer1 = nn.Sequential(wn(nn.Conv2d(fc,gc,3,1,1)),nn.ReLU(True),wn(nn.Conv2d(gc,fc,3,1,1)))
        self.idle_layer2 = nn.Sequential(wn(nn.Conv2d(fc,gc,3,1,1)),nn.ReLU(True),wn(nn.Conv2d(gc,fc,3,1,1)))
        self.idle_layer3 = nn.Sequential(wn(nn.Conv2d(fc,gc,3,1,1)),nn.ReLU(True),wn(nn.Conv2d(gc,fc,3,1,1)))
        self.idle_layer4 = nn.Sequential(wn(nn.Conv2d(fc,gc,3,1,1)),nn.ReLU(True),wn(nn.Conv2d(gc,fc,3,1,1)))
        self.idle_layer5 = nn.Sequential(wn(nn.Conv2d(fc,gc,3,1,1)),nn.ReLU(True),wn(nn.Conv2d(gc,fc,3,1,1)))
        self.idle_layer6 = nn.Sequential(wn(nn.Conv2d(fc,gc,3,1,1)),nn.ReLU(True),wn(nn.Conv2d(gc,fc,3,1,1)))
        self.idle_layer7 = nn.Sequential(wn(nn.Conv2d(fc,gc,3,1,1)),nn.ReLU(True),wn(nn.Conv2d(gc,fc,3,1,1)))
        self.idle_layer8 = nn.Sequential(wn(nn.Conv2d(fc,gc,3,1,1)),nn.ReLU(True),wn(nn.Conv2d(gc,32,3,1,1)))
        
        self.remaining_layer1_sub = wn(nn.Conv2d(fc,fc,3,1,1))
        self.remaining_layer2_sub = wn(nn.Conv2d(fc,fc,3,1,1))
        self.remaining_layer3_sub = wn(nn.Conv2d(fc,fc,3,1,1))
        self.remaining_layer4_sub = wn(nn.Conv2d(fc,fc,3,1,1))
        self.remaining_layer5_sub = wn(nn.Conv2d(fc,fc,3,1,1))
        self.remaining_layer6_sub = wn(nn.Conv2d(fc,fc,3,1,1))
        self.remaining_layer7_sub = wn(nn.Conv2d(fc,fc,3,1,1))
        
        self.residual_layer1_sub = wn(nn.Conv2d(gc,gc,3,1,1))
        self.residual_layer2_sub = wn(nn.Conv2d(gc,gc,3,1,1))
        self.residual_layer3_sub = wn(nn.Conv2d(gc,gc,3,1,1))
        self.residual_layer4_sub = wn(nn.Conv2d(gc,gc,3,1,1))
        self.residual_layer5_sub = wn(nn.Conv2d(gc,gc,3,1,1))
        self.residual_layer6_sub = wn(nn.Conv2d(gc,gc,3,1,1))
        self.residual_layer7_sub = wn(nn.Conv2d(gc,gc,3,1,1))
        self.residual_layer8_sub = wn(nn.Conv2d(gc,gc,3,1,1))

        self.layer1_sub = wn(nn.Conv2d(16+64,32,3,1,1))
        self.layer2_sub = wn(nn.Conv2d(16+64,32,3,1,1))
        self.layer3_sub = wn(nn.Conv2d(16+64,32,3,1,1))
        self.layer4_sub = wn(nn.Conv2d(16+64,32,3,1,1))
        self.layer5_sub = wn(nn.Conv2d(16+64,32,3,1,1))
        self.layer6_sub = wn(nn.Conv2d(16+64,32,3,1,1))
        self.layer7_sub = wn(nn.Conv2d(16+64,32,3,1,1))
        
        self.lff = wn(nn.Conv2d(32*8, gc, kernel_size=1))
        self.lff_split = wn(nn.Conv2d(gc*2, gc, kernel_size=1))
        #contrast channel attention
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            wn(nn.Conv2d(64, 64 // 16, 1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(64 // 16, 64, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        x_residual = x

        layer1 = self.layer1(x)+self.residual_layer1_sub(x)
        distilled_c1, remaining_c1 = torch.split(layer1, (self.distilled_channels, self.remaining_channels), dim=1)
        idle_layer1 = self.idle_layer1(remaining_c1)+self.remaining_layer1_sub(remaining_c1)
        idle_concat1 = torch.cat([distilled_c1,idle_layer1],1)
        layer1_sub = self.layer1_sub(torch.cat([layer1,distilled_c1],1))

        layer2 = self.layer2(torch.cat((x, idle_concat1), 1))+self.residual_layer2_sub(idle_concat1)
        distilled_c2, remaining_c2 = torch.split(layer2, (self.distilled_channels, self.remaining_channels), dim=1)
        idle_layer2 = self.idle_layer2(remaining_c2)+self.remaining_layer2_sub(remaining_c2)
        idle_concat2 = torch.cat([distilled_c2,idle_layer2],1)
        layer2_sub = self.layer2_sub(torch.cat([layer2,distilled_c2],1))
        
        layer3 = self.layer3(torch.cat((x, idle_concat1,idle_concat2), 1))+self.residual_layer3_sub(idle_concat2)
        distilled_c3, remaining_c3 = torch.split(layer3, (self.distilled_channels, self.remaining_channels), dim=1)
        idle_layer3 = self.idle_layer3(remaining_c3)+self.remaining_layer3_sub(remaining_c3)
        idle_concat3 = torch.cat([distilled_c3,idle_layer3],1)
        layer3_sub = self.layer3_sub(torch.cat([layer3,distilled_c3],1))
        
        layer4 = self.layer4(torch.cat((x, idle_concat1,idle_concat2,idle_concat3), 1))+self.residual_layer4_sub(idle_concat3)
        distilled_c4, remaining_c4 = torch.split(layer4, (self.distilled_channels, self.remaining_channels), dim=1)
        idle_layer4 = self.idle_layer4(remaining_c4)+self.remaining_layer4_sub(remaining_c4)
        idle_concat4 = torch.cat([distilled_c4,idle_layer4],1)
        layer4_sub = self.layer4_sub(torch.cat([layer4,distilled_c4],1))
        
        layer5 = self.layer5(torch.cat((x, idle_concat1,idle_concat2,idle_concat3,idle_concat4), 1))+self.residual_layer5_sub(idle_concat4)
        distilled_c5, remaining_c5 = torch.split(layer5, (self.distilled_channels, self.remaining_channels), dim=1)
        idle_layer5 = self.idle_layer5(remaining_c5)+self.remaining_layer5_sub(remaining_c5)
        idle_concat5 = torch.cat([distilled_c5,idle_layer5],1)
        layer5_sub = self.layer5_sub(torch.cat([layer5,distilled_c5],1))
        
        layer6 = self.layer6(torch.cat((x, idle_concat1,idle_concat2,idle_concat3,idle_concat4,idle_concat5), 1))+self.residual_layer6_sub(idle_concat5)
        distilled_c6, remaining_c6 = torch.split(layer6, (self.distilled_channels, self.remaining_channels), dim=1)
        idle_layer6 = self.idle_layer6(remaining_c6)+self.remaining_layer6_sub(remaining_c6)
        idle_concat6 = torch.cat([distilled_c6,idle_layer6],1)
        layer6_sub = self.layer6_sub(torch.cat([layer6,distilled_c6],1))
        
        layer7 = self.layer7(torch.cat((x, idle_concat1,idle_concat2,idle_concat3,idle_concat4,idle_concat5,idle_concat6), 1))+self.residual_layer7_sub(idle_concat6)
        distilled_c7, remaining_c7 = torch.split(layer7, (self.distilled_channels, self.remaining_channels), dim=1)
        idle_layer7 = self.idle_layer7(remaining_c7)+self.remaining_layer7_sub(remaining_c7)
        idle_concat7 = torch.cat([distilled_c7,idle_layer7],1)
        layer7_sub = self.layer7_sub(torch.cat([layer7,distilled_c7],1))
        
        layer8 = self.layer8(torch.cat((x, idle_concat1,idle_concat2,idle_concat3,idle_concat4,idle_concat5,idle_concat6,idle_concat7),1))+self.residual_layer8_sub(idle_concat7)
        distilled_c8, remaining_c8 = torch.split(layer8, (16, 48), dim=1)
        idle_layer8 = self.idle_layer8(remaining_c8)

        out = torch.cat([layer1_sub,layer2_sub,layer3_sub,layer4_sub,layer5_sub,layer6_sub,layer7_sub,idle_layer8], dim=1) 
        out_split = torch.cat([distilled_c1,distilled_c2,distilled_c3,distilled_c4,distilled_c5,distilled_c6,distilled_c7,distilled_c8], dim=1) 
        x = self.lff(out)
        x_2 = self.lff_split(out_split)
        x = x+x_2

        y =self.contrast(x)+self.avg_pool(x)
        y = self.conv_du(y)
        x = x*y
        x = x+x_residual
        return x


class RecursiveBlock(nn.Module):
    def __init__(self,num_features, growth_rate, U,wn):
        super(RecursiveBlock, self).__init__()
        self.U = U
        self.G0 = num_features
        self.G = growth_rate

        self.rdbs = RDB(self.G0,wn=wn)
        #Feed Back
        self.up1 = nn.Sequential(wn(nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2)), nn.PixelShuffle(2))
        self.up1_4 = nn.Sequential(wn(nn.Conv2d(self.G0, self.G0, kernel_size=4,stride=2,padding=3//2)),nn.ReLU(True))

    def forward(self, sfe2):
        x=sfe2
        local_features = []
        for _ in range(self.U):
            x = self.rdbs(x)
            h1 = self.up1(x)
            l1 = self.up1_4(h1)
            h2 = self.up1(l1-x)
            l2 = self.up1_4(h1+h2)
            x = l2-x
            local_features.append(x)

        RB = torch.cat(local_features, 1)
        return RB
        
class MCSRNet(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_layers, B, U):
        super(MCSRNet, self).__init__()
        self.scale_factor=scale_factor
        self.B = B
        self.G0 = num_features
        self.G = growth_rate
        self.U = U
        self.C = num_layers
        #weight normalization
        wn = lambda x: torch.nn.utils.weight_norm(x)

        #shallow feature extraction
        self.sfe1 = wn(nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2))
        self.sfe2 = wn(nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2))
        self.rbs = nn.Sequential(*[RecursiveBlock(num_features,
                                                  growth_rate,
                                                  U,
                                                  wn=wn)])
        # global feature fusion
        self.gff = nn.Sequential(
            wn(nn.Conv2d(self.G * self.U * self.B, self.G0, kernel_size=1)),nn.ReLU(True),
            wn(nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2))
        )
        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )
        #High Frequency Residual Refinement & Reconstruction
        self.conv17_1 = nn.Sequential(wn(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,groups=1,bias=False)),nn.ReLU(inplace=True))
        self.conv17_2 = nn.Sequential(wn(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,groups=1,bias=False)),nn.ReLU(inplace=True))
        self.conv17_3 = nn.Sequential(wn(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,groups=1,bias=False)),nn.ReLU(inplace=True))
        self.conv17_4 = nn.Sequential(wn(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,groups=1,bias=False)),nn.ReLU(inplace=True))
        self.conv18 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1,groups=1,bias=False))

    def forward(self, x):
        x_up = F.interpolate(x, mode='bicubic',scale_factor=self.scale_factor)
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        RB= self.rbs(sfe2)
        x = self.gff(RB) + sfe1
        x = self.upscale(x)
        x_ = x
        x = self.conv17_1(x)
        x = self.conv17_2(x)
        x = self.conv17_3(x)
        x = self.conv17_4(x)+x_
        x = self.conv18(x)+x_up
        return x
