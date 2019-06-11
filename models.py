import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.spatio = spatioModel()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x[:,6:9,:,:]
        out = self.spatio(x)
        out = self.relu(self.input(out))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out

class DVDModel(nn.Module):
    def __init__(self):
        super(DVDModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.spatio = spatioModel()

        self.block1 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128)
        )
        self.block2 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256)
        )
        self.block3 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=512),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=512),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=512),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=512),
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256)
        )

        self.block4 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128)
        )

        self.block5 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128),
        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64)
        )

        self.block6 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=15, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=15),
        nn.Conv2d(in_channels=15, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=3)
        )

    def forward(self, x):
        out_flow = self.spatio(x)
        # input_cat = torch.cat((out_flow, x[:,6:9,:,:]), dim=0)
        out1 = self.relu(self.conv1(out_flow))
        # out1 = self.bn1(out1)
        out2 = self.block1(out1) # 21 w
        out3 = self.block2(out2) # 11 w
        out4 = self.block3(out3) # 6 w
        out4 = torch.add(out4, out3)
        out5 = self.block4(out4) # 12 w
        out5 = torch.add(out5, out2)
        out6 = self.block5(out5) # 24 w
        out6 = torch.add(out6, out1)
        out7 = self.block6(out6) # 41 w
        out7 = torch.add(out7, x[:,6:9,:,:])
        out7 = self.sigmoid(out7)
        # print("Out size is ", out.size())
        # print("Out size is ", out.size())
        return out7


class flowModel(nn.Module):
    def __init__(self):
        super(flowModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.block1 = nn.Sequential(
        nn.Conv2d(in_channels=15, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64)
        )
        self.block2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128)
        )
        self.block3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128)
        )

        self.block4 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64)
        )

        self.block5 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Tanh()
        )

    def forward(self, x):
        # out1 = self.bn1(out1)
        out1 = self.block1(x) # 21 w
        out2 = self.block2(out1) # 11 w
        out3 = self.block3(out2) # 6 w
        out3 = out3.add(out2)
        out4 = self.block4(out3) # 6 w
        out4 = out4.add(out1)
        out5 = self.block5(out4) # 6 w


        # print("Out size is ", out.size())
        # print("Out size is ", out.size())
        return out5


class spatioModel(nn.Module):
    def __init__(self):
        super(spatioModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.block1 = nn.Sequential(
        nn.Conv2d(in_channels=15, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64)
        )
        self.block2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128)
        )
        self.block3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128)
        )

        self.block4 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=128),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64)
        )

        self.block5 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Tanh()
        )

    def tri_interpolate(self, x, uvz):
        # u = uvz[:,0,:,:]
        # v = uvz[:,1,:,:]
        # uvz[:,2,:,:] = 2*uvz[:,2,:,:] + 2
        x_split = x.reshape(x.size(0),3,5,x.size(2), x.size(3))
        uvz = uvz.reshape(uvz.size(0),uvz.size(1),1,uvz.size(2), uvz.size(3))
        uvz = uvz.permute(0, 2, 3, 4, 1)
        warped_img = F.grid_sample(x_split, uvz)
        warped_img = warped_img.view(x.size(0),3,x.size(2), x.size(3))
        # for batch in range(x.size(0)):
        #     for m in range(x.size(2)):
        #         for n in range(x.size(3)):
        #             u_s = u[batch,m,n].int()
        #             v_s = v[batch,m,n].int()
        #             z_s = z[batch,m,n].int()
        #             if m + u_s < x.size(2) and n + v_s < x.size(3):
        #                 warped_tensor[batch,:,m,n] = x[batch, 3*z_s:3*(z_s+1),m-u_s,n-v_s]

        # print("uvz shape is ", warped_img.size())
        # x_split = x[:,6:9,:,:]
        # uvz = uvz.permute(0,2,3,1)
        # warped_img = F.grid_sample(x_split, uvz)
        return self.relu(warped_img)

    def flow_warp(self, x, flow, padding_mode='zeros'):
        """Warp an image or feature map with optical flow
        Args:
            x (Tensor): size (n, c, h, w)
            flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
            padding_mode (str): 'zeros' or 'border'

        Returns:
            Tensor: warped image or feature map
        """
        assert x.size()[-2:] == flow.size()[-2:]
        n, _, h, w = x.size()
        x_ = torch.arange(w).view(1, -1).expand(h, -1)
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x_, y_], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid = grid.add(2 * flow)
        grid = grid.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, padding_mode=padding_mode)


    def flow_warp_5d(self, x, flow, padding_mode='zeros'):
        """Warp an image or feature map with optical flow
        Args:
            x (Tensor): size (n, c, h, w)
            flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
            padding_mode (str): 'zeros' or 'border'

        Returns:
            Tensor: warped image or feature map
        """
        assert x.size()[-2:] == flow.size()[-2:]
        n, _, h, w = x.size()
        x_s = x.view(n, 3, 5, h, w)
        flow = flow.unsqueeze(1)
        x_ = torch.arange(w).view(1, -1).expand(h, -1)
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)
        z_ = torch.zeros((h, w), dtype=torch.long)
        grid = torch.stack([x_, y_, z_], dim=0).float().cuda()
        grid = grid.unsqueeze(0).unsqueeze(1).expand(n, 1, -1, -1, -1)
        grid[:, :, 0, :, :] = 2 * grid[:, :, 0, :, :] / (w - 1) - 1
        grid[:, :, 1, :, :] = 2 * grid[:, :, 1, :, :] / (h - 1) - 1
        grid = grid.add(2 * flow)
        grid = grid.permute(0, 1, 3, 4, 2)
        warp_img = F.grid_sample(x_s, grid, padding_mode=padding_mode)
        warp_img = warp_img.view(n, 3, h, w)
        return warp_img


    def forward(self, x):
        # out1 = self.bn1(out1)
        out1 = self.block1(x) # 21 w
        out2 = self.block2(out1) # 11 w
        out3 = self.block3(out2) # 6 w
        out3 = out3.add(out2)
        out4 = self.block4(out3) # 6 w
        out4 = out4.add(out1)
        out5 = self.block5(out4) # 6 w
        out6 = self.flow_warp_5d(x, out5)


        # print("Out size is ", out.size())
        # print("Out size is ", out.size())
        return out6
