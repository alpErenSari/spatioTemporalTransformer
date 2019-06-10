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
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

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
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.flow_model = flowModel()

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
        out_flow = self.flow_model(x)
        input_cat = torch.cat((out_flow, x[:,6:9,:,:]), dim=0)
        out1 = self.relu(self.conv1(input_cat))
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
    def __init__(self, batchSize, out_size):
        super(spatioModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.x_v = torch.arange(out_size).expand((batchSize,out_size,out_size))
        self.y_v = self.x_v.transpose(1,2)

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
        return warped_img


    def forward(self, x):
        # out1 = self.bn1(out1)
        out1 = self.block1(x) # 21 w
        out2 = self.block2(out1) # 11 w
        out3 = self.block3(out2) # 6 w
        out3 = out3.add(out2)
        out4 = self.block4(out3) # 6 w
        out4 = out4.add(out1)
        out5 = self.block5(out4) # 6 w
        out6 = self.tri_interpolate(x, out5)


        # print("Out size is ", out.size())
        # print("Out size is ", out.size())
        return out6
