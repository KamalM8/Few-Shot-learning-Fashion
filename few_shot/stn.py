from math import sin, cos, pi
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class ClipAffine():
    """Apply constraints on affine transformation parameters.

    Args:
        angle_range (tuple): rotation angle range (example: (-25,25))
        scale_range (list): scaling factor range(s_minx, s_maxx, s_miny, s_maxy)
                            example: [0.2,1.2,0.4,1.4] will scale will
                            constrain x axis scaling between 0.2 and 1.2.
                            Similarly, y axis scaling will be constrained
                            between 0.4 and 1.4
        translation_range (list):
                            translation factor range (function of input size)
                            example: [-0.2,0.2,-0.1,0.1], input size = (50,100)
                            will constrain x axis translation between 0.2*50 and
                            0.3*50. Similarly, y axis translation will be constrained
                            between 0.1*100 and 0.4*100
    """

    def __init__(self, angle_range, scale_range, translation_range):

        assert isinstance(angle_range, (tuple))
        assert isinstance(scale_range, (list))
        assert isinstance(translation_range, (list))

        self.min_angle = angle_range[0]/180.0*pi
        self.max_angle = angle_range[1]/180.0*pi
        self.s_minx = scale_range[0]
        self.s_maxx = scale_range[1]
        self.s_miny = scale_range[2]
        self.s_maxy = scale_range[3]
        self.t_minx = translation_range[0]
        self.t_maxx = translation_range[1]
        self.t_miny = translation_range[2]
        self.t_maxy = translation_range[3]

    def clip(self, params):
        """
        function clips six affine transformation params: [a,b,c,d,e,f]
        """

        output = torch.empty_like(params)
        for i, param in enumerate(params):
            param_clamped = torch.empty_like(param)
            # a
            param_clamped[0] = torch.clamp(param[0], min = cos(self.min_angle)*
                                            self.s_minx, max=self.s_maxx)
            # param_clamped[0] = param[0]

            # b
            param_clamped[1] = torch.clamp(param[1], min = sin(self.min_angle)*
                                            self.s_miny, max = sin(self.max_angle)*
                                            self.s_maxy)

            # param_clamped[1] = 0 # turn on for attention based model


            x_min = self.t_minx
            x_max = self.t_maxx

            # c (clamp horizontal translation)
            param_clamped[2] = torch.clamp(param[2], min=x_min, max = x_max)
            # param_clamped[2] = param[2]

            # d
            param_clamped[3] = torch.clamp(param[3], min = -sin(self.max_angle)*self.s_maxx,
                                            max = -sin(self.min_angle)*self.s_minx)
            # param_clamped[3] = 0 # turn on for attention based model

            # e
            param_clamped[4] = torch.clamp(param[4], min = cos(self.min_angle)*
                                             self.s_miny, max=self.s_maxy)
            # param_clamped[4] = param[4]

            y_min = self.t_miny
            y_max = self.t_maxy

            # f (clamp vertical translation)
            param_clamped[5] = torch.clamp(param[5], min=y_min, max = y_max)

            # param_clamped[5] = param[5]

            output[i] = param_clamped

        return output


class STNv0(nn.Module):
    def __init__(self, xdim, args, constrained):
        super(STNv0, self).__init__()
        self.xdim = xdim
        self.args = args
        self.constrained = constrained
        self.clipper = None

        # if self.constrained:
        angle_range = (-10,10)
        scale_range = [0.5, 1.5, 0.5, 1.5]
        translation_range = [-0.2, 0.2, -0.2, 0.2]
        if constrained:
            self.clipper = ClipAffine(angle_range, scale_range, translation_range)
        hdim = self.args.stn_hid_dim
        self.fcx = int(xdim[1] / 4) if xdim[1] == 28 else int(xdim[1]/8)

        # get the module
        self.identity_transform = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.double)
        self.identity_transform = Variable(self.identity_transform)
        if args is None:
            dropout = 0.5
        else:
            dropout = args.dropout
        self.dropout = dropout
        print('Using dropout of {} for STN'.format(dropout))
        module = []
        module.append(conv_block(xdim[0], hdim))
        module.append(conv_block(hdim, hdim))
        if self.xdim[1] > 28:
            module.append(conv_block(hdim, hdim))
        module.append(Flatten())

        # This is 7x7
        module.append(nn.Linear(hdim * self.fcx * self.fcx, 32))
        module.append(nn.ReLU())
        module.append(nn.Linear(32, 6))
        # module.append(nn.Tanh())
        self.module = nn.Sequential(*module)
        self._init_weights()

    def _init_weights(self):
        # initialize weights here
        index = -1
        self.module[index].weight.data.zero_()
        self.module[index].bias.data.copy_(self.identity_transform.squeeze())

    def forward(self, sample, support=False):
        # do the actual forward passes
        # dropout probability for dropping the final theta and putting
        # default value of [1....10]
        self.identity_transform = self.identity_transform.to(sample.device)
        if self.training and not support:
            dropout = self.dropout
        else:
            dropout = 1

        sample = Variable(sample)
        inp_flatten = sample
        # do the forward pass
        theta = self.module(inp_flatten)
        theta = theta + 0

        # Scale it to have any values
        B = sample.shape[0]
        U = torch.rand(B)
        idx = (U <= dropout).nonzero().squeeze()
        theta[idx, :] = self.identity_transform

        # constrain if enabled
        if self.constrained:
            theta = self.clipper.clip(theta)
        # print(theta)

        # change the shape
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, inp_flatten.size(), align_corners=True)
        results = F.grid_sample(inp_flatten, grid, padding_mode="border", align_corners=True)

        transform = theta
        return results, transform, {}


class STNv1(nn.Module):
    def __init__(self, xdim, args):
        super(STNv1, self).__init__()
        hdim = args.stn_hid_dim
        self.xdim = xdim
        self.args = args
        self.fcx = int(xdim[1] / 4) if xdim[1] == 28 else int(xdim[1]/8)
        print(self.fcx)
        # get the module
        self.identity_transform = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.double)
        self.identity_transform = Variable(self.identity_transform)
        if args is None:
            dropout = 0.5
        else:
            dropout = args.dropout
        self.dropout = dropout
        print('Using dropout of {} for STN'.format(dropout))
        module = []
        module.append(conv_block(xdim[0], hdim))
        module.append(conv_block(hdim, hdim))
        if self.xdim[1] > 28:
            module.append(conv_block(hdim, hdim))
        module.append(Flatten())
        # This is 7x7
        module.append(nn.Linear(hdim * self.fcx * self.fcx, 16))
        module.append(nn.ReLU())
        self.module = nn.Sequential(*module)
        # Add modules for scale, rotation, and translation
        self.scaler = nn.Linear(16, 1)
        self.theta = nn.Linear(16, 1)
        self.translation = nn.Linear(16, 2)

    def forward(self, sample, support=False):
        # do the actual forward passes
        # dropout probability for dropping the final theta and putting
        # default value of [1....10]
        self.identity_transform = self.identity_transform.to(sample.device)
        # Training and this is not the support set
        if self.training and not support:
            dropout = self.dropout
        else:
            dropout = 1

        sample = Variable(sample)
        inp_flatten = sample
        # do the forward pass
        params = self.module(inp_flatten)

        # Theta, translation and scale
        eps = self.args.scalediff
        theta = (torch.tanh(self.theta(params)) * self.args.theta).squeeze()
        txty  = torch.tanh(self.translation(params)) * self.args.t
        tx = txty[:, 0]
        ty = txty[:, 1]
        scale = (torch.tanh(self.scaler(params)) * eps + 1).squeeze()

        # Horizontal flip
        flip = torch.rand_like(scale) <= self.args.fliphoriz
        flip = flip.to(sample.device).double()
        flip = 1 - 2*flip

        # Get theta
        B = scale.shape[0]
        Theta = torch.zeros((B, 6)).double()
        Theta = Theta.to(sample.device)
        Theta[:, 0] = scale*flip*torch.cos(theta)
        Theta[:, 1] = -scale*torch.sin(theta)
        Theta[:, 2] = tx
        Theta[:, 3] = scale*flip*torch.sin(theta)
        Theta[:, 4] = scale*torch.cos(theta)
        Theta[:, 5] = ty
        # Back to theta
        theta = Theta

        # Scale it to have any values
        U = torch.rand(B)
        idx = (U <= dropout).nonzero().squeeze()
        theta[idx, :] = self.identity_transform
        # change the shape
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, inp_flatten.size(), align_corners=True)
        results = F.grid_sample(inp_flatten, grid, align_corners=True)

        transform = theta
        return results, transform, {}


# STN-VAE
class STNVAE(nn.Module):

    """STNVAE - Basically an extension where we get 2 outputs
    which acts as mean and variance
    """

    def __init__(self, xdim, hdim=16, dropout=0.5):
        """TODO: to be defined. """
        super(STNVAE, self).__init__()
        self.xdim = xdim
        self.fcx = int(xdim[1] / 4) if xdim[1] == 28 else int(xdim[1]/8)
        print(self.fcx)
        # get the module
        self.identity_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        self.identity_transform = Variable(self.identity_transform)
        self.dropout = dropout
        print('Using VAE STN')
        module = []
        fc = []
        module.append(conv_block(xdim[0], hdim))
        module.append(conv_block(hdim, hdim))
        if xdim[1] > 28:
            module.append(conv_block(hdim, hdim))
        module.append(Flatten())
        # This is 7x7
        module.append(nn.Linear(hdim * self.fcx * self.fcx, 64))
        # Get mean variance here
        fc.append(nn.Linear(32, 32))
        fc.append(nn.ReLU())
        fc.append(nn.Linear(32, 6))
        fc.append(nn.Tanh())

        self.module = nn.Sequential(*module)
        self.fc = nn.Sequential(*fc)
        self._init_weights()

    def _init_weights(self):
        # initialize weights here
        index = -2
        self.fc[index].weight.data.zero_()
        self.fc[index].bias.data.copy_(self.identity_transform)

    def forward(self, sample, ):
        # do the actual forward passes
        # dropout probability for dropping the final theta and putting
        # default value of [1....10]
        results = dict()
        transform = []
        self.identity_transform = self.identity_transform.to(sample['xs'].device)
        info = dict(mean=[], logstd=[])
        for k in ['xs', 'xq']:
            sample[k] = Variable(sample[k])
            inp = sample[k]
            n_classes, n_shot = inp.shape[:2]
            inp_flatten = inp.view(n_classes*n_shot, *inp.shape[2:])
            # do the forward pass
            out = self.module(inp_flatten)
            outm = out[:, :32]
            outlogstd = out[:, 32:]
            outstd = torch.exp(outlogstd)
            out = outm + torch.randn_like(outstd).to(outstd.device)*outstd
            info['mean'].append(outm)
            info['logstd'].append(outlogstd)
            # Get theta
            theta = self.fc(out)
            # Scale it to have any values
            B = theta.shape[0]
            U = torch.rand(B) < self.dropout
            theta = theta + 0
            theta[U] = self.identity_transform
            # change the shape
            theta = theta.view(-1, 2, 3)
            grid = F.affine_grid(theta, inp_flatten.size())
            x = F.grid_sample(inp_flatten, grid)
            # put into results
            results[k] = x.view(*inp.shape)
            transform.append(theta)
        # Copy all the other non-tensor keys
        for k in sample.keys():
            if k in ['xs', 'xq']:
                continue
            results[k] = sample[k]
        return results, transform, info


