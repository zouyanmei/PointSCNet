from ast import arg
from operator import sub
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import knn_edge_maxpooling

def checkpoint(function, *args, **kwargs):
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)

class LFP(nn.Module):
    r"""
    Local Feature Propagation Layer
    f = linear(f)
    f_i = bn(max{f_j | j in knn_i} - f_i)
    """
    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)
    
    def forward(self, x, knn):
        B, N, C = x.shape
        x = self.proj(x)
        x = knn_edge_maxpooling(x, knn, self.training)
        x = self.bn(x.view(B*N, -1)).view(B, N, -1)
        return x

class Mlp(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.mlp[-1].weight, init)
    
    def forward(self, x):
        B, N, C = x.shape
        x = self.mlp(x.view(B*N, -1)).view(B, N, -1)
        return x

class Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act):
        super().__init__()

        self.depth = depth
        self.lfps = nn.ModuleList([
            LFP(dim, dim, bn_momentum) for _ in range(depth)
        ])
        self.mlp = Mlp(dim, mlp_ratio, bn_momentum, act, 0.2)
        self.mlps = nn.ModuleList([
            Mlp(dim, mlp_ratio, bn_momentum, act) for _ in range(depth // 2)
        ])
        if isinstance(drop_path, list):
            drop_rates = drop_path
            self.dp = [dp > 0. for dp in drop_path]
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
            self.dp = [drop_path > 0.] * depth
        #print(drop_rates)
        self.drop_paths = nn.ModuleList([
            DropPath(dpr) for dpr in drop_rates
        ])
    
    def drop_path(self, x, i, pts):
        if not self.dp[i] or not self.training:
            return x
        return torch.cat([self.drop_paths[i](xx) for xx in torch.split(x, pts, dim=1)], dim=1)

    def forward(self, x, knn, pts=None):
        x = x + self.drop_path(self.mlp(x), 0, pts)
        for i in range(self.depth):
            x = x + self.drop_path(self.lfps[i](x, knn), i, pts)
            if i % 2 == 1:
                x = x + self.drop_path(self.mlps[i // 2](x), i, pts)
        return x


class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()

        self.depth = depth
        self.up_depth = len(args.depths) - 1

        self.first = first = depth == 0
        self.last = last = depth == self.up_depth
        self.dcd_use = depth == 0

        self.k = args.ks[depth]

        self.cp = cp = args.use_cp
        cp_bn_momentum = args.cp_bn_momentum if cp else args.bn_momentum

        dim = args.dims[depth]
        nbr_in_dim = 7 if first else 3
        nbr_hid_dim = args.nbr_dims[0] if first else args.nbr_dims[1] // 2
        nbr_out_dim = dim if first else args.nbr_dims[1]
        self.nbr_embed = nn.Sequential(
            nn.Linear(nbr_in_dim, nbr_hid_dim//2, bias=False),
            nn.BatchNorm1d(nbr_hid_dim//2, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim//2, nbr_hid_dim, bias=False),
            nn.BatchNorm1d(nbr_hid_dim, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim, nbr_out_dim, bias=False),
        )
        if first:
            x_dim = 4
        else:
            x_dim = dim
        self.x_embed = nn.Sequential(
            nn.Linear(x_dim, nbr_hid_dim//2, bias=False),
            nn.BatchNorm1d(nbr_hid_dim//2, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim//2, nbr_hid_dim, bias=False),
            nn.BatchNorm1d(nbr_hid_dim, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim, nbr_out_dim, bias=False),
        )

        self.nbr_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
        nn.init.constant_(self.nbr_bn.weight, 0.8 if first else 0.2)
        self.nbr_proj = nn.Identity() if first else nn.Linear(nbr_out_dim, dim, bias=False)

        if not first:
            in_dim = args.dims[depth - 1]
            self.lfp = LFP(in_dim, dim, args.bn_momentum, 0.3)
            self.skip_proj = nn.Sequential(
                nn.Linear(in_dim, dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.blk = Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, cp_bn_momentum, args.act)
        self.drop = DropPath(args.head_drops[depth])
        self.postproj = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            nn.Linear(dim, args.head_dim, bias=False),
        )
        nn.init.constant_(self.postproj[0].weight, (args.dims[0] / dim) ** 0.5)

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        if not last:
            self.sub_stage = Stage(args, depth + 1)
        self.dcd = DCDLayer(args=args, ration=4)
        self.channel_matric = nn.Parameter(torch.ones(1,args.head_dim))
    
    def local_aggregation(self, x, knn, pts):
        x = x.unsqueeze(0)
        x = self.blk(x, knn, pts)
        x = x.squeeze(0)
        return x

    def forward(self, x, xyz, prev_knn, indices, pts_list, npoint):
        """
        x: N x C
        """
        # downsampling
        if not self.first:
            ids = indices.pop()
            xyz = xyz[ids]
            x = self.skip_proj(x)[ids] + self.lfp(x.unsqueeze(0), prev_knn).squeeze(0)[ids]

        knn = indices.pop()
        
        # spatial encoding
        N, k = knn.shape
        nbr = xyz[knn] - xyz.unsqueeze(1)
        nbr = torch.cat([nbr, x[knn]], dim=-1).view(-1, 7) if self.first else nbr.view(-1, 3)
        if self.training and self.cp:
            nbr.requires_grad_()
        nbr_embed_func = lambda x: self.nbr_embed(x).view(N, k, -1)
        x_em = self.x_embed(x)
        x_em = x_em[knn].view(N, k, -1)
        nbr = checkpoint(nbr_embed_func, nbr) if self.training and self.cp else nbr_embed_func(nbr)
        nbr = self.nbr_proj((nbr+x_em).max(dim=1)[0])
        nbr = self.nbr_bn(nbr)
        x = nbr if self.first else nbr + x

        # main block
        knn = knn.unsqueeze(0)
        pts = pts_list.pop() if pts_list is not None else None
        x = checkpoint(self.local_aggregation, x, knn, pts) if self.training and self.cp else self.local_aggregation(x, knn, pts)

        # get subsequent feature maps
        if not self.last:
            sub_x, sub_c = self.sub_stage(x, xyz, knn, indices, pts_list, npoint)
        else:
            sub_x = sub_c = None
        
        # regularization
        if self.training:
            rel_k = torch.randint(self.k, (N, 1), device=x.device)
            rel_k = torch.gather(knn.squeeze(0), 1, rel_k).squeeze(1)
            rel_cor = (xyz[rel_k] - xyz)
            rel_cor.mul_(self.cor_std)
            # print(rel_cor.std(dim=0))
            rel_p = x[rel_k] - x
            rel_p = self.cor_head(rel_p)
            closs = F.mse_loss(rel_p, rel_cor)
            sub_c = sub_c + closs if sub_c is not None else closs
        
        # upsampling
        x = self.postproj(x)
        if not self.first:
            back_nn = indices[self.depth-1]
            x = x[back_nn]
        x = self.drop(x)
        if self.last:
            x= self.dcd(x, npoint)
        #sub_x = sub_x*0.5 + x*0.5 if sub_x is not None else x
        sub_x = sub_x + x if sub_x is not None else x

        return sub_x, sub_c

class DCDLayer(nn.Module):
    '''
    x:n, c
    '''
    def __init__(self, args, ration=4):
        super(DCDLayer, self).__init__()
        in_channels = args.head_dim
        mid_channels = in_channels*ration
 
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(mid_channels, in_channels, bias=False),
            args.act(),
            )
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(mid_channels, in_channels, bias=False),
            args.act(),
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x2, npoint):
        '''
        x1:n, c
        x2:n, c, best
        '''
        batch_size = len(npoint)
        cur_feature = x2[0:npoint[0], :]
        mean_f = torch.mean(cur_feature, 0, keepdim=True)
        for i in range(1, batch_size):
            cur_feature = x2[sum(npoint[0:i]):sum(npoint[0:(i+1)]), :]

            cur_mean = torch.mean(cur_feature, 0, keepdim=True)#1*c
            mean_f = torch.cat((mean_f, cur_mean), dim=0)

        out_mean = self.fc(mean_f)
        out_w = self.fc1(mean_f)
        out_w = self.sigmoid(out_w)
 
        out_cur = out_mean[0:1, :].repeat(npoint[0], 1)
        w_cur = out_w[0:1, :].repeat(npoint[0], 1)
        for i in range(1, batch_size):
            wei_cur = out_mean[i:i+1, :].repeat(npoint[i], 1)
            out_cur = torch.cat((out_cur, wei_cur), dim=0)

            wei_w = out_w[i:i+1, :].repeat(npoint[i], 1)
            w_cur = torch.cat((w_cur, wei_w), dim=0)
        
        x = w_cur*x2*0.5 + x2*0.75 + out_cur
        return x

class DelaSemSeg(nn.Module):
    r"""
    DeLA for Semantic Segmentation  

    args:               examples
        depths:         [4, 4, ..., 4]         
        dims:           [128, 256, ..., 512]
        nbr_dims:       [32, 32], dims in spatial encoding || 7->16->32->out->pool | 3->8->16->32->pool->out
        head_dim:       256, hidden dim in cls head
        num_classes:    13
        drop_paths:     [0., 0., ..., 0.1], in-stage drop path rate, can be list of lists, len(dp[i]) = depth[i]
        head_drops:     [0., 0.05, ..., 0.2], scale wise drop rate before cls head
        bn_momentum:    0.02         
        act:            nn.GELU
        mlp_ratio:      2, can be float
        use_cp:         False, enable gradient checkpoint to save memory
                        If True, blocks and spatial encoding are checkpointed
    """
    def __init__(self, args):
        super().__init__()

        # bn momentum for checkpointed layers
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum)**0.5

        self.stage = Stage(args)

        hid_dim = args.head_dim
        out_dim = args.num_classes

        self.head = nn.Sequential(
            nn.BatchNorm1d(hid_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(hid_dim, out_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
              nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, pts_list=None, npoint=None):
        indices = indices[:]
        x, closs = self.stage(x, xyz, None, indices, pts_list, npoint)
        if self.training:
            return self.head(x), closs
        return self.head(x)
