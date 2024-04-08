import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.utils import *
from models.resnet2d import ResNetBackbone2d


class Attention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0., temperature=0.1):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not (heads == 1 and inner_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.attn = nn.Softmax(dim = -1)
        self.temp = temperature
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t : rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # qnorm, knorm = torch.linalg.norm(q, dim=-1, keepdims=True), torch.linalg.norm(k, dim=-1, keepdims=True)
        # qknorm = torch.matmul(qnorm, knorm.transpose(-1, -2))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots /= self.temp

        attn = self.attn(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.layer(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm = nn.Identity()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
        
    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x
        return x
        

class VIT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, head_dim, mlp_dim, channels=1, dropout=0., emb_dropout=0., pool='cls') -> None:
        super().__init__()
        img_size, patch_size = make_pair(img_size), make_pair(patch_size)
        num_patches = 1
        patch_dim = channels

        for i, p in zip(img_size, patch_size):
            assert i % p == 0, 'Image size must be divisible by patch size'
            num_patches *= i // p
            patch_dim *= p
        print(num_patches, patch_dim)

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', p1=patch_size[0], p2=patch_size[1], p3=patch_size[2]),
            nn.Linear(patch_dim, dim)
        )
        
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, head_dim, mlp_dim, dropout)

        assert pool in ['cls', 'mean'], 'Only support cls or mean for pool type'
        self.pool = pool

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        x = self.to_patch_emb(x)
        b, n, _ = x.shape

        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([x, cls_token], dim = 1) 
        x += self.pos_emb
        x = self.dropout(x)

        out = self.transformer(x)
        out = out[:, 0] if self.pool == 'cls' else out.mean(dim=1)
        out = self.classifier(out)
        return out


class ConvVIT(nn.Module):
    def __init__(self, img_size, num_classes, dim, depth, heads, head_dim, mlp_dim, channels=1, dropout=0., emb_dropout=0.):
        super().__init__()
        img_size = make_pair(img_size)

        self.backbone = ResNetBackbone2d(channels, [2,2,2,2], [32,64,128,256], dropout)
        # self.deephead = nn.Linear(256, num_classes)

        self.patch_dim = 256 
        self.patch_num = img_size[-1]

        self.pos_emb = nn.Parameter(torch.randn(1, self.patch_num, dim))
        if self.patch_dim != dim:
            self.to_patch_emb = nn.Linear(self.patch_dim, dim)
        else:
            self.to_patch_emb = nn.Identity()
        self.transformer = Transformer(dim, depth, heads, head_dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(emb_dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b c d h w -> (b w) c d h')
        x = self.backbone(x)

        deep_out = torch.clone(x)
        # deep_out = self.deephead(deep_out)

        x = rearrange(x, '(b n) d -> b n d', b = b)

        x = self.to_patch_emb(x)
        x += self.pos_emb
        x = self.dropout(x)

        out = self.transformer(x)
        out = out.mean(dim=1)
        out = self.classifier(out)

        return out, deep_out


class SliceResNet(nn.Module):
    def __init__(self, img_size, num_classes, dim, depth, heads, head_dim, mlp_dim, channels=1, dropout=0., emb_dropout=0.):
        super().__init__()
        img_size = make_pair(img_size)

        self.backbone = ResNetBackbone2d(channels, [1,1,1,1], [32, 64, 128, 256], dropout)
        self.classifier = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b c d h w -> (b w) c d h')
        x = self.backbone(x)
        x = rearrange(x, '(b n) d -> b n d', b = b)
        x = x.mean(dim=1)

        out = self.classifier(x)
        return out


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class Recorder(nn.Module):
    def __init__(self, vit, device = None):
        super().__init__()
        self.vit = vit

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.vit.transformer, Attention)
        for module in modules:
            handle = module.attn.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings))

        attns = torch.stack(recordings, dim = 1) if len(recordings) > 0 else None
        return pred, attns