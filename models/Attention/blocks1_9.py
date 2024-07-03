import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
import math
# feedforward and attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        heads = 8,
        dim = 64,
        dropout = 0.,
        inner_dim = 0, 
    ):
        super().__init__()

        self.heads = heads
        if inner_dim == 0:
            inner_dim = dim
        self.scale = (inner_dim/heads) ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_out = False):
        h = self.heads
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        if attn_out:
            return out, attn
        else:
            return out



class ProdAttention(nn.Module):
    def __init__(
        self,
        heads = 8,
        dim = 64,
        dropout = 0.,
        inner_dim = 0, 
        topk = -1,
    ):
        super().__init__()

        self.heads = heads
        if inner_dim == 0:
            inner_dim = dim

        # if topk = -1:
        self.dim = dim
        self.topk = topk
        
        self.scale = (inner_dim/heads) ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.2)
        ) 

    def forward(self, x, attn_out = False):
        h = self.heads
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)



        value, idx = torch.topk(attn, dim=-1, k=self.topk) # bs group num_per_group
        idx = idx.unsqueeze(-1).repeat((1,1,1,1,self.dim // h))
        vv = v.unsqueeze(-2).repeat((1,1,1,self.topk,1))
        xx_ = torch.gather(vv, 2, idx)


        # =================== output ===================
        x = xx_.sum(dim=-2)
        x = (x - x.min())/ (x.max()-x.min())
        out = torch.exp(x)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.out(out)

        return out

        
# class ToMemoryProd(nn.Module):
#     def __init__(
#             self, 
#             token_num, 
#             heads, 
#             dim, 
#             attn_dropout, 
#             cluster, 
#             target_mode, 
#             groups, 
#             num_per_group,
#             use_cls_token) -> None:
#         super().__init__()

#         num_per_group = max(math.ceil(token_num/groups), num_per_group)

#         self.target_token = nn.Parameter(torch.rand([groups, dim]))
#         self.soft = nn.Softmax(dim=-1)

#         self.dropout = nn.Dropout(attn_dropout)
#         self.gather_layer = nn.Conv1d(groups * num_per_group, groups, groups=groups, kernel_size=1)

        
#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.out = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Dropout(attn_dropout)
#         ) 

#         self.groups = groups
#         self.num_per_group = num_per_group
#         self.heads = heads
#         self.target_mode = target_mode
#         self.cluster = cluster

#         # if cluster and (target_mode == 'mix'):
#         #     self.to_target = nn.Linear(groups+token_num+int(use_cls_token), groups)
#         if cluster :
#             if target_mode == 'mix':
#                 self.target_token = nn.Parameter(torch.rand([groups, dim]))
#                 self.to_target = nn.Linear(groups+token_num+int(use_cls_token), groups+int(use_cls_token))
#             else:
#                 self.target_token = nn.Parameter(torch.rand([groups+int(use_cls_token), dim]))
        

#     def forward(self, x):
#         b,l,d = x.shape
#         h = self.heads

#         x = torch.log(nn.ReLU()(x) + 1)
#         target = self.target_token
#         target = target.reshape(1, self.groups, -1).repeat((b,1,1))

#         # =================== define qkv ===================
#         if self.cluster:
#             if self.target_mode == 'mix':
#                 target = torch.cat([target, x], dim=-2)
#                 target = self.to_target(target.transpose(-1, -2)).transpose(-1, -2)
#             q = self.q(target)
#         else:
#             q = self.q(x)
#         k = self.k(x)
#         v = self.v(x)
#         q = q.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)
#         k = k.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)
#         v = v.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)

#         # =================== cak attn ===================
#         attn = self.soft(
#             torch.matmul(q, k.transpose(-1,-2)) * (self.args.dim_head ** -0.5)
#         )
#         attn = self.dropout(attn)

#         # =================== find topk related ===================
#         value, idx = torch.topk(attn, dim=-1, k=self.num_per_group) # bs group num_per_group
#         idx = idx.unsqueeze(-1).repeat((1,1,1,1,d // h))
#         vv = v.unsqueeze(-2).repeat((1,1,1,self.num_per_group,1))
#         xx_ = torch.gather(vv, 2, idx)


#         # =================== output ===================
#         x = xx_.sum(dim=-2)
#         x = (x - x.min())/ (x.max()-x.min())
#         out = torch.exp(x)
#         out = rearrange(out, 'b h n d -> b n (h d)', h = h)
#         out = self.out(out)

#         return out
    



# class ToMemorySum(nn.Module):
#     def __init__(
#             self, 
#             token_num, 
#             heads, 
#             dim, 
#             attn_dropout, 
#             cluster, 
#             target_mode, 
#             groups, 
#             num_per_group,
#             use_cls_token) -> None:
#         super().__init__()

#         num_per_group = max(math.ceil(token_num/groups), num_per_group)

#         self.target_token = nn.Parameter(torch.rand([groups, dim]))
#         self.soft = nn.Softmax(dim=-1)

#         self.dropout = nn.Dropout(attn_dropout)
#         self.gather_layer = nn.Conv1d(groups * num_per_group, groups, groups=groups, kernel_size=1)

        
#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.out = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Dropout(attn_dropout)
#         ) 

#         self.groups = groups
#         self.num_per_group = num_per_group
#         self.heads = heads
#         self.target_mode = target_mode
#         self.cluster = cluster

#         # if cluster and (target_mode == 'mix'):
#         #     self.to_target = nn.Linear(groups+token_num+int(use_cls_token), groups)
#         if cluster :
#             if target_mode == 'mix':
#                 self.target_token = nn.Parameter(torch.rand([groups, dim]))
#                 self.to_target = nn.Linear(groups+token_num+int(use_cls_token), groups+int(use_cls_token))
#             else:
#                 self.target_token = nn.Parameter(torch.rand([groups+int(use_cls_token), dim]))
        

#     def forward(self, x):
#         b,l,d = x.shape
#         h = self.heads

#         target = self.target_token
#         target = target.reshape(1, self.groups, -1).repeat((b,1,1))

#         # =================== define qkv ===================
#         if self.cluster:
#             if self.target_mode == 'mix':
#                 target = torch.cat([target, x], dim=-2)
#                 target = self.to_target(target.transpose(-1, -2)).transpose(-1, -2)
#             q = self.q(target)
#         else:
#             q = self.q(x)
#         k = self.k(x)
#         v = self.v(x)
#         q = q.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)
#         k = k.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)
#         v = v.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)

#         # =================== cak attn ===================
#         attn = self.soft(
#             torch.matmul(q, k.transpose(-1,-2)) * (self.args.dim_head ** -0.5)
#         )
#         attn = self.dropout(attn)

#         # =================== find topk related ===================
#         value, idx = torch.topk(attn, dim=-1, k=self.num_per_group) # bs group num_per_group
#         idx = idx.unsqueeze(-1).repeat((1,1,1,1,d // h))
#         vv = v.unsqueeze(-2).repeat((1,1,1,self.num_per_group,1))
#         xx_ = torch.gather(vv, 2, idx)


#         # =================== output ===================
#         x = xx_.sum(dim=-2)
#         out = rearrange(out, 'b h n d -> b n (h d)', h = h)
#         out = self.out(out)

#         return out
    



class MemoryBlock(nn.Module):
    def __init__(
            self, 
            token_num, 
            heads, 
            dim, 
            attn_dropout, 
            cluster, 
            target_mode, 
            groups, 
            num_per_group,
            use_cls_token,
            sum_or_prod = None,
            qk_relu = False) -> None:
        super().__init__()

        if num_per_group == -1:
            self.num_per_group = -1
            # Do not use grouping, calculate for all
        else:
            self.num_per_group = max(math.ceil(token_num/groups), num_per_group)
            num_per_group = max(math.ceil(token_num/groups), num_per_group)
            self.gather_layer = nn.Conv1d((groups+int(use_cls_token)) * num_per_group, groups+int(use_cls_token), groups=groups+int(use_cls_token), kernel_size=1)

        
        self.soft = nn.Softmax(dim=-1)
        self.qk_relu = qk_relu
        self.dropout = nn.Dropout(attn_dropout)
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(attn_dropout)
        ) 

        self.groups = groups
        self.use_cls_token = int(use_cls_token)
        # self.num_per_group = num_per_group
        self.heads = heads
        self.target_mode = target_mode
        self.cluster = cluster

        if cluster :
            if target_mode == 'mix':
                self.target_token = nn.Parameter(torch.rand([groups, dim]))
                self.to_target = nn.Linear(groups+token_num+int(use_cls_token), groups+int(use_cls_token))
            else:
                self.target_token = nn.Parameter(torch.rand([groups+int(use_cls_token), dim]))

        if sum_or_prod not in ['sum', 'prod']:
            print('{} is not in [sum, prod]'.format(sum_or_prod))
            raise ValueError
        self.sum_or_prod = sum_or_prod
        self.scale = dim/heads

        

    def forward(self, x):
        b,l,d = x.shape
        h = self.heads

        if self.sum_or_prod == 'prod':
            x = torch.log(nn.ReLU()(x) + 1e-9)
        target = self.target_token
        target = target.reshape(1, -1, d).repeat((b,1,1))

        # =================== define qkv ===================
        if self.cluster:
            if self.target_mode == 'mix':
                target = torch.cat([target, x], dim=-2)
                target = self.to_target(target.transpose(-1, -2)).transpose(-1, -2)
            q = self.q(target)
        else:
            q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)
        k = k.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)
        v = v.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)

        # if self.qk_relu:
        #     q = nn.ReLU()(q)
        #     k = nn.ReLU()(k)

        # =================== cak attn ===================
        attn = self.soft(
            torch.matmul(q, k.transpose(-1,-2)) * (self.scale ** -0.5)
        )
        attn = self.dropout(attn)


        # =================== gather relative tokens ===================
        if self.num_per_group == -1:
            x = einsum('b h i j, b h j d -> b h i d', attn, v)
            # x = rearrange(x, 'b h n d -> b n (h d)', h = h)
        else:
            # ===== find topk related for each group =====
            value, idx_original = torch.topk(attn, dim=-1, k=self.num_per_group) # bs head group num_per_group

            # ===== apply summation =====
            idx = idx_original.unsqueeze(-1).repeat((1,1,1,1,d // h))
            vv = v.unsqueeze(-2).repeat((1,1,1,self.num_per_group,1))
            xx_ = torch.gather(vv, 2, idx)


            #???????????????????????????????
            # x = xx_.sum(dim=-2)
            x = self.gather_layer(xx_.reshape(b*h, -1, d//h)).reshape(b, h, -1, d//h)
            
            # flag_map = torch.zeros_like(attn) #bs head group token
            # flag_map = flag_map.scatter_(-1,idx_original,1)
            # v_ = v.unsqueeze(dim=2).repeat(1,1,self.groups+1,1,1)
            # flag_map_ = flag_map.unsqueeze(-1).repeat(1,1,1,1,d//h)
            # xx_ = flag_map_ * v_ 
            # x = xx_.sum(dim=-2)



        # =================== output ===================
        
        if self.sum_or_prod == 'prod':
            x = (x - x.min())/ (x.max()-x.min())
            x = torch.exp(x)
        out = rearrange(x, 'b h n d -> b n (h d)', h = h)
        out = self.out(out)

        return out