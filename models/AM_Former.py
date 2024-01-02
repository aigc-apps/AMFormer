import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
import math
'''
主模型：FTTransformer
Forward输入：
x_categ, x_numer, label
x_categ：离散特征，形状为bs * col-num-categ，其中col-num表示有多少列离散特征，位置需要固定
x_numer：连续特征，形状为bs * col-num-numer
label因为在训练验证时需要计算loss，所以也直接放在这里。

Forward输出：
logit, loss
logit：概率分布
loss：用于训练和验证，在推理的时候可以直接去掉

参数说明在FTTransformer内部
'''

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
            sum_or_prod = None) -> None:
        super().__init__()

        if num_per_group == -1:
            self.num_per_group = -1
            # Do not use grouping, calculate for all
        else:
            self.num_per_group = max(math.ceil(token_num/groups), num_per_group)
            num_per_group = max(math.ceil(token_num/groups), num_per_group)
            self.gather_layer = nn.Conv1d((groups+int(use_cls_token)) * num_per_group, groups+int(use_cls_token), groups=groups+int(use_cls_token), kernel_size=1)

        
        self.soft = nn.Softmax(dim=-1)
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
            x = torch.log(nn.ReLU()(x) + 1)
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


        # =================== calculate attn ===================
        attn = self.soft(
            torch.matmul(q, k.transpose(-1,-2)) * (self.scale ** -0.5)
        )
        attn = self.dropout(attn)


        # =================== gather relative tokens ===================
        if self.num_per_group == -1:
            x = einsum('b h i j, b h j d -> b h i d', attn, v)
        else:
            value, idx_original = torch.topk(attn, dim=-1, k=self.num_per_group) # bs head group num_per_group
            idx = idx_original.unsqueeze(-1).repeat((1,1,1,1,d // h))
            vv = v.unsqueeze(-2).repeat((1,1,1,self.num_per_group,1))
            xx_ = torch.gather(vv, 2, idx)
            x = self.gather_layer(xx_.reshape(b*h, -1, d//h)).reshape(b, h, -1, d//h)


        # =================== output ===================
        if self.sum_or_prod == 'prod':
            x = (x - x.min())/ (x.max()-x.min())
            x = torch.exp(x)
        out = rearrange(x, 'b h n d -> b n (h d)', h = h)
        out = self.out(out)

        return out
    

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        attn_dropout,
        ff_dropout,
        use_cls_token,
        groups,
        sum_num_per_group,
        prod_num_per_group,
        cluster,
        target_mode,
        token_num,
        token_descent=False,
        use_prod=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        flag = int(use_cls_token)
        if not token_descent:
            groups = [token_num for _ in groups]

        for i in range(depth):
            token_num = token_num if i == 0 else groups[i-1]
            self.layers.append(nn.ModuleList([
                MemoryBlock(
                    token_num=token_num, 
                    heads=heads, 
                    dim=dim, 
                    attn_dropout=attn_dropout, 
                    cluster=cluster, 
                    target_mode=target_mode, 
                    groups=groups[i], 
                    num_per_group=prod_num_per_group[i],
                    use_cls_token=use_cls_token,
                    sum_or_prod='prod') if use_prod else nn.Identity(),
                MemoryBlock(
                    token_num=token_num, 
                    heads=heads, 
                    dim=dim, 
                    attn_dropout=attn_dropout, 
                    cluster=cluster, 
                    target_mode=target_mode, 
                    groups=groups[i], 
                    num_per_group=sum_num_per_group[i],
                    use_cls_token=use_cls_token,
                    sum_or_prod='sum') if token_descent else Attention(heads=heads, dim=dim, dropout=attn_dropout),
                nn.Linear(2*(groups[i] + flag), groups[i] + flag),
                nn.Linear(token_num + flag, groups[i] + flag) if token_descent else nn.Identity(),
                FeedForward(dim, dropout = ff_dropout),
            ]))   
        self.use_prod = use_prod



    def forward(self, x):
        for toprod, tosum, down, downx, ff in self.layers:
            
            attn_out = tosum(x)
            if self.use_prod:
                prod = toprod(x)
                attn_out = down(torch.cat([attn_out, prod], dim=1).transpose(2,1)).transpose(2,1)

            x = attn_out + downx(x.transpose(-1, -2)).transpose(-1, -2)
            x = ff(x) + x

        return x


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        b,n = x.shape
        x = x.reshape(b, n, 1)
        return x * self.weights + self.biases


class FTTransformer(nn.Module):
    def __init__(
        self,
        dim = 192,
        depth = 3,
        heads = 8,
        attn_dropout = 0.2,
        ff_dropout = 0.1,
        use_cls_token = True,
        groups = [54, 54, 54],
        sum_num_per_group = [32, 16, 8],
        prod_num_per_group = [6, 6, 6],
        cluster = True,
        target_mode = 'mix',
        token_descent = True,
        use_prod = True,
        num_special_tokens = 2,
        num_unique_categories = 10000,
        out = 2,
        num_cont = 104,
        num_cate = 16,
        use_sigmoid = True,
        # args,
    ):
        super().__init__()

        token_num = num_cont + num_cate
        '''
        dim: token dim
        depth: Attention block numbers
        heads: heads in multi-head attn
        attn_dropout: dropout in attn
        ff_dropout: drop in ff in attn
        use_cls_token: use cls token in FT-transformer but autoint it should be False
        groups: used in Memory block --> how many cluster prompts
        sum_num_per_group: used in Memory block --> topk to sum in each sum cluster prompts
        prod_num_per_group: used in Memory block --> topk to sum in each prod cluster prompts
        cluster: if True, prompt --> q, False, x --> q
        target_mode: if None, prompt --> q, if mix, [prompt, x] --> q
        token_descent: will decrease the number of tokens if True
        use_prod: use prod block
        num_special_tokens: =2
        num_unique_categories: how many cate features in total
        out: =1 if regressioin else =cls number
        num_cont = args.num_cont
        num_cate: how many cate col
        use_sigmoid: wether use sigmoid if out == 1
        '''


        total_tokens = num_unique_categories + num_special_tokens + 1
        if num_unique_categories > 0:
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        if num_cont > 0:
            self.numerical_embedder = NumericalEmbedder(dim, num_cont)


        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))


        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_cls_token=use_cls_token,
            groups=groups,
            sum_num_per_group=sum_num_per_group,
            prod_num_per_group=prod_num_per_group,
            cluster=cluster,
            target_mode=target_mode,
            token_num=token_num,
            token_descent=token_descent,
            use_prod=use_prod,
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, out)
        )

        self.pool = nn.Linear(num_cont + num_cate, 1)

        # Params. used in forward
        self.use_sigmoid = use_sigmoid
        self.use_cls_token = use_cls_token
        self.num_unique_categories = num_unique_categories
        self.num_cont = num_cont
        self.out = out
        

    # def model_name(self):
    #     return 'test'

    def forward(self, x_categ, x_numer, label, step=0):

        # print(x_categ.shape, x_numer.shape, label.shape)


        # x_categ, x_numer = x # x_categ: 离散的特征，x_numer：连续的特征

        xs = []
        if self.num_unique_categories > 0:
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)


        if self.num_cont > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)




        x = torch.cat(xs, dim = 1)
        b = x.shape[0]

        if self.use_cls_token:
            cls_tokens = self.cls_token.repeat([b, 1, 1])
            x = torch.cat((cls_tokens, x), dim = 1)


        x = self.transformer(x)


        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = self.pool(x.transpose(-1, -2)).squeeze(-1)


        logit = self.to_logits(x)



        if self.out == 1:
            if self.use_sigmoid:
                logit = torch.sigmoid(logit)
            loss = nn.MSELoss()(logit.reshape(-1), label.float())
        else:
            loss = nn.CrossEntropyLoss()(logit, label)
                

        return logit, loss
