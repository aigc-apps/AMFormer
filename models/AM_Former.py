import torch
import torch.nn.functional as F
from torch import nn, einsum
from models.Attention.blocks import *
from einops import rearrange, repeat


# transformer

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
        qk_relu = False
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
                    sum_or_prod='prod',
                    qk_relu=qk_relu) if use_prod else nn.Identity(),
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
                    sum_or_prod='sum',
                    qk_relu=qk_relu) if token_descent else Attention(heads=heads, dim=dim, dropout=attn_dropout),
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


# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# main class

class FTTransformer(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
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
        token_num: how many token in the input x
        token_descent: use in MUCH-TOKEN dataset
        use_prod: use prod block
        num_special_tokens: =2
        categories: how many different cate in each cate ol
        out: =1 if regressioin else =cls number
        self.num_cont: how many cont col
        num_cont = args.num_cont
        num_cate: how many cate col
        '''
        dim = args.dim
        depth = args.depth
        heads = args.heads
        attn_dropout = args.attn_dropout
        ff_dropout = args.ff_dropout
        self.use_cls_token = args.use_cls_token
        groups = args.groups
        sum_num_per_group = args.sum_num_per_group
        prod_num_per_group = args.prod_num_per_group
        cluster = args.cluster
        target_mode = args.target_mode
        token_num = args.num_cont + args.num_cate
        token_descent = args.token_descent
        use_prod = args.use_prod
        num_special_tokens = args.num_special_tokens
        categories  =  args.categories
        out = args.out
        self.out = out
        self.num_cont = args.num_cont
        num_cont = args.num_cont
        num_cate = args.num_cate
        self.use_sigmoid = args.use_sigmoid
        qk_relu = args.qk_relu


        self.args = args
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_cont > 0, 'input shape must not be null'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        total_tokens = self.num_unique_categories + num_special_tokens + 1

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(args.categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        

        if self.num_cont > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_cont)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_cls_token=self.use_cls_token,
            groups=groups,
            sum_num_per_group=sum_num_per_group,
            prod_num_per_group=prod_num_per_group,
            cluster=cluster,
            target_mode=target_mode,
            token_num=token_num,
            token_descent=token_descent,
            use_prod=use_prod,
            qk_relu=qk_relu,
        )

        # to logits


        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, out)
        )


        self.pool = nn.Linear(num_cont + num_cate, 1)
        

    def model_name(self):
        return 'ft_trans'
    
    def forward(self, x_categ, x_numer, label, step=0, return_attn = False):
        # assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            # x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_cont > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]

        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)

        # attend

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
