import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from einops.layers.torch import Rearrange
from functools import partial



__all__ = ['ST_MEM', 'st_mem_vit_small_dec256d4b', 'st_mem_vit_base_dec256d4b']


def get_1d_sincos_pos_embed(embed_dim: int,
                            grid_size: int,
                            temperature: float = 10000,
                            sep_embed: bool = False):
    """Positional embedding for 1D patches.
    """
    assert (embed_dim % 2) == 0, \
        'feature dimension must be multiple of 2 for sincos emb.'
    grid = torch.arange(grid_size, dtype=torch.float32)

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / (temperature ** omega)

    grid = grid.flatten()[:, None] * omega[None, :]
    pos_embed = torch.cat((grid.sin(), grid.cos()), dim=1)
    if sep_embed:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed, torch.zeros(1, embed_dim)],
                              dim=0)
    return pos_embed


class ST_MEM(nn.Module):
    def __init__(self,
                 seq_len: int = 2250,
                 patch_size: int = 75,
                 num_leads: int = 12,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 decoder_embed_dim: int = 256,
                 decoder_depth: int = 4,
                 decoder_num_heads: int = 4,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 norm_layer: nn.Module = nn.LayerNorm,
                 norm_pix_loss: bool = False):
        super().__init__()
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'embed_dim': embed_dim,
                           'depth': depth,
                           'num_heads': num_heads,
                           'decoder_embed_dim': decoder_embed_dim,
                           'decoder_depth': decoder_depth,
                           'decoder_num_heads': decoder_num_heads,
                           'mlp_ratio': mlp_ratio,
                           'qkv_bias': qkv_bias,
                           'norm_layer': str(norm_layer),
                           'norm_pix_loss': norm_pix_loss}
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.num_leads = num_leads
        # --------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = ST_MEM_ViT(seq_len=seq_len,
                                  patch_size=patch_size,
                                  num_leads=num_leads,
                                  width=embed_dim,
                                  depth=depth,
                                  mlp_dim=mlp_ratio * embed_dim,
                                  heads=num_heads,
                                  qkv_bias=qkv_bias)
        self.to_patch_embedding = self.encoder.to_patch_embedding
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # MAE decoder specifics
        self.to_decoder_embedding = nn.Linear(embed_dim, decoder_embed_dim)

        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 2, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([TransformerBlock(input_dim=decoder_embed_dim,
                                                              output_dim=decoder_embed_dim,
                                                              hidden_dim=decoder_embed_dim * mlp_ratio,
                                                              heads=decoder_num_heads,
                                                              dim_head=64,
                                                              qkv_bias=qkv_bias)
                                             for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, patch_size)
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.encoder.pos_embedding.shape[-1],
                                            self.num_patches,
                                            sep_embed=True)
        self.encoder.pos_embedding.data.copy_(pos_embed.float().unsqueeze(0))
        self.encoder.pos_embedding.requires_grad = False

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.num_patches,
                                                    sep_embed=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder.sep_embedding, std=.02)
        torch.nn.init.normal_(self.mask_embedding, std=.02)
        for i in range(self.num_leads):
            torch.nn.init.normal_(self.encoder.lead_embeddings[i], std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, series):
        """
        series: (batch_size, num_leads, seq_len)
        x: (batch_size, num_leads, n, patch_size)
        """
        p = self.patch_size
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b c (n p) -> b c n p', p=p)
        return x

    def unpatchify(self, x):
        """
        x: (batch_size, num_leads, n, patch_size)
        series: (batch_size, num_leads, seq_len)
        """
        series = rearrange(x, 'b c n p -> b c (n p)')
        return series

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: (batch_size, num_leads, n, embed_dim)
        """
        b, num_leads, n, d = x.shape
        len_keep = int(n * (1 - mask_ratio))  ## 마스킹되지 않고 남은 패치 개수

        noise = torch.rand(b, num_leads, n, device=x.device)  # noise in [0, 1]  ## 0에서 11사이의 램덤 노이즈 생성 (무직위로 마스킹할 패치 결정에 사용됨)
                ## (batch_size, num_leads, n) => 배치 내 각 리드별로 시퀀스에 랜덤하게 노이즈 부여됨
                
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove ## noise 값이 작은 순서대로 패치의 인덱스 배치 (작은 패치는 유지, 큰 패치 마스크)
        ids_restore = torch.argsort(ids_shuffle, dim=2) ## 원래 순서로 복원하기 위해 ids_shuffle의 역순서 저장. 나중에 디코딩 과정에서 마스크된 부분 원래 위치에 복원하기 위함.

        # keep the first subset ## 마스킹되지 않은 패치 유지
        ids_keep = ids_shuffle[:, :, :len_keep] ## 마스킹되지 않고 남길 패치의 인덱스
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, d)) ## ids_keep에 해당하는 패치만 추출 (추출된 패치는 마스킹X)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([b, num_leads, n], device=x.device) ## 원래 시퀀스와 동일한 마스크 텐서 생성 (처음 모든 패치 1로 생성 => 마스킹X 부분은 0)
        mask[:, :, :len_keep] = 0 ## 마스킹 X 부분은 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore) ## 마스크를 원래 순서대로 복원 => 마스킹된 패치와 유지된 패치의 순서를 입력 시퀀스와 동일하게 맞춤.

        return x_masked, mask, ids_restore
        ## x_masked: 마스킹되지 않은 패치로 이뤄진 시퀀스
        ## mask: 각 패치가 마스크되었는지 여부 확인하는 바이너리 마스크
        ## ids_restore: 원래 시퀀스로 복원하기 위한 인덱스

    def forward_encoder(self, x, mask_ratio):
        """
        x: (batch_size, num_leads, seq_len) [256, 12, 2250]
        """
        # embed patches
        x = self.to_patch_embedding(x) ## [256, 12, 30, 768]
        b, _, n, _ = x.shape

        # add positional embeddings
        x = x + self.encoder.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = torch.zeros([b, self.num_leads, n], device=x.device)
            ids_restore = torch.arange(n, device=x.device).unsqueeze(0).repeat(b, self.num_leads, 1)

        # apply lead indicating modules
        ## 시작과 끝 알려주는 임베딩
        # 1) SEP embedding
        sep_embedding = self.encoder.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
            ## 구분자 + 원본 시퀀스 + 구분자
            
        # 2) lead embeddings
        n_masked_with_sep = x.shape[2] ## 패치에 해당
        lead_embeddings = torch.stack([self.encoder.lead_embeddings[i] for i in range(self.num_leads)]).unsqueeze(0) ## 리드별 임베딩을 한 배열로 모음(stack) -> unsqueeze 통해서 batch 차원 추가
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n_masked_with_sep, -1) ## unsqueeze(2) 통해서 sequence 차원 추가 -> expand 통해서 배치 크기와 리드 개수, 시퀀스 길이 만큼 리드 임베딩 확장
        x = x + lead_embeddings ## 리드 임베딩 더해줌

        x = rearrange(x, 'b c n p -> b (c n) p')
        for i in range(self.encoder.depth):
            x = getattr(self.encoder, f'block{i}')(x)
        x = self.encoder.norm(x)
        ## 마스킹 되지 않은 부분만 encoder의 입력으로 들어가서 representation을 뽑음

        return x, mask, ids_restore
            ## x: 마스킹되지 않은 부분의 representation 값
            ## mask: 각 패치의 마스킹 여부
            ## ids_restore: 복원 해야할 패치 인덱스

    def forward_decoder(self, x, ids_restore):
        ## x: 마스킹되지 않은 부분의 representation 값 [256, 108, 768]
        ## ids_restore: 복원해야할 패치의 인덱스 [256, 12, 30]
        
        x = self.to_decoder_embedding(x) ## Linear 층 태움. [256, 108, 256]

        # append mask embeddings to sequence
        ## 리드별로 시퀀스 분리 (채널 차원 복구)
        x = rearrange(x, 'b (c n) p -> b c n p', c=self.num_leads) ## [256, 12, 9, 256] (마스킹 X)
        b, _, n_masked_with_sep, d = x.shape ## b=256, n_masked_with_sep=9, d=256
        n = ids_restore.shape[2] # 30
        
        ## 마스크 임베딩 준비
        mask_embeddings = self.mask_embedding.unsqueeze(1) ## 마스크 임베딩에 리드 차원 추가
        mask_embeddings = mask_embeddings.repeat(b, self.num_leads, n + 2 - n_masked_with_sep, 1) ## 배치와 리드에 맞게 확장
        # [256, 12, 23, 256] (마스킹 O)

        ## 구분자 제외한 시퀀스에 마스크 임베딩 추가 (Unshuffle)
        # Unshuffle without SEP embedding
        x_wo_sep = torch.cat([x[:, :, 1:-1, :], mask_embeddings], dim=2) ## 구분자 제외한 시퀀스에 마스크 임베딩 추가 (패치의 구분자 - 구분자 ECG(패치 수) 구분자)
        x_wo_sep = torch.gather(x_wo_sep, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, d)) ## 복원한 순서대로 재 배열
            # [256, 12, 30, 256]
            
        ## 위치 임베딩 및 SEP 임베딩 추가
        # positional embedding and SEP embedding
        x_wo_sep = x_wo_sep + self.decoder_pos_embed[:, 1:n + 1, :].unsqueeze(1) ## 위치 임베딩 추가 # [256, 12, 30, 256]
        left_sep = x[:, :, :1, :] + self.decoder_pos_embed[:, :1, :].unsqueeze(1) ## 왼쪽 구분자에 위치 임베딩 추가
        right_sep = x[:, :, -1:, :] + self.decoder_pos_embed[:, -1:, :].unsqueeze(1) ## 오른쪽 구분자에 위치 임베딩 추가
        x = torch.cat([left_sep, x_wo_sep, right_sep], dim=2) ## 구분자와 시퀀스 합침 # [256, 12, 32, 256]

        # lead-wise decoding
        ## 리드별 디코딩
        x_decoded = []
        for i in range(self.num_leads): # 12
            x_lead = x[:, i, :, :] ## 리드별 데이터 추출 [256, 32, 256]
            for block in self.decoder_blocks: ## 각 리드에 디코더 블록 적용
                x_lead = block(x_lead) ## [256, 32, 256]
            x_lead = self.decoder_norm(x_lead) ## 정규화 [256, 32, 256]
            x_lead = self.decoder_head(x_lead) ## 디코더 헤드 적용 [256, 32, 256]
            x_decoded.append(x_lead[:, 1:-1, :]) ## 구분자 제외하고 저장 [256, 30, 256]
        x = torch.stack(x_decoded, dim=1) ## 리드별 디코딩 결과를 다시 합침
        
        return x

    def forward_loss(self, series, pred, mask):
        """
        series: (batch_size, num_leads, seq_len)
        pred: (batch_size, num_leads, n, patch_size)
        mask: (batch_size, num_leads, n), 0 is keep, 1 is remove,
        """
        ## series 데이터 패치화
        target = self.patchify(series)
        
        ## 픽셀값 정규화 (옵션, default True 같음)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True) ## 패치별 평균
            var = target.var(dim=-1, keepdim=True) ## 패치별 분산
            target = (target - mean) / (var + 1.e-6)**.5 ## 정규화 수행

        ## 예측값과 실제값의 차이 제곱
        loss = (pred - target) ** 2 ## MSE Loss
        
        ## 각 패치에 대한 평균 손실 계산
        loss = loss.mean(dim=-1)  # (batch_size, num_leads, n), mean loss per patch ## 패치 단위로 손실 정규화
        
        ## 마스킹된 패치에 대해서만 손실 적용
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches ## 마스킹된 패치만을 대상으로 손실 계산
        return loss

    def forward(self,
                series,
                mask_ratio=0.75):
        recon_loss = 0
        pred = None
        mask = None

        latent, mask, ids_restore = self.forward_encoder(series, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        recon_loss = self.forward_loss(series, pred, mask)

        return {"loss": recon_loss, "pred": pred, "mask": mask}

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


def st_mem_vit_small_dec256d4b(**kwargs):
    model = ST_MEM(embed_dim=384,
                   depth=12,
                   num_heads=6,
                   decoder_embed_dim=256,
                   decoder_depth=4,
                   decoder_num_heads=4,
                   mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                   **kwargs)
    return model


def st_mem_vit_base_dec256d4b(**kwargs):
    model = ST_MEM(embed_dim=768,
                   depth=12,
                   num_heads=12,
                   decoder_embed_dim=256,
                   decoder_depth=4,
                   decoder_num_heads=4,
                   mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                   **kwargs)
    return model


class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    '''
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PreNorm(nn.Module):
    def __init__(self,
                 dim: int,
                 fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    MLP Module with GELU activation fn + dropout.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 drop_out_rate=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(drop_out_rate),
                                 nn.Linear(hidden_dim, output_dim),
                                 nn.Dropout(drop_out_rate))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == input_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=qkv_bias)

        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, output_dim),
                                        nn.Dropout(drop_out_rate))
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 heads: int = 8,
                 dim_head: int = 32,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        attn = Attention(input_dim=input_dim,
                         output_dim=output_dim,
                         heads=heads,
                         dim_head=dim_head,
                         qkv_bias=qkv_bias,
                         drop_out_rate=drop_out_rate,
                         attn_drop_out_rate=attn_drop_out_rate)
        self.attn = PreNorm(dim=input_dim,
                            fn=attn)
        self.droppath1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        ff = FeedForward(input_dim=output_dim,
                         output_dim=output_dim,
                         hidden_dim=hidden_dim,
                         drop_out_rate=drop_out_rate)
        self.ff = PreNorm(dim=output_dim,
                          fn=ff)
        self.droppath2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.droppath1(self.attn(x)) + x
        x = self.droppath2(self.ff(x)) + x
        return x

class ST_MEM_ViT(nn.Module):
    def __init__(self,
                 seq_len: int,
                 patch_size: int,
                 num_leads: int,
                 num_classes: Optional[int] = None,
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        
        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'num_classes': num_classes if num_classes is not None else 'None',
                           'width': width,
                           'depth': depth,
                           'mlp_dim': mlp_dim,
                           'heads': heads,
                           'dim_head': dim_head,
                           'qkv_bias': qkv_bias,
                           'drop_out_rate': drop_out_rate,
                           'attn_drop_out_rate': attn_drop_out_rate,
                           'drop_path_rate': drop_path_rate}
        self.width = width
        self.depth = depth

        # embedding layers
        num_patches = seq_len // patch_size
        patch_dim = patch_size
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (n p) -> b c n p', p=patch_size),
                                                nn.LayerNorm(patch_dim),
                                                nn.Linear(patch_dim, width),
                                                nn.LayerNorm(width))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(width))
                                                for _ in range(num_leads))

        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(input_dim=width,
                                     output_dim=width,
                                     hidden_dim=mlp_dim,
                                     heads=heads,
                                     dim_head=dim_head,
                                     qkv_bias=qkv_bias,
                                     drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=drop_path_rate_list[i])
            self.add_module(f'block{i}', block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

        # classifier head
        self.head = nn.Identity() if num_classes is None else nn.Linear(width, num_classes)

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def forward_encoding(self, series, lead_num):
        num_leads = series.shape[1]
        if num_leads > len(self.lead_embeddings):
            raise ValueError(f'Number of leads ({num_leads}) exceeds the number of lead embeddings')
        
        x = self.to_patch_embedding(series)

        b, _, n, _ = x.shape
        x = x + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # lead indicating modules
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)

        #lead_embeddings = torch.stack([lead_embedding for lead_embedding in self.lead_embeddings]).unsqueeze(0)
        lead_embeddings = torch.stack([self.lead_embeddings[lead_num-1]]).unsqueeze(0)
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n + 2, -1)
        
        x = x + lead_embeddings

        x = rearrange(x, 'b c n p -> b (c n) p')
        x = self.dropout(x)
        
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        # remove SEP embeddings
        # x.shape -> [32, 384, 768]
        x = rearrange(x, 'b (c n) p -> b c n p', c=num_leads)
        x = x[:, :, 1:-1, :]

        # x.shape -> [32, 12, 30, 768]
        x = torch.mean(x, dim=(1, 2)) # [32, 768]
        
        
        return self.norm(x) # [32, 768]

    def forward(self, series, lead_num=1):
        x = self.forward_encoding(series, lead_num)
        return self.head(x)

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


class SelfAttention(nn.Module):
    def __init__(self, h_size):
        super(SelfAttention, self).__init__()
        self.h_size = h_size
        self.mha = nn.MultiheadAttention(h_size, 8, batch_first=True)
        self.ln = nn.LayerNorm([h_size])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([h_size]),
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, h_size),
        )

    def forward(self, x):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


class SAWrapper(nn.Module):
    def __init__(self, h_size):
        super(SAWrapper, self).__init__()
        self.sa = nn.Sequential(*[SelfAttention(h_size) for _ in range(1)])
        self.h_size = h_size

    def forward(self, x):
        x = self.sa(x.swapaxes(1, 2))
        return x.swapaxes(2, 1)

# U-Net code adapted from: https://github.com/milesial/Pytorch-UNet

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels*2, out_channels)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        #x = self.conv2(x)

        return x

class SegmentUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        
        x1 = self.up(x1)
        #x = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        #x = self.conv2(x)

        return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DiffusionUNet(nn.Module):
    def __init__(self, in_size, channels, device):
        super().__init__()
        self.in_size = in_size
        self.channels = channels
        self.device = device
        
        self.inc_x = DoubleConv(channels, 64)
        self.inc_freq = DoubleConv(channels, 64)

        self.down1_x = Down(64, 128)
        self.down2_x = Down(128, 256)
        self.down3_x = Down(256, 512)
        self.down4_x = Down(512, 1024)
        self.down5_x = Down(1024, 2048 // 2)
        
        self.up1_x = Up(1024, 512)
        self.up2_x = Up(512, 256)
        self.up3_x = Up(256, 128)
        self.up4_x = Up(128, 64)
        self.up5_x = Up(64, 32)

        self.sa1_x = SAWrapper(128)
        self.sa2_x = SAWrapper(256)
        self.sa3_x = SAWrapper(512)
        self.sa4_x = SAWrapper(1024)
        self.sa5_x = SAWrapper(1024)
        
        self.outc_x = OutConv(32, channels)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1).repeat(1, 1, embed_size)

    def forward(self, x, c, t, verbose=False, arch_type="FULL"):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """

        t = t.unsqueeze(-1)

        # Level 1
        x1 = self.inc_x(x)
        x1 = x1 * c["down_conditions"][0]
        
        if verbose == True:
            print("x1 shape: ", x1.shape)
        
        x2 = self.down1_x(x1) + self.pos_encoding(t, 128, 256)
        
        if verbose == True:
            print("x2 shape: ", x2.shape)
        
        # Level 2
        x2 = self.sa1_x(x2)
        x2 = x2 * c["down_conditions"][1]
        x3 = self.down2_x(x2) + self.pos_encoding(t, 256, 128)
        
        if verbose == True:
            print("x3 shape: ", x3.shape)

        # Level 3
        x3 = x3 * c["down_conditions"][2]
        x3 = self.sa2_x(x3)
        x4 = self.down3_x(x3) + self.pos_encoding(t, 512, 64)
        
        if verbose == True:
            print("x4 shape: ", x4.shape)
        
        # Level 4
        x4 = self.sa3_x(x4)
        x4 = x4 * c["down_conditions"][3]
        x5 = self.down4_x(x4) + self.pos_encoding(t, 1024, 32)
        
        if verbose == True:
            print("x5 shape: ", x5.shape)
        
        # Level 5
        x5 = self.sa4_x(x5)
        x5 = x5 * c["down_conditions"][4]
        x6 = self.down5_x(x5) + self.pos_encoding(t, 1024, 16)
        
        if verbose == True:
            print("x6 shape: ", x5.shape)
        
        x6 = self.sa5_x(x6)
        x6 = x6 * c["down_conditions"][5]
        
        # Upward path
        x = self.up1_x(x6, x5) + self.pos_encoding(t, 512, 32)

        if arch_type == "FULL":
            x = x * c["up_conditions"][0]

        x = self.up2_x(x, x4) + self.pos_encoding(t, 256, 64)
        
        if arch_type == "FULL":
            x = x * c["up_conditions"][1]

        x = self.up3_x(x, x3) + self.pos_encoding(t, 128, 128)
        
        if arch_type == "FULL":
            x = x * c["up_conditions"][2]

        x = self.up4_x(x, x2) + self.pos_encoding(t, 64, 256)
        
        if arch_type == "FULL":
            x = x * c["up_conditions"][3]

        x = self.up5_x(x, x1) + self.pos_encoding(t, 32, 512)
        
        if arch_type == "FULL":
            x = x * c["up_conditions"][4]

        output = self.outc_x(x)
    
        return output.view(-1, self.channels, 512)

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim], elementwise_affine=True)
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([embed_dim], elementwise_affine=True),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, c):
        x_ln = self.ln(x.permute(0, 2, 1))
        c_ln = self.ln(c.permute(0, 2, 1))
        attention_value, _ = self.cross_attention(x_ln, c_ln, c_ln)
        attention_value = attention_value + x_ln
        attention_value = self.ff_cross(attention_value) + attention_value
        return attention_value.permute(0, 2, 1)

class DiffusionUNetCrossAttention(nn.Module):
    def __init__(self, in_size, channels, device, num_heads=8):
        super().__init__()
        self.in_size = in_size
        self.channels = channels
        self.device = device
        
        self.inc_x = DoubleConv(channels, 64)
        self.inc_freq = DoubleConv(channels, 64)

        self.down1_x = Down(64, 128)
        self.down2_x = Down(128, 256)
        self.down3_x = Down(256, 512)
        self.down4_x = Down(512, 1024)
        self.down5_x = Down(1024, 2048 // 2)
        
        self.up1_x = Up(1024, 512)
        self.up2_x = Up(512, 256)
        self.up3_x = Up(256, 128)
        self.up4_x = Up(128, 64)
        self.up5_x = Up(64, 32)
        
        self.cross_attention_down1 = CrossAttentionBlock(64, num_heads)
        self.cross_attention_down2 = CrossAttentionBlock(128, num_heads)
        self.cross_attention_down3 = CrossAttentionBlock(256, num_heads)
        self.cross_attention_down4 = CrossAttentionBlock(512, num_heads)
        self.cross_attention_down5 = CrossAttentionBlock(1024, num_heads)
        self.cross_attention_down6 = CrossAttentionBlock(1024, num_heads)

        self.cross_attention_up1 = CrossAttentionBlock(512, num_heads)
        self.cross_attention_up2 = CrossAttentionBlock(256, num_heads)
        self.cross_attention_up3 = CrossAttentionBlock(128, num_heads)
        self.cross_attention_up4 = CrossAttentionBlock(64, num_heads)
        self.cross_attention_up5 = CrossAttentionBlock(32, num_heads)

        self.outc_x = OutConv(32, channels)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1).repeat(1, 1, embed_size)
    
    def forward(self, x, c, t, verbose=False):
        """
        Model is U-Net with added positional encodings and cross-attention layers.
        """
        t = t.unsqueeze(-1)

        # Level 1
        x1 = self.inc_x(x)
        x1 = self.cross_attention_down1(x1, c["down_conditions"][0])
        
        if verbose == True:
            print("x1 shape: ", x1.shape)
        
        x2 = self.down1_x(x1) + self.pos_encoding(t, 128, x1.shape[-1] // 2)
        x2 = self.cross_attention_down2(x2, c["down_conditions"][1])

        if verbose == True:
            print("x2 shape: ", x2.shape)
        
        # Level 2
        x3 = self.down2_x(x2) + self.pos_encoding(t, 256, x1.shape[-1] // 4)
        x3 = self.cross_attention_down3(x3, c["down_conditions"][2])
        
        if verbose == True:
            print("x3 shape: ", x3.shape)

        # Level 3
        x4 = self.down3_x(x3) + self.pos_encoding(t, 512, x1.shape[-1] // 8)
        x4 = self.cross_attention_down4(x4, c["down_conditions"][3])

        if verbose == True:
            print("x4 shape: ", x4.shape)
        
        # Level 4
        x5 = self.down4_x(x4) + self.pos_encoding(t, 1024, x1.shape[-1] // 16)
        x5 = self.cross_attention_down5(x5, c["down_conditions"][4])

        if verbose == True:
            print("x5 shape: ", x5.shape)
        
        # Level 5
        x6 = self.down5_x(x5) + self.pos_encoding(t, 1024, x1.shape[-1] // 32)
        x6 = self.cross_attention_down6(x6, c["down_conditions"][5])
        
        if verbose == True:
            print("x6 shape: ", x6.shape)
        
        # Upward path
        x = self.up1_x(x6, x5) + self.pos_encoding(t, 512, x1.shape[-1] // 16)
        x = self.cross_attention_up1(x, c["up_conditions"][0])

        x = self.up2_x(x, x4) + self.pos_encoding(t, 256, x1.shape[-1] // 8)
        x = self.cross_attention_up2(x, c["up_conditions"][1])

        x = self.up3_x(x, x3) + self.pos_encoding(t, 128, x1.shape[-1] // 4)
        x = self.cross_attention_up3(x, c["up_conditions"][2])

        x = self.up4_x(x, x2) + self.pos_encoding(t, 64, x1.shape[-1] // 2)
        x = self.cross_attention_up4(x, c["up_conditions"][3])

        x = self.up5_x(x, x1) + self.pos_encoding(t, 32, x1.shape[-1])
        x = self.cross_attention_up5(x, c["up_conditions"][4])

        output = self.outc_x(x)

        return output.view(-1, self.channels, output.shape[-1])

class ConditionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        
        self.inc_c = DoubleConv(1, 64)
        self.inc_freq = DoubleConv(1, 64)

        self.down1_c = Down(64, 128)
        self.down2_c = Down(128, 256)
        self.down3_c = Down(256, 512)
        self.down4_c = Down(512, 1024)
        self.down5_c = Down(1024, 2048 // 2)
        
        self.up1_c = SegmentUp(1024, 512)
        self.up2_c = SegmentUp(512, 256)
        self.up3_c = SegmentUp(256, 128)
        self.up4_c = SegmentUp(128, 64)
        self.up5_c = SegmentUp(64, 32)

    def forward(self, x, verbose=False):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """

        # Level 1

        d1 = self.inc_c(x)
        d2 = self.down1_c(d1)

        if verbose==True:
            print("d2: ", d2.shape)
        
        d3 = self.down2_c(d2)

        if verbose==True:
            print("d3: ", d3.shape)

        d4 = self.down3_c(d3)

        if verbose==True:
            print("d4: ", d4.shape)

        d5 = self.down4_c(d4)

        if verbose==True:
            print("d5: ", d5.shape)
        
        d6 = self.down5_c(d5)

        if verbose==True:
            print("d6: ", d6.shape)
        
        u1 = self.up1_c(d6)
        
        if verbose==True:
            print("u1: ", u1.shape)
        
        u2 = self.up2_c(u1)
        
        if verbose==True:
            print("u2: ", u2.shape)
        
        u3 = self.up3_c(u2)
        
        if verbose==True:
            print("u3: ", u3.shape)

        u4 = self.up4_c(u3)
        
        if verbose==True:
            print("u4: ", u4.shape)

        u5 = self.up5_c(u4)
        
        if verbose==True:
            print("u5: ", u5.shape)

        return {
            "down_conditions": [d1, d2, d3, d4, d5, d6],
            "up_conditions": [u1, u2, u3, u4, u5],
        }


# class ConditionNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.device = "cuda"
        
#         self.inc_c = DoubleConv(1, 64) # DoubleConv(in channel, 64)
#         self.inc_freq = DoubleConv(32, 64)

#         self.down1_c = Down(64, 128)
#         self.down2_c = Down(128, 256)
#         self.down3_c = Down(256, 512)
#         self.down4_c = Down(512, 1024)
#         self.down5_c = Down(1024, 2048 // 2)
        
#         self.up1_c = SegmentUp(1024, 512)
#         self.up2_c = SegmentUp(512, 256)
#         self.up3_c = SegmentUp(256, 128)
#         self.up4_c = SegmentUp(128, 64)
#         self.up5_c = SegmentUp(64, 32)

#     def forward(self, x, verbose=False):
#         """
#         Model is U-Net with added positional encodings and self-attention layers.
#         """

#         # Level 1

#         d1 = self.inc_c(x)
#         d2 = self.down1_c(d1)

#         if verbose==True:
#             print("d2: ", d2.shape)
        
#         d3 = self.down2_c(d2)

#         if verbose==True:
#             print("d3: ", d3.shape)

#         d4 = self.down3_c(d3)

#         if verbose==True:
#             print("d4: ", d4.shape)

#         d5 = self.down4_c(d4)

#         if verbose==True:
#             print("d5: ", d5.shape)
        
#         d6 = self.down5_c(d5)

#         if verbose==True:
#             print("d6: ", d6.shape)
        
#         u1 = self.up1_c(d6)
        
#         if verbose==True:
#             print("u1: ", u1.shape)
        
#         u2 = self.up2_c(u1)
        
#         if verbose==True:
#             print("u2: ", u2.shape)
        
#         u3 = self.up3_c(u2)
        
#         if verbose==True:
#             print("u3: ", u3.shape)

#         u4 = self.up4_c(u3)
        
#         if verbose==True:
#             print("u4: ", u4.shape)

#         u5 = self.up5_c(u4)
        
#         if verbose==True:
#             print("u5: ", u5.shape)

#         return {
#             "down_conditions": [d1, d2, d3, d4, d5, d6],
#             "up_conditions": [u1, u2, u3, u4, u5],
#         }


    
class ConditionNetWithFFT(nn.Module):
    def __init__(self, device="cuda"): # device 파라미터 추가
        super().__init__()
        self.device = device # device 속성 설정

        # Time domain path
        self.inc_time = DoubleConv(1, 32)  # Output channels reduced to 32 to match FFT path before concatenation
        self.down1_time = Down(32, 64)
        self.down2_time = Down(64, 128)
        self.down3_time = Down(128, 256)
        self.down4_time = Down(256, 512)
        self.down5_time = Down(512, 512) # Last down_time to match d6_combined channels

        # Frequency domain path
        self.inc_freq = DoubleConv(1, 32) # Output channels reduced to 32
        self.down1_freq = Down(32, 64)
        self.down2_freq = Down(64, 128)
        self.down3_freq = Down(128, 256)
        self.down4_freq = Down(256, 512)
        self.down5_freq = Down(512, 512) # Last down_freq to match d6_combined channels

        # Combined path (after concatenating time and frequency features)
        # Channels are doubled because of concatenation
        self.down1_c = Down(64, 128)    # Input: 32(time)+32(freq) = 64
        self.down2_c = Down(128, 256)
        self.down3_c = Down(256, 512)
        self.down4_c = Down(512, 1024)
        self.down5_c = Down(1024, 1024) # Kept 1024 to match d6_combined

        # Upsampling path
        self.up1_c = SegmentUp(1024, 512)
        self.up2_c = SegmentUp(512, 256)
        self.up3_c = SegmentUp(256, 128)
        self.up4_c = SegmentUp(128, 64)
        self.up5_c = SegmentUp(64, 32) # Final up_condition channel size

    def forward(self, x_time, x_fft=None, verbose=False):
        """
        Input:
        - x_time: Time domain signal [B, 1, T]
        - x_fft: Frequency domain signal [B, 1, F] (magnitude of complex numbers) - Optional
        """
        B = x_time.shape[0]

        if x_fft is None:
            if verbose:
                print(f"x_time shape for rfft: {x_time.squeeze(1).shape}")
            
            # Perform FFT on the time domain signal if x_fft is not provided
            # Ensure x_time is [B, T] for rfft's last dimension processing
            x_fft_calc = torch.abs(torch.fft.rfft(x_time.squeeze(1), dim=1, norm="ortho"))
            
            if verbose:
                print(f"x_fft_calc after rfft shape: {x_fft_calc.shape}")
            
            x_fft_calc = x_fft_calc.unsqueeze(1)  # Reshape to [B, 1, F]
            
            if verbose:
                print(f"x_fft_calc after unsqueeze shape: {x_fft_calc.shape}")
            
            # Interpolate to match the original signal length T
            x_fft = F.interpolate(x_fft_calc, size=x_time.shape[2], mode='linear', align_corners=False)
            if verbose:
                print(f"x_fft after interpolate shape: {x_fft.shape}")
        elif verbose:
            print(f"Using provided x_fft with shape: {x_fft.shape}")
            # If x_fft is provided, ensure it has the correct shape [B, 1, T]
            if x_fft.shape[2] != x_time.shape[2]:
                 x_fft = F.interpolate(x_fft, size=x_time.shape[2], mode='linear', align_corners=False)
                 if verbose:
                    print(f"Interpolated provided x_fft to shape: {x_fft.shape}")


        # Initial convolution for time and frequency paths
        t_d1 = self.inc_time(x_time) # [B, 32, T]
        f_d1 = self.inc_freq(x_fft)   # [B, 32, T]

        if verbose:
            print(f"t_d1 shape: {t_d1.shape}, f_d1 shape: {f_d1.shape}")

        # Concatenate initial features
        d1_combined = torch.cat([t_d1, f_d1], dim=1) # [B, 64, T]
        if verbose:
            print(f"d1_combined shape: {d1_combined.shape}")

        # Downsampling path using combined features
        d2_combined = self.down1_c(d1_combined) # [B, 128, T/2]
        if verbose: print(f"d2_combined shape: {d2_combined.shape}")
        d3_combined = self.down2_c(d2_combined) # [B, 256, T/4]
        if verbose: print(f"d3_combined shape: {d3_combined.shape}")
        d4_combined = self.down3_c(d3_combined) # [B, 512, T/8]
        if verbose: print(f"d4_combined shape: {d4_combined.shape}")
        d5_combined = self.down4_c(d4_combined) # [B, 1024, T/16]
        if verbose: print(f"d5_combined shape: {d5_combined.shape}")
        d6_combined = self.down5_c(d5_combined) # [B, 1024, T/32]
        if verbose: print(f"d6_combined shape: {d6_combined.shape}")

        # Upsampling path
        u1 = self.up1_c(d6_combined) # [B, 512, T/16]
        if verbose: print(f"u1 shape: {u1.shape}")
        u2 = self.up2_c(u1)          # [B, 256, T/8]
        if verbose: print(f"u2 shape: {u2.shape}")
        u3 = self.up3_c(u2)          # [B, 128, T/4]
        if verbose: print(f"u3 shape: {u3.shape}")
        u4 = self.up4_c(u3)          # [B, 64, T/2]
        if verbose: print(f"u4 shape: {u4.shape}")
        u5 = self.up5_c(u4)          # [B, 32, T]
        if verbose: print(f"u5 shape: {u5.shape}")

        return {
            "down_conditions": [d1_combined, d2_combined, d3_combined, d4_combined, d5_combined, d6_combined],
            "up_conditions": [u1, u2, u3, u4, u5],
        }
        
class NaiveDDPM(nn.Module):
    def __init__(
        self,
        eps_model,
        betas,
        n_T,
        criterion = nn.MSELoss(),
    ):
        super(NaiveDDPM, self).__init__()
        self.eps_model = eps_model
        self.n_T = n_T
        self.eta = 0
        self.beta1 = betas[0]
        self.beta_diff = betas[1] - betas[0]
        ## register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedule(self.beta1, self.beta1 + self.beta_diff, n_T).items():
            self.register_buffer(k, v)

        self.criterion = criterion

    def forward(self, x=None, cond=None, mode="train", window_size=128*5):

        if mode == "train":
            
            _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device) 

            eps = torch.randn_like(x)
            
            x_t = (
                self.sqrtab[_ts, None, None] * x
                + self.sqrtmab[_ts, None, None] * eps
            )  

            return self.criterion(eps, self.eps_model(x_t, cond, _ts / self.n_T))

        elif mode == "sample":

            n_sample = cond["down_conditions"][-1].shape[0]
            device = cond["down_conditions"][-1].device
            
            x_i = torch.randn(n_sample, 1, window_size).to(device)

            for i in range(self.n_T, 0, -1):
                
                z = torch.randn(n_sample, 1, window_size).to(device) if i > 1 else 0

                eps = self.eps_model(x_i, cond, torch.tensor(i / self.n_T).to(device).repeat(n_sample))
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

            return x_i
            
if __name__ == "__main__":

    device = "cuda:0"

    x = torch.randn(2, 1, 128*30).to(device)
    c = torch.randn(2, 1, 128*30).to(device)
    ts = torch.randint(0, 100, [2]).to(device)

    model = DiffusionUNetCrossAttention(512, 1, device=device).to(device)

    conditions = ConditionNet().to(device)(c)

    print(model(x, conditions, ts, verbose=True).shape)