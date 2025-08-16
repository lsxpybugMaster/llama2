from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 全局配置类
@dataclass
class ModelArgs:
    dim: int = 4096  # llama嵌入维度为4096
    n_layers: int = 32
    n_heads: int = 32 # Q的头数
    n_kv_heads: Optional[int] = None # K,V的头数 使用Group Multiple Query
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # KV cache变量
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int,seq_len: int,device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "单头嵌入维度必须能被2整除"

    # (Head_dim / 2)  [0,2,4,6,...Head_dim - 2]
    theta_numerator = torch.arange(0, head_dim, 2).float()

    # (Head_dim / 2)  θi公式
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # (Seq_len)       [0,1,2,3,4,....,Seq_len - 1]
    m = torch.arange(seq_len,device=device)

    # m.T @ theta (注意仅为示例,m与θ为1D向量)
    # (Seq_len,1) @ (1,Head_dim / 2) => (Seq_len, Head_dim / 2)
    freqs = torch.outer(m,theta).float()

    # 转换为极坐标形式 c = R * exp(m * theta), R = 1 
    # torch.polar(abs = 1,angle = freqs)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):

    # STEP1,2: 每两个合并为1个复数
    # (B, Seq_len , H, Head_dim) -> (B, Seq_len, H, Head_dim/2, 2) 
    # -> 合并复数后: (B, Seq_len , H, Head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # 确保freqs_complex维度匹配
    # (Seq_len, Head_dim/2)    ->  (1, Seq_len, Head_dim/2)
    # (1, Seq_len, Head_dim/2) ->  (1, Seq_len, 1, Head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # STEP3 相乘
    # (B, Seq_len, H, Head_dim/2) * (1, Seq_len, 1, Head_dim/2)
    # => (B, Seq_len, H, Head_dim/2)
    x_rotated = x_complex * freqs_complex

    # STEP4 转换为数对
    # (B, Seq_len, H, Head_dim/2) -> (B, Seq_len, H, Head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)

    # STEP5 Flatten
    # (B, Seq_len, H, Head_dim/2, 2) -> (B, Seq_len, H, Head_dim/2) 
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return(
        # (B, Seq_len, N_KV_heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_len, N_KV_heads, N_Rep , 1, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_len, N_KV_heads * N_Rep , 1, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        # N_KV_heads * N_Rep = N_Q_heads 变为MHA
    )


class SelfAttention(nn.Module):
    def __init__(self,args: ModelArgs):
        super().__init__()

        # 分组查询机制:Q与KV头的数量不同
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q  = args.n_heads
        # 得到的比值指导KV头的复制
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # 仅头的数量不同,所有嵌入维度一致!  4096/32
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # kv-cache
        # 大小: (B, Seq_len, H_kv, Head_dim)
        self.cache_k = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim))

    def forward(
        self,
        x: torch.Tensor, # (B, 1, dim) 仅一行,使用KV-cache
        start_pos: int,
        freqs_complex: torch.Tensor
    ):
        # (B, 1, dim)
        batch_size, seq_len, _ = x.shape

        # (B, 1, dim) -> (B, 1, H_Q * Head_dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, H_KV * Head_dim)
        xk = self.wk(x)
        # (B, 1, dim) -> (B, 1, H_KV * Head_dim)
        xv = self.wv(x)

        # 都只有一行
        # (B, 1, H_Q * Head_dim) -> (B, 1, H_Q , Head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_dim) ->  (B, 1, H_KV , Head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_dim) ->  (B, 1, H_KV , Head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # RoPE 给q,k做旋转
        xq = apply_rotary_embeddings(xq,freqs_complex,device=x.device)
        xk = apply_rotary_embeddings(xk,freqs_complex,device=x.device)

        # 更新kv-cache
        # (B, 1, H_KV , Head_dim) 加入==> (B, Seq_len, H_kv, Head_dim)
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # 更新后取出KV矩阵 (B, Seq_len_now, H_kv, head_dim)
        keys   = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # 实现分组查询的简单方式,根据倍率复制KV权重再按标准MHA计算
        #  (B, Seq_len_now, H_kv, head_dim) ->  (B, Seq_len_now, H_q, head_dim)
        keys = repeat_kv(keys,self.n_rep)
        values = repeat_kv(values,self.n_rep)

        # 准备进行计算
        # (B, 1, H_q, head_dim) -> (B, H_q, 1, head_dim)
        xq = xq.reshape(1,2)
        # (B, Seq_len_now , H_q, head_dim) -> (B, H_q, Seq_len_now, head_dim)
        keys = keys.reshape(1,2)
        values = values.reshape(1,2)

        # (B, H_q, 1, head_dim) @ (B, H_q, head_dim, Seq_len_now) = (B, H_q, 1, Seq_len_now)
        scores = torch.matmul(xq,keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(),dim = -1).type_as(xq)
        # (B, H_q, 1, Seq_len_now) @ (B, H_q, Seq_len_now, head_dim) =  (B, H_q, 1, head_dim)
        output = torch.matmul(scores,values)
        # (B, H_q, 1, head_dim) -> (B, 1, H_q, head_dim) -> (B, 1, H_q*head_dim) = (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size,seq_len,-1))

        return self.wo(output)
    

class RMSNorm(nn.Module):
    def __init__(self,dim: int,eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 缩放参数g,可学习
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_len, dim) * (B, Seq_len, 1) = (B, Seq_len, dim)
        # rsqrt(x) = 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_len, Dim) = (B, Seq_len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)
    

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        # 该倍数用于对齐不同模型架构的隐藏层维度
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        '''
        自动将维度对齐到指定multiple_of的倍数(向上进行最小对齐)
        hidden_dim = 7
        multiple_of = 5
        则公式= 5 * (7 + 5 - 1) / 5 = 10 现在是10的倍数了
        '''
        hidden_dim = args.multiple_of * ( (hidden_dim + args.multiple_of - 1) // args.multiple_of )

        # W部分 
        self.w1 = nn.Linear(args.dim,hidden_dim,bias=False)
        # W2部分
        self.w2 = nn.Linear(hidden_dim,args.dim,bias=False)
        # V部分
        self.w3 = nn.Linear(args.dim,hidden_dim,bias=False)

    def forward(self, x: torch.Tensor):
        # Swish1(xW)  (B,seq_len,dim) -> (B,seq_len,hidden_dim)
        swish = F.silu(self.w1(x))
        # xV          (B,seq_len,dim) -> (B,seq_len,hidden_dim)
        x_V = self.w3(x)
        # Swish1(xW) * xV
        x = swish * x_V
        # (Swish1(xW) * xV)W2
        x = self.w2(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = SelfAttention(args)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(args)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )

        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out



class Transformer(nn.Module):

    def __init__(self,args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "未设置词表大小"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        # 嵌入层
        self.tok_embeddings = nn.Embedding(self.vocab_size,args.dim)

        # N层堆叠的encoder块
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim,eps = args.norm_eps)
        
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # RoPE位置编码
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device = self.args.device
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # KV-cache仅限推理!

        # (B, Seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "KV缓存,Q仅为每次更新的一个token 一次处理一个token!"

        # (B, Seq_len) -> (B, Seq_len , Dim)
        h = self.tok_embeddings(tokens)

        # RoPE编码
        freq_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for layer in self.layers:
            h = layer(h,start_pos,freq_complex)
        
        # RMSNorm
        h = self.norm(h)

        # Linear
        output = self.output(h).float()

        # Softmax 在 loss 中
        return output



