###########################
#      LLM v1 Config      #
###########################
from transformers import PretrainedConfig


class LLMv1Config(PretrainedConfig):
    """
    LLM v1 Config

    基于 HuggingFace 的 PretrainedConfig 来定义模型的配置类
    """
    model_type = "llm_v1"

    def __init__(
        self,
        # Tokenizer & Vocab Size
        vocab_size: int = 25600,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 3,
        # Attention
        n_attn_heads: int = 8,                  # 标准注意力中 Q 的头数
        num_kv_heads: int = 2,                  # GQA 中 KV 的头数
        num_hidden_layers: int = 12,            # Transformer Block 数 (Attention + FFN)
        hidden_size: int = 1024,                # 隐藏层维度
        # FFN
        ffn_hidden_size: int = 2048,            # FFN 隐藏层维度
        # RoPE
        max_position_embeddings: int = 32768,   # 最大位置编码长度
        rope_theta: float = 1000000.0,          # RoPE 的基频
        inference_rope_scaling: bool = False,   # 是否外推 RoPE
        # Others
        dropout: float = 0.0,
        **kwargs
    ):  
        # 继承父类, 并将子类接受到的任何参数传递给父类
        super().__init__(**kwargs)
        # Tokenizer & Vocab Size
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        # 这里可以考虑让 pad_token_id = eos_token_id, 可能更符合 LLM 的习惯
        # 手动指定可以避免这个信息提示 (不是错误, HuggingFace 会自动对齐)
        # - The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. 
        # - The model config and generation config were aligned accordingly, being updated with the tokenizer's values. 
        self.pad_token_id = pad_token_id
        # Attention
        self.n_attn_heads = n_attn_heads
        self.num_kv_heads = num_kv_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        # FFN
        self.ffn_hidden_size = ffn_hidden_size
        # RoPE
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling        
        # YaRN 外推长度 = factor * original_max_position_embeddings = 32768
        if self.inference_rope_scaling:
            self.rope_scaling = {
            "beta_fast": 32,                            # 快速频率调整参数
            "beta_slow": 1,                             # 慢速频率调整参数
            "factor": 16,                               # 外推因子
            "original_max_position_embeddings": 2048,   # 训练时的最大长度
            "attention_factor": 1.0,                    # 注意力缩放因子
            "type": "yarn"}
        else:
            self.rope_scaling = None
        # Others
        self.dropout = dropout


###########################
#      LLM v1 Model       #
###########################
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    """
    预计算 RoPE (Rotary Position Embedding) 的频率矩阵
    RoPE 通过旋转矩阵将位置信息编码到 Query 和 Key 中, 使模型能够理解 Token 的相对位置

    - 本函数预计算所有位置的 cos 和 sin 值, 避免在每次前向传播时重复计算
    - 支持 YaRN (Yet another RoPE extensioN) 外推方法, 可以处理超过训练时最大长度的序列

    Args:
    - dim:          每个注意力头的维度 (head_dim)
    - end:          最大序列长度, 默认 (32768 = 32 * 1024)
    - rope_base:    RoPE 的基频率参数 (默认 1e6 -> 1000000.0)
    - rope_scaling: RoPE 外推配置字典 (YaRN 方法), 如果为 None 则不使用外推

    Returns:
    - freqs_cos:    预计算的 cos 值 [end, dim]
    - freqs_sin:    预计算的 sin 值 [end, dim]
    """
    # 1.计算基础频率
    # - RoPE 频率公式: f_i = 1 / (rope_base^(2i/dim))
    # - 其中 i 是维度索引 (0, 2, 4, ..., dim-2), 只使用偶数索引
    # - 频率随维度索引增加而递减, 形成不同频率的旋转
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # 2.应用 YaRN 外推 (如果启用)
    if rope_scaling is not None:
        # 获取 YaRN 配置参数
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),     # 训练时的最大长度
            rope_scaling.get("factor", 16),                                 # 外推因子
            rope_scaling.get("beta_fast", 32.0),                            # 快速频率调整参数
            rope_scaling.get("beta_slow", 1.0),                             # 慢速频率调整参数
            rope_scaling.get("attention_factor", 1.0)                       # 注意力缩放因子
        )
        # 如果目标长度超过训练长度, 应用 YaRN 外推
        if end / orig_max > 1.0:
            # YaRN 公式: f'(i) = f(i) * ((1-γ) + γ/s)
            # - 其中 γ 是线性斜坡函数, s 是缩放因子 (factor)
            # - 对于低频维度 (i < low), 不进行缩放
            # - 对于高频维度 (i > high), 完全缩放
            # - 对于中间维度, 线性插值

            # 计算频率调整的边界维度
            # inv_dim(b) 返回频率为 b 的维度索引
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low =  max(math.floor(inv_dim(beta_fast)), 0)                   # 低频边界
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)         # 高频边界

            # 计算线性斜坡函数 γ
            # 对于维度 i: γ(i) = (i - low) / (high - low), 限制在 [0, 1]
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            
            # 应用 YaRN 缩放: f'(i) = f(i) * ((1-γ) + γ/s)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 3.计算所有位置的频率
    # - 为每个位置计算频率: freqs[pos, dim] = pos * freqs[dim]
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # 4.计算 cos 和 sin 值
    # - 将频率转换为 cos 和 sin 值, 用于旋转矩阵
    # - 由于 RoPE 使用复数旋转, 需要将 dim//2 的频率复制到完整的 dim 维度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor   # [end, dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor   # [end, dim]
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用 Rotary Position Embedding (RoPE) 到 Query 和 Key

    RoPE 通过复数旋转将位置信息编码到 Q 和 K 中
    - R_θ(x) = [x_0*cos(θ) - x_1*sin(θ), x_0*sin(θ) + x_1*cos(θ)]
    - 在实现中, 将复数旋转分解为实部和虚部的线性组合, 使用 rotate_half 函数实现

    Args:
    - q:                Query 张量      [bs, seq_len, n_attn_heads, head_dim]
    - k:                Key   张量      [bs, seq_len, num_kv_heads, head_dim]
    - cos:              预计算的 cos 值     [seq_len, head_dim]
    - sin:              预计算的 sin 值     [seq_len, head_dim]
    - position_ids:     位置索引 (未使用, cos/sin 已包含位置信息)
    - unsqueeze_dim:    在哪个维度插入新维度以匹配 q/k 的形状 (默认1)

    Returns:
    - q_embed:          应用 RoPE 后的 Query    [bs, seq_len, n_attn_heads, head_dim]
    - k_embed:          应用 RoPE 后的 Key      [bs, seq_len, num_kv_heads, head_dim]
    """
    def rotate_half(x):
        """
        旋转向量的后半部分
        将向量分成两半, 交换位置并取反后半部分
        [a, b, c, d] -> [-c, -d, a, b]
        """
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 1.调整形状以匹配 q/k: 
    #   cos.unsqueeze(unsqueeze_dim) / sin.unsqueeze(unsqueeze_dim)
    #   cos/sin shape: [seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    # 2.对 Query 和 Key 分别应用旋转
    #   q_embed = (q * cos) + (rotate_half(q) * sin)
    #   k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

class Attention(torch.nn.Module):
    """
    Attention 模块

    基于 PyTorch SDPA (scaled_dot_product_attention) 实现的注意力模块
    - 支持 GQA
    - 支持 KV Cache 推理加速
    - 支持 门控注意力 (https://arxiv.org/abs/2505.06708)

    https://huggingface.co/docs/transformers/v5.1.0/zh/main_classes/text_generation#transformers.GenerationConfig.use_cache
    - KV Cache & use_cache 不需要手动创建和指定
    - 在使用 model.generate() 方法时会自动设置 use_cache = True
    - 在使用 model.generate() 方法时会自动传递 Cache 对象, 默认为 DynamicCache 类型 (past_key_values)
    """
    def __init__(self, config: LLMv1Config, layer_idx: int = None):
        super().__init__()
        # 1.初始化 & 检查
        assert config.n_attn_heads % config.num_kv_heads == 0,              "请确保 n_attn_heads 能被 num_kv_heads 整除!"
        self.layer_idx = layer_idx                                          # 记录当前是第几层, HuggingFace Cache 对象需要用它来索引
        self.head_dim = config.hidden_size // config.n_attn_heads           # 每个头的维度
        self.hidden_size = config.hidden_size                               # 隐藏层维度
        self.n_attn_heads = config.n_attn_heads                             # 标准注意力 Query 头数
        self.num_kv_heads = config.num_kv_heads                             # GQA Key / Value 头数
        self.dropout = config.dropout                                       # Dropout 的值              
        self.resid_dropout = torch.nn.Dropout(config.dropout)               # 残差 Dropout 层
        # 门控注意力
        # https://arxiv.org/abs/2505.06708
        self.gate_act = torch.nn.Sigmoid()                                             # 门控激活函数
        self.g_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)   # 门控线性层投影
        torch.nn.init.xavier_uniform_(self.g_proj.weight)                              # xavier 正太分布初始化
        # torch.nn.init.constant_(self.g_proj.bias, 0.0)                               # 让门控的 bias 初始为 0, 即接近开放状态 (Sigmoid(0) = 0.5)

        # 2.投影层: 将隐藏状态映射到 Query、Key、Value 矩阵 (标准注意力的 4 个投影矩阵)
        # - Q 投影 hidden_size -> n_attn_heads * head_dim
        self.q_proj = torch.nn.Linear(self.hidden_size, self.n_attn_heads * self.head_dim, bias=False)
        # - K 投影 hidden_size -> num_kv_heads * head_dim
        self.k_proj = torch.nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        # - V 投影 hidden_size -> num_kv_heads * head_dim
        self.v_proj = torch.nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        # - O 投影 n_attn_heads * head_dim -> hidden_size
        self.o_proj = torch.nn.Linear(self.n_attn_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,                                                    # 输入张量 [bs, seq_len, hidden_size]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],             # RoPE 位置编码 (cos, sin) 元组
        past_key_values: Optional[DynamicCache] = None,                     # HuggingFace Cache 对象, 会在 model.generate() 时自动传递
        use_cache: bool = False,                                            # 是否开启 kv 缓存功能, 会在 model.generate() 时自动设置为 True
        attention_mask: Optional[torch.Tensor] = None                       # 注意力掩码矩阵, 形状为 [bs, seq_len]        
    ):
        """
        Return:
        - output:               经过注意力计算的 torch.Tensor: [bs, seq_len, hidden_size]
        - past_key_values:      HuggingFace Cache 对象
        """
        # 1.利用 x 的形状来投影 Q/K/V 矩阵
        bs, seq_len, _ = x.shape                                          
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)            # 获取 Q/K/V 投影矩阵
        # - 将 Q/K/V 矩阵重塑为指定形状 (多头形式)
        q = q.view(bs, seq_len, self.n_attn_heads, self.head_dim)           # [bs, seq_len, n_attn_heads, head_dim]
        k = k.view(bs, seq_len, self.num_kv_heads, self.head_dim)           # [bs, seq_len, num_kv_heads, head_dim]
        v = v.view(bs, seq_len, self.num_kv_heads, self.head_dim)           # [bs, seq_len, num_kv_heads, head_dim]

        # 2.应用 RoPE 位置编码 (对当前的 Q & K 添加位置编码)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3.KV Cache 相关
        # - HF DynamicCache 期望输入形状: [bs, heads, seq_len, head_dim]
        # - 所以这里需要提前转置来适配 SDPA 和 Cache
        q = q.transpose(1, 2)   # [bs, n_attn_heads, seq_len, head_dim]
        k = k.transpose(1, 2)   # [bs, num_kv_heads, seq_len, head_dim]
        v = v.transpose(1, 2)   # [bs, num_kv_heads, seq_len, head_dim]

        if use_cache:
            # https://huggingface.co/docs/transformers/v5.1.0/en/internal/generation_utils#transformers.Cache.update
            # - update 通常接受 [bs, heads, seq_len, head_dim], 返回拼接后的全量 key & value
            # - layer_idx 为用于缓存状态的图层索引
            k, v = past_key_values.update(k, v, layer_idx=self.layer_idx)

        # 4.计算注意力分数
        # - PyTorch 的 PyTorch SDPA (scaled_dot_product_attention) 不支持同时设置 is_causal=True / attn_mask
        # - 因此在 Prefill 阶段如果存在 attn_mask, 则需要手动计算 attn_mask 来融合 causal_mask
        if seq_len > 1:
            # Prefill 阶段          
            if attention_mask is not None:
                is_causal = False
                # 4.1 构建因果掩码 [seq_len, seq_len] (上三角为 -inf)
                causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=q.device), diagonal=1)
                # 4.2 扩展 attention_mask [bs, seq] -> [bs, 1, 1, seq] 用于 padding
                extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # - 将 0 变为 -inf, 1 变为 0
                # - torch.finfo(xq.dtype).min: 返回指定数据类型的最小浮点数
                extended_mask = (1.0 - extended_mask) * torch.finfo(q.dtype).min
                # 4.3 合并掩码矩阵, 此时 mask 同时包含 causal 和 padding 信息
                attention_mask = causal_mask[None, None, :, :] + extended_mask  # [1, 1, seq, seq] + [bs, 1, 1, seq] -> [bs, 1, seq, seq]

            # Prefill 阶段, 但没有 Padding
            else:
                is_causal = True
                attention_mask = None
        
        # Decoding 阶段 (seq_len=1)
        # - seq_len=1 即 Query 为 1, 因此不需要因果掩码
        else:
            is_causal = False
            if attention_mask is not None:
                extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)        # [bs, seq] -> [bs, 1, 1, seq]
                attention_mask = (1.0 - extended_mask) * torch.finfo(q.dtype).min
            else:
                attention_mask = None

        # 使用 PyTorch SDPA (scaled_dot_product_attention) 计算注意力分数
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=is_causal,
            attn_mask = attention_mask,
            enable_gqa=True,
        )
         
        # 5.输出投影
        output = output.transpose(1, 2).reshape(bs, seq_len, -1)                # [bs, seq_len, n_attn_heads * head_dim]
        # 门控路径: 生成通过比例
        gate = self.gate_act(self.g_proj(output))
        # 保留原始信息
        output = gate * output
        # 最终输出投影
        output = self.resid_dropout(self.o_proj(output))                        # [bs, seq_len, hidden_size]

        return output, past_key_values


class FeedForward(torch.nn.Module):
    """
    前反馈神经网络 FFN 模块
    """
    def __init__(self, config: LLMv1Config):
        super().__init__()
        # FFN 三个线形层投影
        self.gate_proj = torch.nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.down_proj = torch.nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)
        self.up_proj = torch.nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        # Dropout 层
        self.dropout = torch.nn.Dropout(config.dropout)
        # 激活函数
        self.act_fun = torch.nn.Mish()
    
    def forward(self, x):
        """
        Args:
        - x:    输入 torch.Tensor: [bs, seq_len, hidden_size]

        Returns:
        - out:  输出 torch.Tensor: [bs, seq_len, hidden_size]
        """
        # gate * up -> down
        out = self.down_proj(self.act_fun(self.gate_proj(x) * self.up_proj(x)))
        out = self.dropout(out)
        return out


class LLMv1Block(torch.nn.Module):
    """
    LLMv1 模型的一个 Transformer 块
    - 包含一个 Attention 模块 (包含 KV Cache 组件)
    - 包含一个 FFN 模块

    https://huggingface.co/docs/transformers/v5.1.0/zh/main_classes/text_generation#transformers.GenerationConfig.use_cache
    - KV Cache & use_cache 不需要手动创建和指定
    - 在使用 model.generate() 方法时会自动设置 use_cache = True
    - 在使用 model.generate() 方法时会自动传递 Cache 对象, 默认为 DynamicCache 类型 (past_key_values)
    """
    def __init__(self, layer_idx: int, config: LLMv1Config):
        """
        Args:
        - layer_idx:    当前层的索引, HuggingFace Cache 需要用这个索引来定位
        - config:       模型的配置对象
        """
        super().__init__()
        # Attention
        self.attn_layer = Attention(config, layer_idx=layer_idx)
        # FFN
        self.ffn = FeedForward(config)
        # Normalization
        self.input_norm = torch.nn.RMSNorm(config.hidden_size)  # 前置 norm
        self.post_norm = torch.nn.RMSNorm(config.hidden_size)   # 后置 norm

    def forward(self, hidden_states, position_embeddings, past_key_values=None, use_cache=False, attention_mask=None):
        """
        Args:
        - hidden_states:           输入 torch.Tensor, 形状为 [bs, seq_len, hidden_size]
        - position_embeddings:     RoPE 位置编码 (cos, sin) 元组, 形状为 Tuple[torch.Tensor, torch.Tensor]
        - past_key_values:         HuggingFace Cache 对象, 会在 model.generate() 时自动传递
        - use_cache:               是否开启 kv 缓存功能, 会在 model.generate() 时自动设置为 True
        - attention_mask:          注意力掩码矩阵, 形状为 [bs, seq_len]

        Returns:
        - hidden_states:           输出 torch.Tensor, 形状为 [bs, seq_len, hidden_size]
        - past_key_values:         HuggingFace Cache 对象
        """
        # 第一个残差连接：(输入层标准化 -> 注意力) + 残差连接
        residual = hidden_states
        hidden_states, past_key_values = self.attn_layer(
            self.input_norm(hidden_states),   # 输入前进行正则化
            position_embeddings,              # RoPE 位置编码
            past_key_values, 
            use_cache, 
            attention_mask
        )
        hidden_states += residual             # 残差连接

        # 第二个残差连接: (注意力后标准化 -> FFN) + 残差连接
        hidden_states = hidden_states + self.ffn(self.post_norm(hidden_states))

        # 返回输出
        return hidden_states, past_key_values
    

class LLMv1Model(torch.nn.Module):
    """
    LLMv1 模型

    Transformer 的 Decoder-Only 架构实现
    负责将输入的 Token IDs 转换为深层的语义特征表示 (Hidden States)

    主要流程:
        Input IDs -> Embeddings -> [Transformer Blocks x L] -> RMSNorm -> Output Hidden States
    """
    def __init__(self, config: LLMv1Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers   # Transformer Block 数 (Attention + FFN)

        # 1.词嵌入层 (Embedding)
        # - 将离散的 Token ID 映射为稠密向量 [vocab_size, hidden_size]
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)

        # 2.堆叠 Transformer Block
        # - 使用 ModuleList 存储 L 个 LLMv1Block
        # - 每个 Block 包含 Attention 和 FFN
        # - 每个 Transformer Block 均有自己的 layer_idx, 用于标记 HuggingFace 的 KV Cache
        self.transformer_layers = torch.nn.ModuleList(
            [LLMv1Block(layer_idx=i, config=config) for i in range(self.num_hidden_layers)]
        )

        # 3.最终归一化层
        self.norm = torch.nn.RMSNorm(config.hidden_size)

        # 4.预计算旋转位置编码 (Precompute RoPE)
        # - 预先计算所有可能位置的 cos 和 sin 值, 避免前向传播时重复计算
        # - freqs_cos/sin 形状: [MaxPos, head_dim]
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.n_attn_heads,
            end=config.max_position_embeddings, 
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )

        # 5.Dropout 层
        self.dropout = torch.nn.Dropout(config.dropout)

        # 6.注册为缓冲区 (buffer)
        # - buffer 不会被视为 模型参数(parameter), 不参与梯度更新, 但会随模型权重文件保存
        # - persistent=False 表示这些值可以根据 config 动态重新计算, 不强制依赖权重文件
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        **kwargs,   # HuggingFace 的 Trainer 需要传入额外参数
    ):
        """
        前向传播逻辑

        Args:
        - input_ids:        输入 token 序列 [bs, seq_len]
        - attention_mask:   注意力掩码, 1 表示有效 token, 0 表示 padding [bs, seq_len]
        - past_key_values:  HuggingFace Cache 对象, 会在 model.generate() 时自动传递
        - use_cache:        是否开启 kv 缓存功能, 会在 model.generate() 时自动设置为 True

        Returns:
        - hidden_states:    模型输出的特征 [bs, seq_len, hidden_size]
        - past_key_values:  HuggingFace Cache 对象
        """
        # 获取输入的 bs 和 seq_len (推理阶段 seq_len 为 1)
        bs, seq_len = input_ids.shape

        # 计算起始位置: 从 DynamicCache 获取当前起始位置
        start_pos = past_key_values.get_seq_length() if past_key_values is not None else 0

        # 词嵌入 + Dropout
        # - 将 token_ids 转换为词嵌入向量: [bs, seq_len] -> [bs, seq_len, hidden_size]
        # - 此时 hidden_states 包含了语义信息, 但还没有位置信息
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 提取位置编码 (RoPE Slicing)
        # - 根据绝对位置 start_pos 和当前长度 seq_len, 从预计算的表中切片
        # - 切片范围: [start_pos : start_pos + seq_len]
            # 场景1 (训练/Prefill):  start_pos=0, seq_len=N -> 取出前 N 个位置编码
            # 场景2 (推理 Decoding): start_pos=N, seq_len=1 -> 仅取出第 N 个位置的编码 (长度为 1)
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len],
        )

        # 逐层前向传播
        # - layer_idx 这里用不到, 实际上已经在初始化阶段就设定好了
        # - 代码如下: [LLMv1Block(layer_idx=i, config=config) for i in range(self.num_hidden_layers)]
        for layer_idx, layer in enumerate(self.transformer_layers):
            hidden_states, past_key_values = layer(
                hidden_states,                      # 当前隐藏状态
                position_embeddings,                # 传入切片好的位置编码
                past_key_values=past_key_values,    # 传入/更新共享的 KV Cache 对象
                use_cache=use_cache,                # 指示 Block 是否需要返回缓存
                attention_mask=attention_mask       # 注意力掩码
            )

        # 最终标准化
        hidden_states = self.norm(hidden_states)

        # 返回最终隐藏状态、缓存 (仅在 use_cache 时返回缓存以兼容 HF 接口)
        return hidden_states, past_key_values if use_cache else None


class LLMv1ForCausalLM(PreTrainedModel, GenerationMixin):
    """
    LLMv1 因果语言模型 (Causal Language Model)

    架构组成:
        Input IDs -> [LLMv1Model] -> Hidden States -> [LM Head] -> Logits
    
    https://huggingface.co/docs/transformers/main_classes/text_generation
    - 这是面向最终任务 (文本生成) 的顶层封装类 (GenerationMixin)
    - 将 LLMv1Model 与 "线性输出头(lm_head)" 组合, 实现完整的文本生成能力
    - 兼容 HuggingFace Transformers 的训练/推理接口 (如 model.generate())
    - 并支持自回归解码、KV 缓存、损失计算等标准功能

    https://huggingface.co/docs/transformers/v5.1.0/zh/main_classes/text_generation#transformers.GenerationConfig.use_cache
    - KV Cache & use_cache 不需要手动创建和指定
    - 在使用 model.generate() 方法时会自动设置 use_cache = True
    - 在使用 model.generate() 方法时会自动传递 Cache 对象, 默认为 DynamicCache 类型 (past_key_values)
    """
    # 指定配置类, Hugging Face 框架自动加载机制需要
    config_class = LLMv1Config
    # 声明共享权重的 key
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LLMv1Config = None):
        # 初始化配置项, 如果未提供则使用默认参数初始化
        self.config = config or LLMv1Config()
        super().__init__(self.config)

        # 1.LLMv1 骨干网络 (Backbone)
        # - 实例化纯 Transformer Decoder, 负责提取深层语义特征
        # - 输入: [bs, seq_Len] -> 输出: [bs, seq_Len, hidden_size]
        self.model = LLMv1Model(self.config)

        # 2.语言模型头 (LM Head)
        # - 这是一个线性投影层 (Linear Layer)
        # - 作用: 将高维特征向量 (Hidden State) 映射回词表空间 (Vocab Space)
        # - 形状: [hidden_Size] -> [vocab_Size]
        # - bias=False: 现代大模型 (LLaMA等) 通常不使用偏置项, 以提升数值稳定性
        self.lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 3.权重共享 (Weight Tying)
        # [重要优化] 将 Input Embedding 的权重指针指向 LM Head 的权重
        # - 物理意义: 语义上, "输入一个词" 和 "预测一个词" 使用的是同一个语义空间
        # - 显存优势: 减少参数量 15 ~ 20%（约 vocab_size * hidden_size）
        self.model.embed_tokens.weight = self.lm_head.weight
        # 此操作需在 super().__init__() 后执行, 因为父类可能初始化参数

        # 4.Fix the no attribute 'all_tied_weights_keys' issue in transformers==5.x
        # https://huggingface.co/ZhengPeng7/BiRefNet/commit/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4
        self.post_init()
    
    def tie_weights(self):
        """
        如果存在共享权重, 需要实现这个方法 (transformers 内部加载时会调用)
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: DynamicCache = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ):
        """
        前向传播 (支持 "训练" 和 "推理" 两种模式)
	
        Args:
        - input_ids:                输入 token 序列 [bs, seq_len]
        - attention_mask:           注意力掩码, 1 表示有效 token, 0 表示 padding [bs, seq_len]
        - labels:                   标签序列 [bs, seq_len]
        - past_key_values:          HuggingFace Cache 对象, 会在 model.generate() 时自动传递
        - use_cache:                是否开启 kv 缓存功能, 会在 model.generate() 时自动设置为 True
        - logits_to_keep:           性能优化参数
                                    - 0 默认: 计算所有 Token 的 Logits (训练时必须选这个)
                                    - 1 常用: 只计算最后一个 Token 的 Logits (推理生成时用)
                                    - 原理: 避免在 lm_head 上进行无用的矩阵乘法计算
	
        Returns:
        - CausalLMOutputWithPast:   包含 loss, logits, hidden_states, past_key_values, aux_loss
        """
        # 防御性代码, 理论上不会执行到这一步, 但这里为了以防万一, 还是在模型最外层创建一个
        # - 在使用 model.generate() 方法时会自动设置 use_cache = True
        # - 在使用 model.generate() 方法时会自动传递 Cache 对象, 默认为 DynamicCache 类型 (past_key_values)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(offloading=True)

        # 1.骨干网络特征提取
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

        # 2.Logits 计算范围优化 (Logits Slicing)
        # - lm_head 的计算量是 O(seq_Len * hidden_size * vocab_size), 非常巨大
        # - 在推理时只需要最后一个词的预测结果, 不需要前文的预测
        if isinstance(logits_to_keep, int):
            # 整数模式
            # logits_to_keep = 1 -> slice(-1, None) -> 取最后 1 个
            # logits_to_keep = 0 -> slice(None)     -> 取全部 (训练时)
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            # 张量模式 (高级用法, 指定特定位置)
            slice_indices = logits_to_keep
        
        # 对 hidden States 进行切片, 只保留需要计算的部分
        # - 推理时: [bs, 100, hidden_size] -> [bs, 1,   hidden_size]
        # - 训练时: [bs, 100, hidden_size] -> [bs, 100, hidden_size]
        sliced_hidden_states = hidden_states[:, slice_indices, :]

        # 3.映射到词表 (Projection)
        # - 执行矩阵乘法: X @ W.T
        # - logits 形状: [bs, sliced_len, vocab_size]
        # - 这里的 logits 是未归一化的概率 (Log-odds)
        logits = self.lm_head(sliced_hidden_states)

        # 4.计算损失 (仅训练模式)
        loss = None
        if labels is not None:
            # 因果语言模型的核心逻辑: "Shift Prediction" (位移预测)
            # 即第 t 个时间步的 Logit, 应该预测第 t+1 个时间步的 Label
            # - 输入: A  B  C  D
            # - 目标: B  C  D  E

            # 去掉最后一个 Logit, 因为预测的是 E, 但 Input 只有到 D   
            # [bs, seq_Len-1, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous()
            # 去掉第一个 Label, 因为 A 之前没有 Logit 预测它
            # [bs, seq_Len-1]
            shift_labels = labels[..., 1:].contiguous()

            # 交叉熵损失 (Cross Entropy)
            # view(-1): 将 batch 和 seq 维度展平, 变成 [total_tokens, vocab_size] 以适配 Loss 函数
            # ignore_index=-100: 忽略标签为 -100 (Padding) 的位置, 不计算梯度
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        # 5.封装输出: 使用 Hugging Face 标准格式返回, 确保兼容性
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

        return output


if __name__ == "__main__":
    # register model
    LLMv1Config.register_for_auto_class()
    LLMv1ForCausalLM.register_for_auto_class()

    # summary model parameters
    from torchinfo import summary
    config = LLMv1Config()
    model = LLMv1ForCausalLM(config)
    summary(model=model, input_data=torch.randint(0, config.vocab_size, (1, 512)))