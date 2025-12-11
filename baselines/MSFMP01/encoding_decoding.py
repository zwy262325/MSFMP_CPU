import torch
from torch import nn

from .patch import PatchEmbedding
from .positional_encoding import PositionalEncoding
# from .transformer_layers import TransformerLayers
from .agcrn1 import AVWGCN
import numpy as np  # 新增：用于掩蔽生成
from .augmentations import masked_data, geom_noise_mask_single # 新增：导入augmentations中的掩蔽函数
from .transformer_layers_new import TransformerLayers
from .tools import ContrastiveWeight, AggregationRebuild


class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, compression_ratio,head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        hidden_dim = int(pn * compression_ratio)
        dimension = 32
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        x = self.pooler(x) # [(bs * n_vars) x dimension]
        return x

class Flatten_Head(nn.Module):
    def __init__(self, d_model, patch_size, head_dropout=0):
        super().__init__()
        self.linear = nn.Linear(d_model, patch_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)  # 新增：应用非负约束
        x = self.dropout(x)
        return x

class Mask(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, encoder_depth,
                 decoder_depth, dim_in, dim_out, agcn_embed_dim, cheb_k, num_node, input_len,mask_distribution='geometric', lm=3,positive_nums=1, temperature =0.1, compression_ratio = 0.1, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.mask_distribution = mask_distribution
        self.lm = lm
        self.positive_nums = positive_nums
        self.masked_data = masked_data
        self.encoder_new = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        self.pooler = Pooler_Head(input_len, embed_dim, compression_ratio,head_dropout=dropout)
        self.contrastive = ContrastiveWeight(temperature,positive_nums)
        self.aggregation = AggregationRebuild(temperature, positive_nums)
        self.projection = Flatten_Head(embed_dim, patch_size, head_dropout=dropout)
        self.node_embeddings = nn.Parameter(torch.randn(num_node, agcn_embed_dim), requires_grad=True)
        self.AVWGCN = AVWGCN(dim_in, dim_out, cheb_k, agcn_embed_dim)
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        self.positional_encoding = PositionalEncoding()

    def encoding_decoding(self, long_term_history):
        mid_patches = self.patch_embedding(long_term_history)
        mid_patches = mid_patches.transpose(-1, -2)
        agcrn_hidden_states = self.AVWGCN(mid_patches, self.node_embeddings)
        patches = self.positional_encoding(agcrn_hidden_states)
        x_enc, mask_index = self.masked_data(patches, self.mask_ratio, self.lm, self.positive_nums, distribution='geometric')
        batch_size, num_nodes, _, _ = long_term_history.shape
        p_enc_out = self.encoder_new(x_enc)
        s_enc_out = self.pooler(p_enc_out)
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out)
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out)
        _, seq_len, nums_dim = agg_enc_out.shape
        agg_enc_out = agg_enc_out.reshape(batch_size * (self.positive_nums + 1), num_nodes, seq_len, nums_dim)
        dec_out = self.projection(agg_enc_out)
        dec_out = dec_out.view(batch_size * (self.positive_nums + 1), num_nodes, 1, -1)
        dec_out = dec_out[:batch_size]
        return dec_out, long_term_history, loss_cl

    def encoding(self, long_term_history):
        mid_patches = self.patch_embedding(long_term_history)
        mid_patches = mid_patches.transpose(-1, -2)
        agcrn_hidden_states = self.AVWGCN(mid_patches, self.node_embeddings)
        patches = self.positional_encoding(agcrn_hidden_states)
        batch_size, num_nodes, _, _ = long_term_history.shape
        patches_reshaped = patches.reshape(batch_size * num_nodes, patches.shape[2], patches.shape[3])
        p_enc_out = self.encoder_new(patches_reshaped)
        _, seq_len, nums_dim = p_enc_out.shape
        p_enc_out = p_enc_out.reshape(batch_size, num_nodes, seq_len, nums_dim)
        dec_out = self.projection(p_enc_out)
        dec_out = dec_out.view(batch_size , num_nodes, 1, -1)

        return dec_out


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None,
                epoch: int = None, **kwargs) -> torch.Tensor:
        history_data = history_data.permute(0, 2, 3, 1)  # B, N, 1, L * P
        if self.mode == "pre-train":
            predict_result, original_result, loss_cl = self.encoding_decoding(history_data)
            return predict_result, original_result, loss_cl

        else:
            hidden_states_full = self.encoding(history_data)
            return hidden_states_full
