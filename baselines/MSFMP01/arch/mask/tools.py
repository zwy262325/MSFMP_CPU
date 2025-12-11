import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

plt.switch_backend('agg')

class ContrastiveWeight(nn.Module):

    def __init__(self, temperature, positive_nums):
        super(ContrastiveWeight, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='sum')
        self.positive_nums = positive_nums

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)
        B_D, _, _ = similarity_matrix.shape
        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(self.positive_nums + 1):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)

            ll = np.expand_dims(ll, axis=0)
            lr = np.expand_dims(lr, axis=0)
            ll = np.repeat(ll, B_D, axis=0)
            lr = np.repeat(lr, B_D, axis=0)

            positives_mask += ll
            positives_mask += lr

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        positives_mask[..., mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[..., mask] = 0

        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape

        # 1.计算序列的相似度矩阵batch_emb_om(7*8,128),similarity_matrix(56,56)
        norm_emb = F.normalize(batch_emb_om, dim=2)
        new_norm_emb = norm_emb.transpose(0, 1)
        similarity_matrix = torch.matmul(new_norm_emb, new_norm_emb.transpose(-2, -1))

        # 2.获取正样本和负样本的掩码,true表示第i个序列和第j个序列是正样本对, 布尔类型positives_mask(56,56), negatives_mask(56,56)
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0])

        # 3.根据掩码获取正样本和负样本的相似度值,positives(56,3), negatives(56,52), 缺少的一个是样本本身
        positives = similarity_matrix[positives_mask].view(cur_batch_shape[1], cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[1], cur_batch_shape[0], -1)

        # 4.合并正负样本相似度, 每行包含: [正1, 正2, 正3, 负1, 负2, ..., 负52] logits(56,55)
        logits = torch.cat((positives, negatives), dim=-1)
        # 5.构建标签, 前面3个为正样本，后面52个为负样本  y_true(56,55)
        y_true = torch.cat((torch.ones(cur_batch_shape[1], cur_batch_shape[0], positives.shape[-1]), torch.zeros(cur_batch_shape[1], cur_batch_shape[0], negatives.shape[-1])), dim=-1).to(batch_emb_om.device).float()

        # 6.对相似度除以温度系数后取log_softmax 得到概率分布的对数形式
        predict = self.log_softmax(logits / self.temperature)
        # 7.使用KL散度损失,使预测分布与理想分布接近
        loss = self.kl(predict, y_true)

        normalization_term = cur_batch_shape[1] * cur_batch_shape[0] * positives.shape[-1]
        loss = loss / normalization_term

        return loss, similarity_matrix, logits, positives_mask


class AggregationRebuild(torch.nn.Module):

    def __init__(self, temperature, positive_nums):
        super(AggregationRebuild, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.positive_nums = positive_nums

    def forward(self, similarity_matrix, batch_emb_om):

        cur_batch_shape = batch_emb_om.shape

        # 1.缩放相似度similarity_matrix(56,56)
        similarity_matrix /= self.temperature

        # 2.排除自身相似度影响(对角线置为极小值) rebuild_weight_matrix(56,56)
        eye_mask = torch.eye(cur_batch_shape[0], device=similarity_matrix.device, dtype=similarity_matrix.dtype)
        eye_mask = eye_mask.unsqueeze(0)
        similarity_matrix = similarity_matrix - eye_mask * 1e12

        rebuild_weight_matrix = self.softmax(similarity_matrix)

        # 3.重塑三维batch_emb_om(56,48,128)->二维batch_emb_om(56,6144)
        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[1],cur_batch_shape[0], -1)

        # 4.矩阵乘法加权重建batch_emb_om(56,6144)
        rebuild_batch_emb = torch.bmm(rebuild_weight_matrix, batch_emb_om)

        # 5.恢复为原来的三维形状
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1],cur_batch_shape[2], -1)

        return rebuild_weight_matrix, rebuild_oral_batch_emb

# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if torch.__version__ >= '1.5.0' else 2
#         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
#                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_in', nonlinearity='leaky_relu')
#
#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#         return x
#
# class DataEmbedding(nn.Module):
#     def __init__(self, c_in, d_model, dropout=0.1):
#         super(DataEmbedding, self).__init__()
#         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x, x_mark=None):
#         x = self.value_embedding(x)
#         return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.d_model = d_model
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        B_D, N, L, C = x.shape
        x_reshaped = x.reshape(B_D * N, L, C)
        x_conv = self.tokenConv(x_reshaped.permute(0, 2, 1))
        x_out = x_conv.transpose(1, 2).reshape(B_D, N, L, self.d_model)
        return x_out


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x)
        return self.dropout(x)