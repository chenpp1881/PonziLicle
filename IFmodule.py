import torch
import torch.nn as nn
from Multihead_Attention import DecoderAttention
from SubLayerConnection import SublayerConnection
from transformers import T5EncoderModel
import random

class GatedFusion(nn.Module):
    def __init__(self, input_dim_code, input_dim_quali, hidden_dim):
        super().__init__()
        self.fc_code = nn.Linear(input_dim_code, hidden_dim)
        self.fc_quali = nn.Linear(input_dim_quali, hidden_dim)
        self.gate_layer = nn.Sigmoid()  # 产生门控权重

    def forward(self, code_repr, quali_repr):
        h_code = self.fc_code(code_repr)  # [B, H]
        h_quali = self.fc_quali(quali_repr)  # [B, H]

        gate = self.gate_layer(h_code + h_quali)  # [B, H]
        fusion = gate * h_code + (1 - gate) * h_quali  # Gated blending

        return fusion  # [B, H]


# class InformationFusionBlock(nn.Module):
#
#     def __init__(self, hidden, dropout, args):
#         super().__init__()
#         self.args = args
#         self.num_layers = 5
#         self.self_attention = nn.ModuleList()
#         self.sublayer_connection1 = nn.ModuleList()
#         self.linear_layers = nn.ModuleList()
#         self.codeT5 = T5EncoderModel.from_pretrained(args.code_model_path)
#         self.fc = nn.Linear(hidden, 2)
#         self.fusion = GatedFusion(hidden, 5, hidden)
#         for _layer in range(self.num_layers):
#             self.self_attention.append(DecoderAttention(d_model=hidden))
#             self.sublayer_connection1.append(SublayerConnection(size=hidden, dropout=dropout))
#             self.linear_layers.append(nn.Linear(in_features=hidden, out_features=hidden))
#
#     def forward(self, code_tokens, desc_tokens, qualitative_tensor):
#
#         code_attention_mask = code_tokens['attention_mask']  # (batch_size, code_seq_len)
#         code_embeddings = self.codeT5(**code_tokens).last_hidden_state  # (batch_size, code_seq_len, hidden_dim)
#
#         ini_emb = code_embeddings
#         desc_embeddings = []
#         desc_attention_masks = []
#
#         for des in desc_tokens:
#             desc_embeddings.append(self.codeT5(**des).last_hidden_state)
#             desc_attention_masks.append(des['attention_mask'])
#
#         combined = list(zip(desc_embeddings, desc_attention_masks))
#         random.shuffle(combined)
#         desc_embeddings, desc_attention_masks = zip(*combined)
#
#         for layer_idx in range(len(desc_embeddings)):
#             desc_attention_mask = desc_attention_masks[layer_idx]
#             attention_mask = torch.bmm(
#                 code_attention_mask.unsqueeze(2).float(),
#                 desc_attention_mask.unsqueeze(1).float()
#             )
#
#             code_embeddings = self.sublayer_connection1[layer_idx](
#                 code_embeddings,
#                 lambda _code_embeddings: self.self_attention[layer_idx](
#                     _code_embeddings, desc_embeddings[layer_idx], desc_embeddings[layer_idx], attention_mask
#                 )
#             )
#             code_embeddings = self.linear_layers[layer_idx](code_embeddings)
#
#         mask = code_attention_mask.unsqueeze(-1).expand(code_embeddings.size()).float()
#         masked_embeddings = code_embeddings * mask
#         sum_embeddings = masked_embeddings.sum(dim=1)
#         valid_token_counts = mask.sum(dim=1)
#         mean_embeddings = sum_embeddings / valid_token_counts
#
#         masked_ini_emb = ini_emb * mask
#         sum_ini_emb = masked_ini_emb.sum(dim=1)
#         mean_ini_emb = sum_ini_emb / valid_token_counts
#
#         code_repr = mean_embeddings + mean_ini_emb
#
#
#         # --------------------------------------------------------
#         quali_float = qualitative_tensor.float()  # [B, 5]
#         fused = self.fusion(code_repr, quali_float)  # [B, fusion_dim]
#         logits = self.fc(fused)
#
#         # ---------------------------------------------------
#         # logits = self.fc(code_repr)
#
#         return logits


class InformationFusionBlock(nn.Module):

    def __init__(self, hidden, dropout, args):
        super().__init__()
        self.args = args
        self.num_layers = 5
        self.self_attention = nn.ModuleList()
        self.sublayer_connection1 = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.codeT5 = T5EncoderModel.from_pretrained(args.code_model_path)
        self.fc = nn.Linear(hidden, 2)
        self.fusion = GatedFusion(hidden, 5, hidden)
        for _layer in range(self.num_layers):
            self.self_attention.append(DecoderAttention(d_model=hidden))
            self.sublayer_connection1.append(SublayerConnection(size=hidden, dropout=dropout))
            self.linear_layers.append(nn.Linear(in_features=hidden, out_features=hidden))

    def forward(self, code_tokens, desc_tokens, qualitative_tensor):

        code_attention_mask = code_tokens['attention_mask']  # (batch_size, code_seq_len)
        code_embeddings = self.codeT5(**code_tokens).last_hidden_state  # (batch_size, code_seq_len, hidden_dim)

        ini_emb = code_embeddings
        desc_embeddings = []
        desc_attention_masks = []

        for des in desc_tokens:
            desc_embeddings.append(self.codeT5(**des).last_hidden_state)
            desc_attention_masks.append(des['attention_mask'])

        combined = list(zip(desc_embeddings, desc_attention_masks))
        random.shuffle(combined)
        desc_embeddings, desc_attention_masks = zip(*combined)

        for layer_idx in range(len(desc_embeddings)):
            desc_attention_mask = desc_attention_masks[layer_idx]
            attention_mask = torch.bmm(
                code_attention_mask.unsqueeze(2).float(),
                desc_attention_mask.unsqueeze(1).float()
            )

            code_embeddings = self.sublayer_connection1[layer_idx](
                code_embeddings,
                lambda _code_embeddings: self.self_attention[layer_idx](
                    _code_embeddings, desc_embeddings[layer_idx], desc_embeddings[layer_idx], attention_mask
                )
            )
            code_embeddings = self.linear_layers[layer_idx](code_embeddings)

        mask = code_attention_mask.unsqueeze(-1).expand(code_embeddings.size()).float()
        masked_embeddings = code_embeddings * mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_token_counts = mask.sum(dim=1)
        mean_embeddings = sum_embeddings / valid_token_counts

        masked_ini_emb = ini_emb * mask
        sum_ini_emb = masked_ini_emb.sum(dim=1)
        mean_ini_emb = sum_ini_emb / valid_token_counts

        code_repr = mean_embeddings + mean_ini_emb

        # --------------------------------------------------------
        quali_float = qualitative_tensor.float()  # [B, 5]
        fused = self.fusion(code_repr, quali_float)  # [B, fusion_dim]
        logits = self.fc(fused)

        # ---------------------------------------------------
        # logits = self.fc(code_repr)

        return logits