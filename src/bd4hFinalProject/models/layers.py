import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Embeddings(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_position_size,
        max_segment,
        word_emb_padding_idx,
        dropout_rate=0.1,
    ):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=word_emb_padding_idx
        )
        self.position_embeddings = nn.Embedding(
            max_position_size, hidden_size, padding_idx=max_position_size - 1
        )
        self.segment_embeddings = nn.Embedding(
            max_segment, hidden_size, padding_idx=max_segment - 1
        )

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.max_seq_len = max_position_size
        self.padding_idx_list = [
            word_emb_padding_idx,
            max_position_size - 1,
            max_segment - 1,
        ]

    def padding(self, input_ids, padding_idx):
        if input_ids.shape[1] < self.max_seq_len:
            input_ids = F.pad(
                input_ids[0],
                (0, self.max_seq_len - input_ids.shape[1]),
                "constant",
                padding_idx,
            ).unsqueeze(0)
        padding_mask = input_ids == padding_idx
        return input_ids, padding_mask

    def forward(self, input_ids_list):
        input_ids_list = [
            idx[:, 1:] if i > 0 else idx for i, idx in enumerate(input_ids_list)
        ]
        seq_length = [idx.shape[1] for idx in input_ids_list]
        position_ids = [
            torch.arange(
                seq_length[i], dtype=torch.long, device=input_ids_list[i].device
            )
            for i in range(len(seq_length))
        ]
        position_ids = torch.cat(
            [position_ids[i].unsqueeze(0) for i in range(len(position_ids))], dim=-1
        ).expand_as(torch.cat(input_ids_list, dim=-1))
        segment_ids = [
            torch.zeros(
                seq_length[i], dtype=torch.long, device=input_ids_list[i].device
            )
            + k
            for k, i in enumerate(range(len(seq_length)))
        ]
        segment_ids = torch.cat(
            [segment_ids[i].unsqueeze(0) for i in range(len(segment_ids))], dim=-1
        ).expand_as(torch.cat(input_ids_list, dim=-1))
        input_ids_list = torch.cat(input_ids_list, dim=-1)
        input_ids_list, padding_mask = self.padding(
            input_ids_list, self.padding_idx_list[0]
        )
        position_ids, _ = self.padding(position_ids, self.padding_idx_list[1])
        segment_ids, _ = self.padding(segment_ids, self.padding_idx_list[2])

        words_embeddings = self.word_embeddings(input_ids_list)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        embeddings = words_embeddings + position_embeddings + segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, padding_mask


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layer,
        hidden_size,
        num_attention_heads,
        vocab_size,
        max_position_size,
        max_segment,
        word_emb_padding_idx,
        dropout=0.1,
    ):
        super(TransformerEncoder, self).__init__()
        self.embeddings = Embeddings(
            hidden_size,
            vocab_size,
            max_position_size,
            max_segment,
            word_emb_padding_idx,
        )
        self.multihead_attention_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads, dropout=dropout
        )
        layer_norm = nn.LayerNorm(hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            self.multihead_attention_encoder_layer, num_layers=n_layer, norm=layer_norm
        )

    def forward(self, x):
        emb, padding_mask = self.embeddings(x)
        emb = self.transformer_encoder(
            emb.transpose(0, 1), src_key_padding_mask=padding_mask
        ).transpose(0, 1)
        return emb  # [1, seq_len, hidden_size]


class MultiStageTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layer1,
        n_layer2,
        hidden_size,
        num_attention_heads,
        vocab_size,
        max_position_size1,
        max_position_size2,
        max_segment,
        word_emb_padding_idx,
        dropout,
        device=torch.device("cpu:0"),
    ):
        super(MultiStageTransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
        )
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder1 = TransformerEncoder(
            n_layer1,
            hidden_size,
            num_attention_heads,
            vocab_size,
            max_position_size1,
            max_segment,
            word_emb_padding_idx,
            dropout,
        )
        self.encoder2 = nn.TransformerEncoder(encoder_layer, n_layer2, encoder_norm)
        self.max_position_size2 = max_position_size2
        self.device = device
        self.hidden_size = hidden_size

    def forward(self, input_med_ids_list):
        """
        Optimized version of forward that pre-allocates tensors instead of using lists.
        """
        num_inputs = len(input_med_ids_list)

        # Pre-allocate output tensor: [1, max_position_size2, hidden_size]
        emb = torch.zeros(
            1, self.max_position_size2, self.hidden_size, device=self.device
        )

        # Fill in the actual embeddings
        for i, input_ids_list in enumerate(input_med_ids_list):
            processed_emb = self.encoder1(input_ids_list)
            emb[0, i] = processed_emb[:, 0].squeeze(
                0
            )  # Squeeze to remove batch dimension

        padding_mask = torch.zeros(
            1, self.max_position_size2, dtype=torch.bool, device=self.device
        )
        padding_mask[0, num_inputs:] = True  # True means padding (masked)

        emb = self.encoder2(
            emb.transpose(0, 1), src_key_padding_mask=padding_mask
        ).transpose(0, 1)
        return emb


# For multi-visit scenario - Optimized version
class MulAttention(nn.Module):
    def __init__(self, hidden_size, device):
        super(MulAttention, self).__init__()
        self.key = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.q = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.device = device

    def _create_local_mask(self, seq_len, k_mul, batch_size=1):
        """
        Create local attention mask using vectorized operations.
        Position i can attend to positions j where: max(0, i - k_mul) <= j <= i
        """
        # Create indices using broadcasting
        i = torch.arange(seq_len, device=self.device).unsqueeze(1)  # [seq_len, 1]
        j = torch.arange(seq_len, device=self.device).unsqueeze(0)  # [1, seq_len]

        # Create mask: j <= i AND j >= max(0, i - k_mul)
        # This is equivalent to: j >= max(0, i - k_mul) AND j <= i
        lower_bound = torch.clamp(i - k_mul, min=0)  # [seq_len, 1]
        mask = (j >= lower_bound) & (j <= i)  # [seq_len, seq_len]

        # Expand to batch dimension
        mask = mask.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, seq_len, seq_len]

        return mask.float()

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        Scaled dot-product attention
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Ensure mask has the same shape as scores
            if mask.dim() == scores.dim() - 1:
                mask = mask.unsqueeze(0)
            # Apply mask: set masked positions to very negative value
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, input_seq_rep, k_mul):
        """
        Forward pass with optimized mask creation.

        Args:
            input_seq_rep: [batch_size, seq_len, hidden_size]
            k_mul: Local attention window size

        Returns:
            out: [batch_size, seq_len, hidden_size]
            attn: [batch_size, seq_len, seq_len] attention weights
        """
        batch_size, seq_len, _ = input_seq_rep.shape

        # Project inputs
        input_seq_key = self.key(input_seq_rep)
        input_seq_q = self.q(input_seq_rep)

        # Create mask using vectorized operations (much faster than nested loops)
        mask = self._create_local_mask(seq_len, k_mul, batch_size)

        # Apply attention
        out, attn = self.attention(input_seq_q, input_seq_key, input_seq_key, mask=mask)

        return out, attn
