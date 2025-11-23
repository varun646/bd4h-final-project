import torch.nn as nn
import torch.nn.functional as F
import torch


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


class SingleStageTransformerEncoder(nn.Module):
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
        super(SingleStageTransformerEncoder, self).__init__()
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
        self.encoder1 = SingleStageTransformerEncoder(
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
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.multihead_attention_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.multihead_attention_encoder, num_layers=n_layer
        )

    def forward(self, x):
        embeddings, padding_mask = self.embeddings(x)

        # NOTE: don't know how many attention heads. Keeping low for faster convergence.
        transformer_encoding = self.transformer_encoder(
            embeddings.transpose(0, 1), src_key_padding_mask=padding_mask
        ).transpose(0, 1)
        normalized = self.layer_norm(transformer_encoding)
        # encoder_2 = self.

        return transformer_encoding
