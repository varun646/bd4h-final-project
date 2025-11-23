import torch.nn as nn
import torch


class DrugRec(nn.Module):
    def __init__(
        self,
        args,
        sym_information,
        ddi_adj,
        input_smiles_init_rep,
        emb_dim,
        device=torch.device("cpu:0"),
    ):
        self.device = device
        self.args = vars(args)

        sym_count, sym2idx, sym_comatrix, sym_input_ids = sym_information

        trans_dim = emb_dim
        self.transformer_dp = Transformer_Encoder(n_layer=2, hidden_size=trans_dim, num_attention_heads=4, vocab_size=28996, max_position_size=576, max_segment=40, word_emb_padding_idx=0, dropout=0.1)
        self.transformer_s = Transformer_Encoder(n_layer=2, hidden_size=trans_dim, num_attention_heads=4, vocab_size=28996, max_position_size=256, max_segment=40, word_emb_padding_idx=0, dropout=0.1)

        self.sym_d = nn.Sequential(nn.ReLU(), nn.Linear(trans_dim*2, trans_dim))
        self.sym_p = nn.Sequential(nn.ReLU(), nn.Linear(trans_dim*2, trans_dim))
        self.lin = nn.Sequential(nn.ReLU(), nn.Linear(trans_dim, emb_dim))

        self.diag_seq_enc = Mul_Attention(hidden_size=emb_dim, device=device)
        self.pro_seq_enc = Mul_Attention(hidden_size=emb_dim, device=device)
        self.sym_seq_enc = Mul_Attention(hidden_size=emb_dim, device=device)

        if self.args['mulhistory']:
            self.diag_agg = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim))
            self.pro_agg = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim))
            self.sym_agg = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        
        


