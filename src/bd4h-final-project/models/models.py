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
