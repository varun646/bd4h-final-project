from rdkit import Chem
import dill
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
import re
from tqdm import tqdm
import torch

SMI_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"
)

# use ChemBERTa-zinc-base-v1 to tokenize the SMILES since original model is not available
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model.eval()

def tokenize(x):
    try:
        mol = Chem.MolFromSmiles(x)
        x = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception as e:
        pass

    x = " ".join(SMI_PATTERN.findall(x))
    return x


atc3toSMILES_file = "./output/atc3toSMILES_iii.pkl"
atc3tosmi = dill.load(open(atc3toSMILES_file, "rb"))

smiles_tok = {}
for smi_list in atc3tosmi.values():
    for smi in smi_list:
        if smi not in smiles_tok:
            smiles_tok[smi] = tokenize(smi)

print(list(smiles_tok.items())[:5], len(smiles_tok))

smiles_tok_ids = {}
for smi, tok in smiles_tok.items():
    enc = tokenizer(
        tok, padding="max_length", max_length=300, truncation=True, return_tensors="pt"
    )
    smiles_tok_ids[smi] = np.asarray(enc["input_ids"][0])


print(list(smiles_tok_ids.items())[0], len(smiles_tok_ids))
dill.dump(smiles_tok_ids, open("./output/SMILES_tok_ids_iii.pkl", "wb"))


voc = dill.load(open("./output/voc_iii_sym1_mulvisit.pkl", "rb"))
diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
atc_list = med_voc.idx2word.values()

input_smiles_reps_total = torch.zeros(len(atc_list), 768)
for k, atc in tqdm(enumerate(atc_list)):
    with torch.no_grad():
        input_smiles_reps = []
        for smi in atc3tosmi[atc]:
            input_ids = torch.LongTensor(smiles_tok_ids[smi]).unsqueeze(0)
            outputs = model(input_ids)
            input_smiles_reps.append(outputs.last_hidden_state[:, 0])
        input_smiles_reps_total[k] = sum(input_smiles_reps) / len(input_smiles_reps)

dill.dump(input_smiles_reps_total, open("./output/input_smiles_init_rep_iii.pkl", "wb"))
