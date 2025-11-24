# DrugRec - Debiased, Longitudinal and Coordinated Drug Recommendation

This repository is a recreation of the DrugRec model from the NeurIPS 2022 paper **"Debiased, Longitudinal and Coordinated Drug Recommendation through Multi-Visit Clinic Records"** by Sun et al.

This implementation is built off the original repository at [https://github.com/ssshddd/DrugRec](https://github.com/ssshddd/DrugRec).

## Overview

DrugRec is a deep learning model for recommending medications based on multi-visit clinical records. The model addresses drug recommendation challenges by:
- **Debiasing**: Reducing bias in drug recommendations through causal inference
- **Longitudinal modeling**: Leveraging multi-visit patient history
- **Coordination**: Ensuring safe drug combinations through DDI (Drug-Drug Interaction) awareness

## Dependencies

### Core Dependencies

The project requires Python 3.13+ and the following key packages:

```bash
# Deep learning framework
torch>=1.10.1  # PyTorch (CUDA version recommended for GPU support)

# Data processing
pandas>=2.3.3
numpy
dill>=0.4.0
polars>=1.35.2

# Machine learning utilities
scikit-learn>=0.24.2
transformers  # For Clinical BERT and SMILES tokenization

# Natural language processing
nltk  # For symptom extraction

# Chemistry
rdkit  # For SMILES processing (install via conda: conda install -c conda-forge rdkit)

# Visualization
matplotlib>=3.10.7

# Healthcare data processing
pyhealth>=1.1.3

# Utilities
tqdm  # Progress bars
```

### Installation

We recommend using a package manager like `uv` (as indicated by `uv.lock` in this repository) or `pip`:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install torch transformers scikit-learn nltk pandas numpy dill matplotlib pyhealth polars
conda install -c conda-forge rdkit
```

**Note**: For SMILES tokenization, you may need to install additional pre-trained models. The original repository uses:
- Clinical BERT from [https://github.com/EmilyAlsentzer/clinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT)
- A roberta-large based model from [https://github.com/microsoft/DVMP](https://github.com/microsoft/DVMP) (requires registration)

## Data Requirements

This project uses the MIMIC-III dataset. You will need:

1. **MIMIC-III dataset** from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) (requires certification)
2. **Mapping files** (available from [SafeDrug repository](https://github.com/ycq091044/SafeDrug)):
   - `RXCUI2atc4.csv`: NDC-RXCUI-ATC4 mapping
   - `rxnorm2RXCUI.txt`: NDC-RXCUI mapping
   - `drugbank_drugs_info.csv`: Drugname-SMILES mapping
   - `drug-atc.csv`: CID-ATC mapping
   - `drug-DDI.csv`: DDI information coded by CID

Place the MIMIC-III CSV files in `data/mimic-iii/` and mapping files in `data/input/`.

## Data Preprocessing

Before training, you need to preprocess the MIMIC-III data. The preprocessing pipeline consists of several steps:

1. **Step 1: Load and merge data** (`data/preprocessing/processing_iii.py`)
   ```bash
   cd data/preprocessing
   python processing_iii.py
   ```
   Generates: `data/output/data_iii_sym0.pkl`

2. **Step 2: Extract symptoms** (`data/preprocessing/sym_tagger_iii.py`)
   ```bash
   python sym_tagger_iii.py
   ```
   Generates: `data/output/data_iii_sym1_mulvisit.pkl`

3. **Step 3: Tokenize symptoms, diagnoses, and procedures** (`data/preprocessing/input_ids_sdp_iii.py`)
   ```bash
   python input_ids_sdp_iii.py
   ```
   Uses Clinical BERT for tokenization.

4. **Step 4: Tokenize medications** (`data/preprocessing/input_smiles_iii_2.py`)
   ```bash
   python input_smiles_iii_2.py
   ```
   Uses a pre-trained model to encode SMILES strings.

5. **Step 5: Generate symptom information** (`data/preprocessing/sym_info_iii.py`)
   ```bash
   python sym_info_iii.py
   ```

6. **Step 6: Generate multi-visit records and DDI matrix** (`data/preprocessing/gen_records_ddi.py`)
   ```bash
   python gen_records_ddi.py
   ```
   Generates: `data/output/records_final_iii.pkl` and `data/output/ddi_A_iii.pkl`

## Running the Code

### Training

To train the DrugRec model from scratch:

```bash
python -m src.bd4hFinalProject.main
```

The training script will:
- Load preprocessed data from `data/output/`
- Split data into train/validation/test sets (2/3 train, 1/6 validation, 1/6 test)
- Train the model for the specified number of epochs (default: 50)
- Save model checkpoints to `saved/DrugRec_mimic-iii/`

### Testing/Evaluation

To evaluate a trained model, modify the `Config` class in `src/bd4hFinalProject/main.py`:

```python
class Config:
    Test = True  # Enable test mode
    resume_path = "saved/DrugRec_mimic-iii/Epoch_X_TARGET_0.05_JA_X.XXXX_AUC_X.XXXX_F1_X.XXXX_DDI_X.XXXX.model"
    # ... other settings
```

Then run:
```bash
python -m src.bd4hFinalProject.main
```

### Configuration

Model hyperparameters can be adjusted in the `Config` class in `src/bd4hFinalProject/main.py`:

- `lr`: Learning rate (default: 5e-4)
- `epoch`: Number of training epochs (default: 50)
- `target_ddi`: Target DDI rate (default: 0.05)
- `dim`: Embedding dimension (default: 64)
- `CI`: Enable causal inference loss (default: True)
- `multivisit`: Use multi-visit history (default: True)

## Project Structure

```
.
├── data/
│   ├── mimic-iii/          # MIMIC-III raw data files
│   ├── preprocessing/      # Data preprocessing scripts
│   └── output/             # Preprocessed data files
├── src/
│   └── bd4hFinalProject/
│       ├── main.py         # Main training/evaluation script
│       ├── models/         # Model architecture
│       ├── util.py         # Utility functions
│       └── plots.py        # Plotting utilities
├── saved/                  # Saved model checkpoints
└── pyproject.toml          # Project dependencies
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{sun2022debiased,
  title={Debiased, Longitudinal and Coordinated Drug Recommendation through Multi-Visit Clinic Records},
  author={Sun, Hongda and Xie, Shufang and Li, Shuqi and Chen, Yuhan and Wen, Ji-Rong and Yan, Rui},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Acknowledgments

This implementation is based on the original DrugRec repository: [https://github.com/ssshddd/DrugRec](https://github.com/ssshddd/DrugRec)

The original repository acknowledges:
- [SafeDrug](https://github.com/ycq091044/SafeDrug) for data processing utilities
- [GAMENet](https://github.com/sjy1203/GAMENet) for reference implementations

## License

See LICENSE file for details.
