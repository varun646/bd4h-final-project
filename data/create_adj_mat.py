import pandas as pd
import numpy as np

import pickle


def create_adj_mat(filepath, output_filepath):
    """Create DDI adjacency matrix from two-sides csv file"""
    twosides_df = pd.read_csv(
        filepath, usecols=["drug_1_concept_name", "drug_2_concept_name"]
    )

    drug_pairs = set()

    for _, row in twosides_df.iterrows():
        drug_1_concept_name = row["drug_1_concept_name"]
        drug_2_concept_name = row["drug_2_concept_name"]
        drug_pairs.add(tuple(sorted((drug_1_concept_name, drug_2_concept_name))))

    with open(output_filepath, "wb") as f:
        pickle.dump(drug_pairs, f)


create_adj_mat("./data/TWOSIDES.csv", "./data/output/ddi_twosides_pairs.pkl")
