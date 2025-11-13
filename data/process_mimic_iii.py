import pandas as pd
from collections import defaultdict
import ast


def process_meds(med_filepath):
    med_df = pd.read_csv(
        med_filepath,
        usecols=["subject_id", "hadm_id", "icustay_id", "startdate", "drug", "ndc"],
        dtype={
            "subject_id": "string",
            "hadm_id": "string",
            "icustay_id": "string",
            "drug": "string",
            "ndc": "string",
        },
    )

    med_df.dropna(subset=["ndc"], inplace=True)

    med_df.ffill(inplace=True)
    med_df.dropna(inplace=True)

    # Convert icustay_id to int64 before sorting
    # med_df["icustay_id"] = med_df["icustay_id"].astype("int64")

    # Convert startdate to datetime
    med_df["startdate"] = pd.to_datetime(
        med_df["startdate"], format="%Y-%m-%d %H:%M:%S"
    )

    med_df.sort_values(
        by=["subject_id", "hadm_id", "icustay_id", "startdate"], inplace=True
    )

    # Drop icustay_id column and remove duplicates
    med_df = med_df.drop(columns=["icustay_id"])
    med_df = med_df.drop_duplicates().reset_index(drop=True)

    return med_df


def map_ndc_to_atc3(med_df, ndc_to_rxnorm_filepath, rxnorm_to_atc4_filepath):
    with open(ndc_to_rxnorm_filepath, "r") as f:
        ndc_rxnorm_dict = ast.literal_eval(f.read())

    rxnorm_to_atc_df = pd.read_csv(
        rxnorm_to_atc4_filepath,
        usecols=["RXCUI", "ATC4"],
        dtype={"RXCUI": "string", "ATC4": "string"},
    )
    rxnorm_to_atc_df.drop_duplicates(subset=["RXCUI"], inplace=True)
    rxnorm_to_atc_df.dropna(inplace=True)

    med_df["rxnorm"] = med_df["ndc"].map(ndc_rxnorm_dict)
    med_df.dropna(subset=["rxnorm"], inplace=True)
    med_df["atc4"] = med_df["rxnorm"].map(rxnorm_to_atc_df.set_index("RXCUI")["ATC4"])
    med_df.dropna(subset=["atc4"], inplace=True)
    med_df["atc4"] = med_df["atc4"].map(lambda x: x[:4])
    med_df.rename(columns={"atc4": "atc3"}, inplace=True)
    med_df.drop(columns=["rxnorm", "ndc"], inplace=True)
    med_df.drop_duplicates(inplace=True)

    print(med_df.head())
    return med_df


# def ATC3ToDrugDict(med_df):
#     atc3ToDrugDict = defaultdict(set)
#     for _, row in med_df.iterrows():
#         atc3 = row['atc3']
#         drug = row['drug']
#         atc3ToDrugDict[atc3].add(drug)

#     return dict(atc3ToDrugDict)


# def diagnosis_process(diagnosis_filepath):
#     diag_df = pd.read_csv(diagnosis_filepath)
#     diag_df.dropna(inplace=True)
#     diag_df.drop(columns=['seq_num','row_id'], inplace=True)
#     diag_df.drop_duplicates(inplace=True)
#     diag_df.sort_values(by=['subject_id','hadm_id'], inplace=True)
#     diag_df = diag_df.reset_index(drop=True)


#     return diag_df


med_df = process_meds("./data/mimic-iii-clinical-database-demo-1.4/PRESCRIPTIONS.csv")
# atc3ToDrugDict = ATC3ToDrugDict(med_df)
# print(atc3ToDrugDict)
print(med_df.head())

print("mapping to atc3")

med_df = map_ndc_to_atc4(
    med_df, "./data/ndc2rxnorm_mapping.txt", "./data/RXCUI2atc4.csv"
)
