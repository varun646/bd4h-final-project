import pandas as pd
from collections import defaultdict
import ast

# Constants
TOP_N_MEDICATIONS = 300
TOP_N_DIAGNOSES = 2000
MAX_SMILES_PER_ATC3 = 3
MIN_VISITS_PER_PATIENT = 2


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


def filter_patients_with_multiple_visits(med_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to patients with at least MIN_VISITS_PER_PATIENT visits.

    Args:
        med_pd: Medication DataFrame

    Returns:
        DataFrame with patient IDs and their visit counts
    """
    visit_counts = (
        med_pd[["SUBJECT_ID", "HADM_ID"]]
        .groupby("SUBJECT_ID")["HADM_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"HADM_ID": "visit_count"})
    )

    patients_with_multiple_visits = visit_counts[
        visit_counts["visit_count"] >= MIN_VISITS_PER_PATIENT
    ][["SUBJECT_ID"]]

    return patients_with_multiple_visits


def filter_top_medications(
    med_pd: pd.DataFrame, top_n: int = TOP_N_MEDICATIONS
) -> pd.DataFrame:
    """
    Filter to top N most common medications by ATC3 code.

    Args:
        med_pd: Medication DataFrame
        top_n: Number of top medications to keep

    Returns:
        Filtered medication DataFrame
    """
    medication_counts = (
        med_pd.groupby("ATC3")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    top_medications = medication_counts.head(top_n)["ATC3"]
    med_pd_filtered = med_pd[med_pd["ATC3"].isin(top_medications)]

    return med_pd_filtered.reset_index(drop=True)


def filter_top_diagnoses(
    diag_pd: pd.DataFrame, top_n: int = TOP_N_DIAGNOSES
) -> pd.DataFrame:
    """
    Filter to top N most common diagnoses by ICD9 code.

    Args:
        diag_pd: Diagnosis DataFrame
        top_n: Number of top diagnoses to keep

    Returns:
        Filtered diagnosis DataFrame
    """
    diagnosis_counts = (
        diag_pd.groupby("ICD9_CODE")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    top_diagnoses = diagnosis_counts.head(top_n)["ICD9_CODE"]
    diag_pd_filtered = diag_pd[diag_pd["ICD9_CODE"].isin(top_diagnoses)]

    return diag_pd_filtered.reset_index(drop=True)


def diag_process(diag_file: str) -> pd.DataFrame:
    """
    Process diagnosis data from MIMIC-III DIAGNOSES_ICD table.

    Args:
        diag_file: Path to DIAGNOSES_ICD.csv file

    Returns:
        Processed diagnosis DataFrame
    """
    diag_pd = pd.read_csv(diag_file)
    diag_pd = diag_pd.dropna().drop(columns=["SEQ_NUM", "ROW_ID"])
    diag_pd = diag_pd.drop_duplicates()
    diag_pd = diag_pd.sort_values(by=["SUBJECT_ID", "HADM_ID"]).reset_index(drop=True)

    # Filter to top diagnoses
    diag_pd = filter_top_diagnoses(diag_pd, top_n=TOP_N_DIAGNOSES)

    # Append diagnosis titles
    diag_pd = append_diagnosis_titles("D_ICD_DIAGNOSES.csv", diag_pd)
    diag_pd = diag_pd.rename(columns={"LONG_TITLE": "ICD9_TEXT"})

    return diag_pd


def ATC3ToDrugDict(med_df):
    # TODO: rewrite
    atc3ToDrugDict = defaultdict(set)
    for _, row in med_df.iterrows():
        atc3 = row["atc3"]
        drug = row["drug"]
        atc3ToDrugDict[atc3].add(drug)

    return dict(atc3ToDrugDict)


def atc3ToSMILES(atc3ToDrugDict, druginfo):
    # TODO: rewrite
    drug2smiles = {}
    atc3tosmiles = {}
    for drugname, smiles in druginfo[["name", "moldb_smiles"]].values:
        if isinstance(smiles, str):
            drug2smiles[drugname] = smiles
    for atc3, drug in atc3ToDrugDict.items():
        temp = []
        for d in drug:
            temp.append(drug2smiles[d])
        if len(temp) > 0:
            atc3tosmiles[atc3] = temp[:3]

    return atc3tosmiles


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

med_df = map_ndc_to_atc3(
    med_df, "./data/ndc2rxnorm_mapping.txt", "./data/RXCUI2atc4.csv"
)
