import pandas as pd


def process_meds(med_filepath):
    med_df = pd.read_csv(
        med_filepath, 
        usecols=['subject_id', 'hadm_id', 'icustay_id', 'startdate', 'drug', 'ndc']
    )
    
    med_df.dropna(subset=["ndc"], inplace=True)
    
    med_df.ffill(inplace=True)
    med_df.dropna(inplace=True)
    
    # Convert icustay_id to int64 before sorting
    med_df["icustay_id"] = med_df["icustay_id"].astype("int64")
    
    # Convert startdate to datetime
    med_df["startdate"] = pd.to_datetime(
        med_df["startdate"], format="%Y-%m-%d %H:%M:%S"
    )
    
    med_df.sort_values(
        by=["subject_id", "hadm_id", "icustay_id", "startdate"], 
        inplace=True
    )
    
    # Drop icustay_id column and remove duplicates
    med_df = med_df.drop(columns=["icustay_id"])
    med_df = med_df.drop_duplicates().reset_index(drop=True)
    
    return med_df

df = process_meds('./data/mimic-iii-clinical-database-demo-1.4/PRESCRIPTIONS.csv')
print(df.head())
print(f"\nProcessed {len(df)} medication records")