from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tasks import drug_recommendation_mimic3_fn

print("Loading data...")
# STEP 1: load data
base_dataset = MIMIC3Dataset(
    root="/Users/vnarayan35/Documents/GitHub/bd4h-final-project/data/mimic-iii",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    dev=True,
    refresh_cache=False,
)
base_dataset.stat()

sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn)
sample_dataset.stat()


print("\nFirst sample structure:")
print(f"Patient ID: {sample_dataset.samples[0]['patient_id']}")
print(f"Number of visits: {len(sample_dataset.samples[0]['conditions'])}")
print(f"Sample conditions (first visit): {sample_dataset.samples[0]['conditions'][0][:5]}...")
print(f"Sample procedures (first visit): {sample_dataset.samples[0]['procedures'][0][:5]}...")
print(f"Sample drugs (target): {sample_dataset.samples[0]['drugs'][:10]}...")




train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, ratios=[2.0/3.0, 1.0/6.0, 1.0/6.0]
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)


