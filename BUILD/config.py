DATASET_NAME = "BUILD"

DATA_FOLDER = "./data"
MODELS_FOLDER = ...  # We provide the best models only

HUGGING_FACE_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"

WANDB_PROJECT = f"RhetoricalRolesClassification-{DATASET_NAME}"

rhetRole2label = {
    "PREAMBLE": 0,
    "FAC": 1,
    "RLC": 2,
    "ISSUE": 3,
    "ARG_PETITIONER": 4,
    "ARG_RESPONDENT": 5,
    "ANALYSIS": 6,
    "STA": 7,
    "PRE_RELIED": 8,
    "PRE_NOT_RELIED": 9,
    "RATIO": 10,
    "RPC": 11,
    "NONE": 12,
}
label2rhetRole = {label: rhetRole for rhetRole, label in rhetRole2label.items()}

NUM_LABELS = len(rhetRole2label)
MAX_SEGMENT_LENGTH = 64
