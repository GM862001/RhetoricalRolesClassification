DATASET_NAME = "ITA-RhetRoles"

DATA_FOLDER = ...  # Ita-RhetRoles is not publicly available
MODELS_FOLDER = ...  # We provide the best models only

HUGGING_FACE_BERT_MODEL = "dlicari/Italian-Legal-BERT"

WANDB_PROJECT = f"RhetoricalRolesClassification-{DATASET_NAME}"

rhetRole2label = {
    "UNK": 0,
    "CONCLUSIONI DELLE PARTI": 1,
    "SVOLGIMENTO DEL PROCESSO": 2,
    "MOTIVI DELLA DECISIONE": 3,
    "P.Q.M.": 4,
}
label2rhetRole = {label: rhetRole for rhetRole, label in rhetRole2label.items()}

NUM_LABELS = len(rhetRole2label)
MAX_SEGMENT_LENGTH = 64
