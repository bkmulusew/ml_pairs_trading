from dataclasses import dataclass

@dataclass
class ModelConfig:
    INPUT_CHUNK_LENGTH: int = 50
    OUTPUT_CHUNK_LENGTH: int = 1
    N_EPOCHS: int = 3
    BATCH_SIZE: int = 1024
    DATA_FILE_PATH: str = "Pairs/sample_data.csv"
    TRAIN_RATIO: float = 0.5