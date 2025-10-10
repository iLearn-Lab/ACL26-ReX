HF_CACHE_DIR = "XXXXXX"
PROJECT_DATA_PATH = "XXXXXX"
EVAL_DATASET_PATH = "XXXXXX"
RESULT_PATH = "XXXXXX"
CODE_PATH = "XXXXXX"
# repalce XXXXXX with your own paths

EVAL_DATASET_SIZE = {
    "gpqa": 200,
    "mmlupro_chemistry": 200,
    "mmlupro_economics": 200,
    "math500_level3": 500,
    "mmlupro_math": 200,
}

MODELHFID_MAP = {
    "llama3.1-8b-instruct": "Llama-3.1-8B-Instruct",
    "llama3.2-3b-instruct": "Llama-3.2-3B-Instruct",
    "mistral-7b-instruct": "Mistral-7B-Instruct-v0.3",
}

MODEL_TARGETMODULES_MAP = {
    "llama3.1-8b-instruct": ["gate_proj", "up_proj", "down_proj"],
    "llama3.2-3b-instruct": ["gate_proj", "up_proj", "down_proj"],
    "mistral-7b-instruct": ["gate_proj", "up_proj", "down_proj"],
}

MODEL_TARGETMODULES_MAP2 = {
    "llama3.1-8b-instruct": ["q_proj", "o_proj", "k_proj", "v_proj"],
    "llama3.2-3b-instruct": ["q_proj", "o_proj", "k_proj", "v_proj"],
    "mistral-7b-instruct": ["q_proj", "o_proj", "k_proj", "v_proj"],
}

MODEL_TARGETMODULES_MAP3 = {
    "llama3.1-8b-instruct": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    "llama3.2-3b-instruct": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral-7b-instruct": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
}

MODEL_TARGETMODULES_MAP4 = {
    "llama3.1-8b-instruct": ["q_proj", "v_proj"],
    "llama3.2-3b-instruct": ["q_proj", "v_proj"],
    "mistral-7b-instruct": ["q_proj", "v_proj"],
}
