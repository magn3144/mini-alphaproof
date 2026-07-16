from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
DATASET_DIR = DATA_DIR / 'dataset'
RUNS_DIR = DATA_DIR / 'runs'
LEAN_PROJECT_DIR = PROJECT_ROOT / 'lean_project'
MODELS_DIR = PROJECT_ROOT / 'models'
DEFAULT_THEOREMS_PATH = DATASET_DIR / 'test_theorems.jsonl'
