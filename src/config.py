from pathlib import Path
from dotenv import find_dotenv
from yaml import safe_load
import dotenv
from dotenv import find_dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv())
config_path = os.getenv('CONFIG_PATH')

BASE_PATH = Path(find_dotenv()).parent
CONFIG = safe_load(open(BASE_PATH / config_path, mode='r'))

DATASET_RAW = os.path.join(BASE_PATH, CONFIG.get('DATASET_RAW'))
