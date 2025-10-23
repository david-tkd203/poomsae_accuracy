import os, yaml
from dotenv import load_dotenv

load_dotenv()

def load_conf():
    conf_path = os.getenv("CONF_PATH", "./config/default.yaml")
    with open(conf_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def env(key, default=None, cast=str):
    val = os.getenv(key, default)
    return cast(val) if val is not None and cast is not str else val
