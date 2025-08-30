# config_manager.py
import os
import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".rag_config.json"

DEFAULT_CONFIG = {
    "embedModel_api_key": None,
    "vector_api_key": None,
    "default_model": "cohere-v3"
}


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    else:
        return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[📝] Config saved to: {CONFIG_PATH}")


def set_api_key(service, key):
    config = load_config()
    if service == "embedModel":
        config["embedModel_api_key"] = key
    elif service == "vector":
        config["vector_api_key"] = key
    else:
        raise ValueError(f"Unknown service: {service}")
    save_config(config)
    print(f"[✅] Saved {service} API key.")


def get_config():
    return load_config()


def get_api_key(service):
    config = load_config()
    if service == "embedModel":
        return config["embedModel_api_key"]
    elif service == "vector":
        return config["vector_api_key"]
    else:
        raise ValueError(f"Unknown service: {service}")
