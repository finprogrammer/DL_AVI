# def load_config(mode):
#     if mode == "train":
#         return __import__("train_config")
#     elif mode == "debug":
#         return __import__("debug_config")
#     else:
#         raise ValueError("Invalid mode. Please specify 'train' or 'debug'.")
import yaml

def load_config(mode):
    if mode not in ["train", "debug", "train_2"]:
        raise ValueError("Invalid mode. Please specify 'train', 'train_2', or 'debug'.")

    if mode == "train":
        filename = "/home/vault/iwfa/iwfa048h/CNN/train_config.yaml"
    elif mode == "train_2":
        filename = "/home/vault/iwfa/iwfa048h/CNN/train_config_2.yaml"
    else:
        filename = "/home/vault/iwfa/iwfa048h/CNN/debug_config.yaml"
    
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    
    return config