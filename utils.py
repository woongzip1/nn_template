def print_config(config, indent=0):
    for k, v in config.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_config(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")