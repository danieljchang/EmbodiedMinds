def load_model(model_path):
    import torch
    model = torch.load(model_path)
    model.eval()
    return model

def save_model(model, model_path):
    import torch
    torch.save(model, model_path)

def load_data(data_path):
    import pandas as pd
    return pd.read_csv(data_path)

def save_data(data, data_path):
    import pandas as pd
    data.to_csv(data_path, index=False)

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path):
    import yaml
    with open(config_path, 'w') as file:
        yaml.dump(config, file)