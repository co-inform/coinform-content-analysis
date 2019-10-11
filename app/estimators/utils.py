import torch

CUDA = 'cuda'
CPU = 'cpu'


def get_device():
    return torch.device(CUDA) if torch.cuda.is_available() else torch.device(CPU)


def load_model_from_file(file_path: str) -> torch.nn.Module:
    model = torch.load(file_path, map_location=get_device())
    return model

