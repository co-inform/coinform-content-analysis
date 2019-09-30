import torch

CUDA = 'cuda'
CPU = 'cpu'


def load_model_from_file(file_path: str) -> torch.nn.Module:
    dev = torch.device(CUDA) if torch.cuda.is_available() else torch.device(CPU)
    model = torch.load(file_path, map_location=dev)
    return model
