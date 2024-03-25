from torch import load
from torch.nn import Module


class ModelLoader:
    """
    Loader model weights from .pth file

    - model_cls - class inheriting from torch.nn.Module, neural network class
    - weights_filename: str - .pth filename with weights for model
    - device: str - device name
    """
    def __init__(self,
                 model_cls,
                 weights_filename: str,
                 device: str) -> None:
        self.model_cls = model_cls
        self.filename = weights_filename
        self.device = device

    def get_model(self) -> Module:
        """
        Load model from file

        Returns model object
        """
        model = self.model_cls().to(self.device)
        model.load_state_dict(load(self.filename))
        return model
