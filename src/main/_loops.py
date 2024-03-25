from abc import ABC, abstractmethod
from time import time
from torch import Tensor, no_grad
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable
from numpy import ndarray, empty, hstack


class _AbstractLoop(ABC):
    """
    Abstract class for implementing
    """
    def __init__(self,
                 model: Module,
                 loss_function: Callable[[Tensor, Tensor], Tensor]
                 | Module) -> None:
        self.model = model
        self.loss_fn = loss_function

    @abstractmethod
    def _step(self,
              X_batch: tuple[Tensor, Tensor],
              y_batch: Tensor) -> float:
        """
        Step of the loop.

        - X_batch: torch.Tensor - batch of X data
        - y_batch: torch.Tensor - batch of y data

        Returns loss on step.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self,
            dataloader: DataLoader) -> ndarray:
        """
        Start loop.

        - dataloader: torch.utils.data.DataLoader - data loader of input data

        Returns loss on all loop steps as numpy array.
        """
        raise NotImplementedError


class Trainer(_AbstractLoop):
    """
    Class for training model.

    - model: torch.nn.Module - module for training
    - loss_function: Callable[[Tensor], Tensor] | torch.nn.Module -
    loss function
    - optimizer: torch.optim.Optimizer - optimizer of parameters
    """
    def __init__(self,
                 model: Module,
                 loss_function: Callable[[Tensor, Tensor], Tensor] | Module,
                 optimizer: Optimizer) -> None:
        super().__init__(model, loss_function)
        self.optim = optimizer

    def _step(self,
              X_batch: tuple[Tensor, Tensor],
              y_batch: Tensor) -> float:
        image, dist = X_batch
        logits = self.model((image, dist)).view(y_batch.shape)
        loss = self.loss_fn(logits, y_batch)

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        return loss.item()

    def run(self, dataloader: DataLoader) -> ndarray:
        start_time = time()
        size = len(dataloader)
        losses = empty(0)

        self.model.train(True)

        for batch, (image, dist, y) in enumerate(dataloader):
            loss = self._step((image, dist), y)
            losses = hstack((losses, loss))

            print(f"Train loop, batch {batch + 1}/{size}, loss: {loss}")
        print(f"Train loop finished! Time: {round(time() - start_time, 3)} s.")

        return losses


class Tester(_AbstractLoop):
    """
    Class for testing model.

    - model: torch.nn.Module - module for testing
    - loss_function: Callable[[Tensor], Tensor] | torch.nn.Module -
    loss function
    """
    def __init__(self,
                 model: Module,
                 loss_function: Callable[[Tensor, Tensor], Tensor]
                 | Module) -> None:
        super().__init__(model, loss_function)

    def _step(self,
              X_batch: tuple[Tensor, Tensor],
              y_batch: Tensor) -> float:
        image, dist = X_batch
        logits = self.model((image, dist)).view(y_batch.shape)
        loss = self.loss_fn(logits, y_batch)

        return loss.item()

    def run(self, dataloader: DataLoader) -> ndarray:
        start_time = time()
        size = len(dataloader)
        losses = empty(0)

        self.model.eval()

        with no_grad():
            for batch, (image, dist, y) in enumerate(dataloader):
                loss = self._step((image, dist), y)
                losses = hstack((losses, loss))

                print(f"Test loop, batch {batch + 1}/{size}, loss: {loss}")

        print(f"Test loop finished! Time: {round(time() - start_time, 3)} s.")
        return losses
