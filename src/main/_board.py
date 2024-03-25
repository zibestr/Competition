from torch.nn import Module
from numpy import ndarray, inf
from matplotlib.pyplot import figure, show as plt_show, Axes
from math import ceil


class Board:
    """
    Class for scoring model
    """
    def __init__(self,
                 train_losses: ndarray,
                 test_losses: ndarray,
                 model: Module):
        self.__count_parameters = sum(p.numel()
                                      for p in model.parameters()
                                      if p.requires_grad)
        self.__model_info = str(model)
        self.__train_losses = train_losses
        self.__test_losses = test_losses

    @property
    def __max_test_loss(self) -> tuple[int, float]:
        """
        Returns number of epoch with maximum test loss and value
        """
        max_epoch: int = -1
        max_loss: float = -inf
        for epoch, losses in enumerate(self.__test_losses):
            if max_loss < losses.max():
                max_epoch = epoch
                max_loss = losses.max()

        return (max_epoch, max_loss)

    @property
    def __min_test_loss(self) -> tuple[int, float]:
        """
        Returns number of epoch with minimum test loss and value
        """
        min_epoch: int = -1
        min_loss: float = inf
        for epoch, losses in enumerate(self.__test_losses):
            if min_loss > losses.min():
                min_epoch = epoch
                min_loss = losses.min()

        return (min_epoch, min_loss)

    def __print_info(self) -> None:
        print(f"\n\nNeural network: {self.__model_info}")
        print(f"------------------------------------------\n"
              "Count parameters: "
              f"{ceil(self.__count_parameters / 1_000_000)}M")
        best_epoch, best_loss = self.__min_test_loss
        worst_epoch, worst_loss = self.__max_test_loss
        print(f"------------------------------------------\n"
              f"Best loss: {round(best_loss, 2)}, epoch: {best_epoch}")
        print(f"Worst loss: {round(worst_loss, 2)}, epoch: {worst_epoch}")
        print("------------------------------------------")

    @staticmethod
    def __prepare_ax(ax: Axes,
                     xlabel: str,
                     ylabel: str,
                     title: str) -> Axes:
        ax.grid()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax

    def __graw_plots(self):
        fig = figure(figsize=(8, 10), dpi=100)
        fig.suptitle("Score Board", fontsize=16)
        axes = fig.subplots(3, 2)

        axes[0, 0].hist(list(map(lambda x: x.min(),
                                 self.__test_losses)))
        axes[0, 0] = self.__prepare_ax(axes[0, 0], "Epoch", "Loss",
                                       "Best Test Losses")

        axes[0, 1].hist(list(map(lambda x: x.max(),
                                 self.__test_losses)))
        axes[0, 1] = self.__prepare_ax(axes[0, 1], "Epoch", "Loss",
                                       "Worst Test Losses")

        axes[1, 0].hist(list(map(lambda x: x.min(),
                                 self.__train_losses)))
        axes[1, 0] = self.__prepare_ax(axes[1, 0], "Epoch", "Loss",
                                       "Best Train Losses")

        axes[1, 1].hist(list(map(lambda x: x.max(),
                                 self.__train_losses)))
        axes[1, 1] = self.__prepare_ax(axes[1, 1], "Epoch", "Loss",
                                       "Worst Train Losses")

        axes[2, 0].plot(list(map(lambda x: x.mean(),
                                 self.__train_losses)),
                        color="green")
        axes[2, 0] = self.__prepare_ax(axes[2, 0], "Epoch", "Loss",
                                       "Mean train losses")

        axes[2, 1].plot(list(map(lambda x: x.mean(),
                                 self.__test_losses)),
                        color="red")
        axes[2, 1] = self.__prepare_ax(axes[2, 1], "Epoch", "Loss",
                                       "Mean test losses")

        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        plt_show()

    def stats(self) -> None:
        """
        Output information about the model and train/test loops
        """
        self.__print_info()
        self.__graw_plots()
