from torch.utils.data import Dataset
from torch import tensor, Tensor, float32
from torchvision.transforms.v2 import Compose, PILToTensor, Resize, ToDtype
from pandas import DataFrame, Series, read_csv
from PIL import Image
import numpy as np


class CoordinatesTransformPipeline:
    def __init__(self, df: DataFrame):
        self.df = df

    @staticmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Great-circle distance:
        https://en.wikipedia.org/wiki/Great-circle_distance
        Returns distance between x and y in meters
        """
        x, y = np.radians(x), np.radians(y)
        r = 6_371_000  # meters
        delta_lambda = np.abs(x[1] - y[1])
        return np.arccos(np.sin(x[0]) * np.sin(y[0]) +
                         np.cos(x[0]) * np.cos(y[0]) *
                         np.cos(delta_lambda)) * r

    def transform(self) -> DataFrame:
        coords_columns = ["building", "camera"]

        for col in coords_columns:
            self.df[col] = self.coords_transform(self.df[col])

        dists = []
        for i in range(len(self.df)):
            dists.append(self.distance(
                self.df[coords_columns[0]].iloc[i],
                self.df[coords_columns[1]].iloc[i]
            ))
        self.df["distance"] = Series(dists)
        return self.df.drop(columns=coords_columns)[["filename",
                                                     "distance",
                                                     "height"]]

    def coords_transform(self, column: Series) -> Series:
        splited = column.str.split(",")
        result = []
        for i in range(len(self.df)):
            result.append(np.array(
                splited.iloc[i],
                dtype=np.float64)
            )
        return Series(result)


class BuildingDataset(Dataset):
    """
    Torch dataset

    Returns tuple of form: tensor_image, tensor_dist, tensor_y
    (resize image to 400x400 pixels)

    - tensor_image is image of shape (n_channels, height, width),
    - tensor_dist is distance between camera and building in meters,
    - tensor_y is target.
    """
    def __init__(self,
                 img_folder: str,
                 df: DataFrame,
                 device: str,
                 mean: float | None = None,
                 std: float | None = None):
        self.normalize_params = {
            "mean": mean,
            "std": std
        }
        self.df = df
        self.df[["distance", "height"]] = self.numerical_normalize(
            self.df[["distance", "height"]]
        )
        self.folder = img_folder
        self.device = device
        # Thanks to Grander78498
        self.img_pipeline = Compose((
            PILToTensor(),
            Resize(size=(400, 400)),
            ToDtype(dtype=float32),
        ))

    def __len__(self) -> int:
        return len(self.df)

    def numerical_normalize(self,
                            data: DataFrame | float
                            | int) -> DataFrame | float:
        """
        Returns normalized data.
        """
        if self.normalize_params["mean"] is None and \
           self.normalize_params["std"] is None:
            self.normalize_params["mean"] = data.mean()
            self.normalize_params["std"] = data.std()

        data = (data - self.normalize_params["mean"]) / \
            self.normalize_params["std"]
        return data

    def __getitem__(self, key: int) -> tuple[Tensor,
                                             Tensor,
                                             Tensor]:
        filename = f"{self.folder}/{self.df['filename'].iloc[key]}"

        return (
                self.img_pipeline(Image.open(filename)).to(self.device),
                tensor(self.df["distance"].iloc[key],
                       device=self.device, dtype=float32),
                tensor(self.df["height"].iloc[key],
                       device=self.device, dtype=float32)
                )


def make_datasets(filename: str,
                  img_folder: str,
                  train_ratio: float,
                  device: str) -> tuple[Dataset, Dataset]:
    """
    Make train and test datasets

    Returns torch datasets.
    """
    df = read_csv(filename)
    transformer = CoordinatesTransformPipeline(df)
    df = transformer.transform()
    train_test_split_ind = int(df.shape[0] * train_ratio)
    train_dataset = BuildingDataset(img_folder,
                                    df.iloc[:train_test_split_ind, :],
                                    device)
    test_dataset = BuildingDataset(img_folder,
                                   df.iloc[train_test_split_ind:, :],
                                   device)

    return train_dataset, test_dataset
