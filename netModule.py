import torch

from torch import Tensor

import pandas as pd
import numpy as np

from PIL import Image


class CoordinatesTransformPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Great-circle distance: https://en.wikipedia.org/wiki/Great-circle_distance
        Returns distance between x and y in meters
        """
        x, y = np.radians(x), np.radians(y)
        r = 6_371_000  # meters
        delta_lambda = np.abs(x[1] - y[1])
        return np.arccos(np.sin(x[0]) * np.sin(y[0]) +
                         np.cos(x[0]) * np.cos(y[0]) *
                         np.cos(delta_lambda)) * r

    def transform(self) -> pd.DataFrame:
        coords_columns = ["building", "camera"]

        for col in coords_columns:
            self.df[col] = self.coords_transform(self.df[col])

        dists = []
        for i in range(len(self.df)):
            dists.append(self.distance(
                self.df[coords_columns[0]].iloc[i],
                self.df[coords_columns[1]].iloc[i]
            ))
        self.df["distance"] = pd.Series(dists)
        return self.df.drop(columns=coords_columns)[
            ["filename", "distance"]]

    def coords_transform(self, column: pd.Series) -> pd.Series:
        splited = column.str.split(",")
        result = []
        for i in range(len(self.df)):
            result.append(np.array(
                splited.iloc[i],
                dtype=np.float64)
            )
        return pd.Series(result)


class BuildingDataset(torch.utils.data.Dataset):
    """
    Torch dataset

    Returns tuple of form: tensor_image, tensor_dist, tensor_y
    (resize image to 400x400 pixels)

    - tensor_image is image of shape (n_channels, height, width),
    - tensor_dist is distance between camera and building in meters,
    - tensor_y is target.
    """

    def __init__(self, folder: str, df: pd.DataFrame, device: str,
                 mean: torch.float, std: torch.float):
        self.df = df
        self.df[["distance"]] = self.numerical_normalize(
            self.df[["distance"]], mean, std)
        self.folder = folder
        self.device = device

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def image2tensor(pic: Image, device: str = None) -> torch.Tensor:
        img = torch.as_tensor(np.array(pic, copy=True), device=device,
                              dtype=torch.float32) / 255
        img = img.view(pic.size[1], pic.size[0], 3)
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1))
        return img

    def numerical_normalize(self,
                            data: pd.DataFrame | float | int,
                            mean, std) -> pd.DataFrame | float:
        """
        Returns normalized data.
        """

        data = (data - mean) / std
        return data

    def __getitem__(self, key: int) -> tuple[Tensor, Tensor]:
        filename = f"{self.folder}/{self.df['filename'].iloc[key]}"

        return (self.image2tensor(Image.open(filename), device=self.device)[:,
                100:500, 300:700],
                torch.tensor(self.df["distance"].iloc[key], device=self.device,
                             dtype=torch.float32))


def get_height(dataloader, model) -> float:
    model.eval()
    pred = -1

    with torch.no_grad():
        for image, dist in dataloader:
            pred = model((image, dist))

    return pred


def model_calculate(photo_path: str, folder: str,
               coords: tuple[str, str], model_path: str):
    device = (
        "cuda"  # nvidia GPU
        if torch.cuda.is_available()
        else "mps"  # mac GPU
        if torch.backends.mps.is_available()
        else "cpu"
    )
    train_df = pd.read_csv("data/train/train.csv")
    mean = train_df["height"].mean()
    std = train_df["height"].std()
    data = {"filename": [photo_path], "building": [coords[0]],
            "camera": [coords[1]]}
    df = pd.DataFrame(data)

    transformer = CoordinatesTransformPipeline(df)
    df = transformer.transform()

    dataset = BuildingDataset(folder, df, device, mean, std)

    dataloader = (torch.utils
                       .data.DataLoader(dataset,
                                        batch_size=4, shuffle=True,
                                        pin_memory=(device != 'cuda')))
    model = torch.load(model_path, map_location=torch.device(device))

    result = get_height(dataloader, model)
    return result * std + mean
