# Thanks Grander78498
import torch
import pandas as pd
from src.main import CoordinatesTransformPipeline, BuildingDataset


def get_height(dataloader, model) -> float:
    model.eval()
    pred = -1

    with torch.no_grad():
        for image, dist in dataloader:
            pred = model((image, dist))

    return pred


def model_calculate(photo_path: str,
                    folder: str,
                    coords: tuple[str, str],
                    model_path: str) -> float:
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

    dataset = BuildingDataset(folder, df, device, mean=mean, std=std)
    dataloader = (torch.utils
                       .data.DataLoader(dataset,
                                        batch_size=4, shuffle=True,
                                        pin_memory=True))
    model = torch.load(model_path, map_location=torch.device(device))

    result = get_height(dataloader, model)
    return result * std + mean
