import numpy as np


def attack(sample: dict) -> dict:
    image = np.asarray(sample["image"])
    return {"image": np.ascontiguousarray(image.copy())}
