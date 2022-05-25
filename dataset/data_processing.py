from pathlib import Path
import numpy as np 
from scipy import interpolate
from cv2_rolling_ball import subtract_background_rolling_ball
import matplotlib.pyplot as plt
import sys

def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize the data to the range [0, 1]
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def fix_spacing(data: np.ndarray, spacing: float = 0.02) -> np.ndarray:
    """
    Fix the spacing of the data to the given value
    """
    f = interpolate.CubicSpline(data[:, 0], data[:, 1])
    ret = np.arange(data[:, 0].min(), data[:,0].max(), spacing)
    xs = f(ret)
    return np.vstack([ret, xs]).T

def remove_background(data: np.ndarray) -> np.ndarray:
    """
    Remove the background from the data
    """
    pixels = (data*255).astype(np.uint8)
    pixels = np.tile(pixels, (3, 1))
    pixels, _ = subtract_background_rolling_ball(pixels, 400,
        light_background=False, use_paraboloid=True, do_presmooth=False)
    
    return pixels[0]/255

def zero_pad(data: np.ndarray, pad_size: int = int((160-5)/0.02)) -> np.ndarray:
    """
    Zero pad the data
    """
    return np.pad(data, ((0, pad_size-len(data)), (0, 0)), 'constant', constant_values=0)

if __name__ == '__main__':
    path = Path(__file__).resolve().parent / "train/raw"
    path_save = Path(__file__).resolve().parent / "train/processed"
    nfiles = len(list(path.glob("*.xy")))
    for i, f in enumerate(path.glob("*.xy")):
        print(f"processing {i}/{nfiles}")
        data = np.genfromtxt(f)
        data[:, 1] = normalize(data[:, 1])
        data = fix_spacing(data)
        data[:, 1] = remove_background(data[:, 1])
        data = zero_pad(data)
        np.save(path_save / (f.name.split(".")[0] + ".npy"), data)