from pathlib import Path
import numpy as np 
from scipy import interpolate
from cv2_rolling_ball import subtract_background_rolling_ball
import matplotlib.pyplot as plt
import sys

plt.rcParams['font.size'] = 22

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
    f = list(path.glob("*.xy"))[0]
    data = np.genfromtxt(f)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax = ax.ravel()
    ax[0].set_title("Raw data")
    ax[0].plot(data[:, 0], data[:, 1])
    ax[0].set_xlabel(r"2 $\theta$")
    ax[0].set_ylabel("Intensity")
    data[:, 1] = normalize(data[:, 1])
    data = fix_spacing(data)
    data[:, 1] = remove_background(data[:, 1])
    ax[1].set_title("Processed data")
    ax[1].plot(data[:, 0], data[:, 1])
    ax[1].set_xlabel(r"2 $\theta$")
    ax[1].set_ylabel("Intensity")
    plt.tight_layout()
    plt.show()