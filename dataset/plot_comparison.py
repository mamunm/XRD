from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path_raw = Path(__file__).resolve().parent / "train/raw"
    path_proc = Path(__file__).resolve().parent / "train/processed"
    
    for f in path_raw.glob("*.xy"):
        fig, ax = plt.subplots(2, 1)
        ax = ax.ravel()
        data_raw = np.genfromtxt(f)
        data_proc = np.load(
            path_proc / (f.name.split(".")[0] + ".npy"))[()]
        ax[0].plot(data_raw[:, 0], data_raw[:, 1])
        ax[1].plot(data_proc[:, 0], data_proc[:, 1])
        plt.show()

        