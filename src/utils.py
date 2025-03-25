"""Utilities for metrics processing."""

import os

import imageio
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


MODES = ["rewards", "energies"]
STAGES = ["train", "eval"]
METRICS = ["reward", "energy", "length"]
TITLES = ["Cumulative Regular Reward", "Cumulative Energy Reward", "Episode Length"]
LIMITS = [9360, 11772, 1000]


def mp4_to_gif(folder: str) -> None:
    """Convert MP4 video to GIF.

    Parameters
    ----------
    folder : str
        The folder containing MP4 files to be converted.

    """
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4")]
    gif_paths = [p[: p.rfind(".")] + ".gif" for p in paths]

    for video_path, gif_path in tqdm.tqdm(
        zip(paths, gif_paths), total=len(paths), desc="Video files conversion"
    ):
        with imageio.get_reader(video_path) as reader:
            fps = reader.get_meta_data()["fps"]

            writer = imageio.get_writer(gif_path, fps=fps, loop=0)
            for frame in reader:
                writer.append_data(frame)
            writer.close()

        os.remove(video_path)


def moving_average(input: np.ndarray, n: int = 500, mode="valid") -> tuple[np.ndarray]:
    """
    Calculate the moving average of a given input array.

    Parameters
    ----------
    input : np.ndarray
        The input array for which the moving average is to be calculated.
    n : int, optional
        The number of elements to include in the moving average window. Default is 500.
    mode : str, optional
        The mode parameter determines the type of convolution. 
        'valid' returns output of length max(M, N) - min(M, N) + 1. 
        'same' returns output of length max(M, N). Default is 'valid'.

    Returns
    -------
    tuple[np.ndarray]
        A tuple containing:
        - steps : np.ndarray
            The array of step indices corresponding to the moving average values.
        - output : np.ndarray
            The array of moving average values.

    """
    output = np.convolve(np.array(input).flatten(), np.ones(n), mode=mode) / n
    if mode == "valid":
        steps = np.arange(output.size) + n // 2
    elif mode == "same":
        steps = np.arange(output.size)
    return steps, output


def plot_metrics(model: str, smooth: int = 100, alpha: float = 0.2) -> Figure:
    """
    Plots the metrics for a given model.

    Parameters
    ----------
    model : str
        The name of the model to plot metrics for.
    smooth : int, optional
        The window size for smoothing the metrics, by default 100.
    alpha : float, optional
        The transparency level for the raw data plots, by default 0.2.

    Returns
    -------
    Figure
        The matplotlib figure containing the plotted metrics.

    Notes
    -----
    This function reads CSV files containing the metrics for different stages and modes,
    and plots them in a 2x3 grid of subplots. The metrics are smoothed using a moving 
    average.

    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"{model.upper()} agent metrics", fontsize=20)
    model = model.lower()

    data = {}

    for stage in STAGES:
        data[stage] = {}
        for reward in MODES:
            data[stage][reward] = pd.read_csv(f"./results/{model}-{stage}-{reward}.csv")

    data["eval"]["random"] = pd.read_csv("./results/random.csv")

    for row, stage in enumerate(data.keys()):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            ax.axhline(LIMITS[col], color="k", linestyle="--", label="Metric's limit")
            for color, mode in enumerate(MODES):
                # Plot raw data
                ax.plot(data[stage][mode][metric], color=f"C{color}", alpha=alpha)
                reward_label = "Regular" if mode == "rewards" else "Energy"
                ax.plot(
                    *moving_average(data[stage][mode][metric], smooth),
                    color=f"C{color}",
                    linewidth=2,
                    label=f"{reward_label} reward is used",
                )
                if stage == "eval" and mode == "rewards":
                    ax.plot(
                        *moving_average(data["eval"]["random"][metric], smooth),
                        color="C2",
                        linewidth=2,
                        label="Random agent",
                    )
            mode_label = "Validation" if stage == "eval" else "Train"
            ax.set_title(f"{mode_label}: {TITLES[col]}")
            ax.set_xlabel("Episodes")
            ax.set_ylabel(TITLES[col])
            ax.legend()
            ax.grid()

    plt.tight_layout()
    plt.show()
    return fig

