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
TITLES = ["Regular Reward", "Energy Reward", "Episode Length"]


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
    """Get the moving average."""
    output = np.convolve(np.array(input).flatten(), np.ones(n), mode=mode) / n
    if mode == "valid":
        steps = np.arange(output.size) + n // 2
    elif mode == "same":
        steps = np.arange(output.size)
    return steps, output


def plot_metrics(model: str, smooth: int = 100) -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"{model.upper()} Model Metrics", fontsize=16)
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
            for color, mode in enumerate(MODES):
                # Plot raw data
                ax.plot(data[stage][mode][metric], color=f"C{color}", alpha=0.3)
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
            ax.set_title(f"Train: {TITLES[col]}")
            ax.set_xlabel("Episodes")
            ax.set_ylabel(TITLES[col])
            ax.legend()
            ax.grid()

    plt.tight_layout()
    plt.show()
    return fig

