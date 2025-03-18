"""Utilities for metrics processing."""

import os

import imageio
import numpy as np
import tqdm


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


def cumulative(input: np.ndarray) -> tuple[np.ndarray]:
    """Get the cumulative value."""
    input = np.array(input).flatten()
    temp = 0
    for i in range(input.size):
        temp += input[i]
        input[i] = temp
    steps = np.arange(input.size)
    return steps, input
