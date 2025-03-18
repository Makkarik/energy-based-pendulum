"""Utilities for metrics processing."""

import os

import imageio


def mp4_to_gif(folder: str) -> None:
    """Convert MP4 video to GIF.

    Parameters
    ----------
    folder : str
        The folder containing MP4 files to be converted.

    """
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4")]
    gif_paths = [p[: p.rfind(".")] + ".gif" for p in paths]

    for video_path, gif_path in zip(paths, gif_paths):
        with imageio.get_reader(video_path) as reader:
            fps = reader.get_meta_data()["fps"]

            writer = imageio.get_writer(gif_path, fps=fps, loop=0)
            for frame in reader:
                writer.append_data(frame)
            writer.close()

        os.remove(video_path)
