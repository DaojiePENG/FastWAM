from .fs import ensure_dir

__all__ = ["ensure_dir", "save_mp4"]


def __getattr__(name: str):
    # Video dependencies are optional for LeapBot's action-only inference path.
    if name == "save_mp4":
        from .video_io import save_mp4

        return save_mp4
    raise AttributeError(name)
