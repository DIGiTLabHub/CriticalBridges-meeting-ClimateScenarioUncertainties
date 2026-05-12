__all__ = ["create_all_plots"]


def __getattr__(name):
    if name == "create_all_plots":
        from .visualization import create_all_plots

        return create_all_plots
    raise AttributeError(name)
