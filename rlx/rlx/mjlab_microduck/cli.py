"""Console entry points for the vendored mjlab MicroDuck workflow."""

from __future__ import annotations


def list_envs() -> object:
    """List registered mjlab tasks after loading the MicroDuck plugin."""
    import rlx.mjlab_microduck.tasks  # noqa: F401
    from mjlab.scripts.list_envs import main

    return main()


def play() -> object:
    """Run mjlab's interactive checkpoint player with MicroDuck tasks loaded."""
    import rlx.mjlab_microduck.tasks  # noqa: F401
    from mjlab.scripts.play import main

    return main()


def evaluate_running() -> None:
    """Run the deterministic running-checkpoint evaluation battery."""
    import tyro

    from .tools.evaluate_running_checkpoint import Config, main

    main(tyro.cli(Config))
