from lightning.pytorch.cli import ArgsType

from ml_tracking.experiments.colliderml.data import ColliderMLDataModule
from ml_tracking.experiments.colliderml.model import ColliderMLModel
from ml_tracking.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=ColliderMLModel,
        datamodule_class=ColliderMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
