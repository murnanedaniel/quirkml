"""Quirk MaskFormer entry point: python main.py {fit,validate,predict} --config configs/quirk.yaml ..."""

from lightning.pytorch.cli import ArgsType

from ml_tracking.experiments.quirk.data import QuirkDataModule
from ml_tracking.experiments.quirk.model import QuirkModel
from ml_tracking.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(model_class=QuirkModel, datamodule_class=QuirkDataModule, args=args, parser_kwargs={"default_env": True})


if __name__ == "__main__":
    main()
