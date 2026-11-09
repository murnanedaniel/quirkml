"""Hit-filter entry point: python filter_main.py {fit,predict} --config configs/filter_mu60.yaml ..."""
from lightning.pytorch.cli import ArgsType

from ml_tracking.experiments.quirk.data import QuirkDataModule
from ml_tracking.experiments.quirk.filter_model import QuirkFilterModel
from ml_tracking.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(model_class=QuirkFilterModel, datamodule_class=QuirkDataModule, args=args, parser_kwargs={"default_env": True})


if __name__ == "__main__":
    main()
