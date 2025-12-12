from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class Settings:
    data_dir: str = "data"
    master_file: str = "master_dataset.csv"
    model_ready_file: str = "model_ready_dataset.csv"
    disaster_file: str = "natural_disasters.csv"
    epidemic_file: str = "epidemic_and_pandemics.csv"
    gdelt_file: str = "gdelt_daily_world_2013_present.csv"
    output_dir: str = "outputs"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    test_size_ratio: float = 0.2
    test_split: float = 0.2
    random_seed: int = 42
    gb_params: Dict[str, float] = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "learning_rate": 0.01,
            "max_depth": 3,
            "subsample": 0.7,
        }
    )
    run_output_dir: Optional[str] = None

    @property
    def master_path(self) -> str:
        return os.path.join(self.data_dir, self.master_file)

    @property
    def model_ready_path(self) -> str:
        return os.path.join(self.data_dir, self.model_ready_file)

    @property
    def disaster_path(self) -> str:
        return os.path.join(self.data_dir, self.disaster_file)

    @property
    def epidemic_path(self) -> str:
        return os.path.join(self.data_dir, self.epidemic_file)

    @property
    def gdelt_path(self) -> str:
        return os.path.join(self.data_dir, self.gdelt_file)


ENV_MAP = {
    "data_dir": "DATA_DIR",
    "master_file": "MASTER_FILE",
    "model_ready_file": "MODEL_READY_FILE",
    "disaster_file": "DISASTER_FILE",
    "epidemic_file": "EPIDEMIC_FILE",
    "gdelt_file": "GDELT_FILE",
    "output_dir": "OUTPUT_DIR",
    "run_id": "RUN_ID",
    "test_size_ratio": "TEST_SIZE_RATIO",
    "test_split": "TEST_SPLIT",
    "random_seed": "RANDOM_SEED",
    "gb_n_estimators": "GB_N_ESTIMATORS",
    "gb_learning_rate": "GB_LEARNING_RATE",
    "gb_max_depth": "GB_MAX_DEPTH",
    "gb_subsample": "GB_SUBSAMPLE",
}


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", help="Base directory for input data")
    parser.add_argument("--master-file", help="Master dataset filename")
    parser.add_argument("--model-ready-file", help="Model-ready dataset filename")
    parser.add_argument("--disaster-file", help="Natural disasters filename")
    parser.add_argument("--epidemic-file", help="Epidemics filename")
    parser.add_argument("--gdelt-file", help="GDELT filename")
    parser.add_argument("--output-dir", help="Base directory for outputs")
    parser.add_argument("--run-id", help="Override run identifier for outputs")
    parser.add_argument("--test-size-ratio", type=float, help="Test size ratio for baseline models")
    parser.add_argument("--test-split", type=float, help="Train/test split for model-ready data")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--gb-n-estimators", type=int, help="Gradient boosting: number of estimators")
    parser.add_argument("--gb-learning-rate", type=float, help="Gradient boosting: learning rate")
    parser.add_argument("--gb-max-depth", type=int, help="Gradient boosting: max depth")
    parser.add_argument("--gb-subsample", type=float, help="Gradient boosting: subsample ratio")


def _env_or_default(key: str, default: Optional[str]) -> Optional[str]:
    env_key = ENV_MAP.get(key, key.upper())
    return os.environ.get(env_key, default)


def load_config(argv: Optional[list[str]] = None) -> Settings:
    defaults = Settings()

    parser = argparse.ArgumentParser(add_help=False)
    add_common_arguments(parser)
    args, _ = parser.parse_known_args(argv)

    def choose(name: str, default_value):
        arg_value = getattr(args, name.replace('-', '_'), None)
        env_value = _env_or_default(name, None)
        if arg_value is not None:
            return arg_value
        if env_value is not None:
            if isinstance(default_value, float):
                return float(env_value)
            if isinstance(default_value, int):
                return int(env_value)
            return env_value
        return default_value

    gb_params = defaults.gb_params.copy()
    if args.gb_n_estimators is not None or _env_or_default("gb_n_estimators", None):
        gb_params["n_estimators"] = choose("gb_n_estimators", gb_params["n_estimators"])
    if args.gb_learning_rate is not None or _env_or_default("gb_learning_rate", None):
        gb_params["learning_rate"] = choose("gb_learning_rate", gb_params["learning_rate"])
    if args.gb_max_depth is not None or _env_or_default("gb_max_depth", None):
        gb_params["max_depth"] = choose("gb_max_depth", gb_params["max_depth"])
    if args.gb_subsample is not None or _env_or_default("gb_subsample", None):
        gb_params["subsample"] = choose("gb_subsample", gb_params["subsample"])

    cfg = Settings(
        data_dir=choose("data_dir", defaults.data_dir),
        master_file=choose("master_file", defaults.master_file),
        model_ready_file=choose("model_ready_file", defaults.model_ready_file),
        disaster_file=choose("disaster_file", defaults.disaster_file),
        epidemic_file=choose("epidemic_file", defaults.epidemic_file),
        gdelt_file=choose("gdelt_file", defaults.gdelt_file),
        output_dir=choose("output_dir", defaults.output_dir),
        run_id=choose("run_id", defaults.run_id),
        test_size_ratio=choose("test_size_ratio", defaults.test_size_ratio),
        test_split=choose("test_split", defaults.test_split),
        random_seed=choose("seed", defaults.random_seed),
        gb_params=gb_params,
    )

    cfg.run_output_dir = os.path.join(cfg.output_dir, cfg.run_id)
    os.makedirs(cfg.run_output_dir, exist_ok=True)
    return cfg

