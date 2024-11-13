from pathlib import Path


def get_project_root_path() -> Path:
    return Path(__file__).parent.parent


def get_project_config_path() -> Path:
    root = get_project_root_path()
    Path(str(root) + "/config").mkdir(parents=True, exist_ok=True)
    return Path(str(root) + "/config")


def get_project_model_path() -> Path:
    root = get_project_root_path()
    Path(str(root) + "/models_trained").mkdir(parents=True, exist_ok=True)
    return Path(str(root) + "/models_trained")


def get_project_model_checkpoint_path() -> Path:
    root = get_project_root_path()
    Path(str(root) + "/models_checkpoints").mkdir(parents=True, exist_ok=True)
    return Path(str(root) + "/models_checkpoints")


def get_project_model_raytune_checkpoint_path() -> Path:
    root = get_project_root_path()
    Path(str(root) + "/models_raytune_checkpoints").mkdir(parents=True, exist_ok=True)
    return Path(str(root) + "/models_raytune_checkpoints")


def get_raytune_log_path() -> Path:
    root = get_project_root_path()
    Path(str(root) + "/models_raytune_checkpoints/model_logs").mkdir(
        parents=True, exist_ok=True
    )
    return Path(str(root) + "/models_raytune_checkpoints/model_logs")


def get_project_data_path() -> Path:
    root = get_project_root_path()
    Path(str(root) + "/dataset").mkdir(parents=True, exist_ok=True)
    return Path(str(root) + "/dataset")
