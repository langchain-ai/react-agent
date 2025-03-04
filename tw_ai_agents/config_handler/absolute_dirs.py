from pathlib import Path


def get_root_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_mock_db_folder() -> Path:
    return get_root_path() / "mock_db"
