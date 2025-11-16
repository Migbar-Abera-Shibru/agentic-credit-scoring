# config.py - FIXED PATH HANDLING
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "Agentic Credit Scoring MaaS"
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Data Settings - Force Path objects
    BASE_DIR: Path = Path(__file__).parent.absolute()
    DATA_PATH: Path = BASE_DIR / "data" / "raw"
    MODEL_PATH: Path = BASE_DIR / "models" / "trained_models"
    
    # Model Settings
    TARGET_COLUMN: str = "SeriousDlqin2yrs"
    ID_COLUMN: str = "Unnamed: 0"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    PROBLEM_TYPE: str = "classification"
    
    class Config:
        env_file = ".env"
        # Ensure Path objects are properly handled
        arbitrary_types_allowed = True

# Convert to Path objects if they come as strings from environment
def ensure_path(path):
    if isinstance(path, str):
        return Path(path)
    return path

settings = Settings()
settings.BASE_DIR = ensure_path(settings.BASE_DIR)
settings.DATA_PATH = ensure_path(settings.DATA_PATH)
settings.MODEL_PATH = ensure_path(settings.MODEL_PATH)

print(f"🔧 Config paths (verified):")
print(f"   BASE_DIR: {settings.BASE_DIR} (type: {type(settings.BASE_DIR)})")
print(f"   DATA_PATH: {settings.DATA_PATH} (type: {type(settings.DATA_PATH)})")
print(f"   MODEL_PATH: {settings.MODEL_PATH} (type: {type(settings.MODEL_PATH)})")