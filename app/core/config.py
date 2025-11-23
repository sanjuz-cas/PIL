import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    PROJECT_NAME: str = "indxai OS"
    VERSION: str = "1.0.0"

    # AI Model Settings
    TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    LATENT_DIM: int = 64
    EMBEDDING_DIM: int = 384

    # --- GOOGLE SEARCH CREDENTIALS ---
    # 1. The ID you just gave me:
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")

    # 2. The Key you get from Cloud Console (starts with AIza...):
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")


settings = Settings()
