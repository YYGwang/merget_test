from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env 파일 혹은 환경변수에서 자동으로 읽어옴
    ENV: str = 'server'
    REGION: str
    OPENAI_API_KEY: str
    TAVILY_API_KEY: str

    USER_POOL_ID: str
    APP_CLIENT_ID: str



    # CORS 설정
    ALLOWED_ORIGINS: list = ["http://localhost:5173"]

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8"
    )


settings = Settings()