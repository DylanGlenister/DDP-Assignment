from pydantic_settings import BaseSettings

class Settings(BaseSettings):
	database: str
	database_host: str
	database_username: str
	database_password: str

	class Config:
		env_file = ".env"
		env_file_encoding = 'utf-8'

SETTINGS = Settings()  # type: ignore
