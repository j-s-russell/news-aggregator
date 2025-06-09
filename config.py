import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='/Users/joerussell/desktop/projects/news-aggregator/.env', override=True)

USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DBNAME = os.getenv("DBNAME")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def validate_config():
    creds = ['USER', 'PASSWORD', 'HOST', 'PORT', 'DBNAME', 'NEWS_API_KEY', 'GOOGLE_API_KEY']
    missing = [cred for cred in creds if not os.getenv(cred)]
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    print("✅ All required environment variables are set")

if __name__ == "__main__":
    validate_config()