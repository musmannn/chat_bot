from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

key = os.getenv('OPENAI_API_KEY')
print(key)