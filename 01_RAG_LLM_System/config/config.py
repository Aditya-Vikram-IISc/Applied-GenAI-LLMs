# config.config

import os
from dotenv import load_dotenv
from configparser import ConfigParser


# get the base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_PATH, "config")

# load the .env file
load_dotenv(os.path.join(CONFIG_PATH, '.env'),verbose=True)


# read the config using configparser
config = ConfigParser()
config.read(os.path.join(CONFIG_PATH, "config.ini"))

# Get the info of Vector DB
RAG_INDEX = config["RAG_SYSTEM"]["index_name"]

# Get the info of LLM model
LLM_MODEL_PARAMS = {
                    "model_name" : str(config["LLM_SYSTEM"]["model_name"]),
                    "temperature": float(config["LLM_SYSTEM"]["temperature"]),
                    "max_tokens": int(config["LLM_SYSTEM"]["max_tokens"]),
                    "top_p": float(config["LLM_SYSTEM"]["top_p"]),
                    "frequency_penalty": float(config["LLM_SYSTEM"]["frequency_penalty"]),
                    "presence_penalty": float(config["LLM_SYSTEM"]["presence_penalty"]),
                    "stop": str(config["LLM_SYSTEM"]["stop"]),
                    "embeddings": str(config["LLM_SYSTEM"]["embeddings"])
                    }

# Get the API creds
CREDS = {
        "PINECONE_API_CREDS" : str(os.getenv("PINECONE_API_CREDS")),
        "OPENAI_API_CREDS"  : str(os.getenv("OPENAI_API_CREDS"))
        }


if __name__ == "__main__":
    print("RAG DB Index name", RAG_INDEX)
    print("LLM_MODEL_PARAMS", LLM_MODEL_PARAMS)
    print(CREDS)