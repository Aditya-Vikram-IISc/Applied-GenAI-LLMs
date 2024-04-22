from config.config import CREDS
from openai import OpenAI


class OpenAI_MasterClass:
    def __init__(self, embedding_model:str= "text-embedding-ada-002",
                 llm_model:str = "chatgpt",
                key = CREDS["OPENAI_API_CREDS"],
                ):
        
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.client = OpenAI(api_key= key)

    def create_embeddings(self, text:list[str]):
        result = self.client.embeddings.create(
                            input=text,
                            model= self.embedding_model
                        )
        
        return result.data[0].embedding
    
