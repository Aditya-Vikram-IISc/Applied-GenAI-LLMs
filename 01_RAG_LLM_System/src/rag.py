from pinecone import Pinecone, ServerlessSpec
from config.config import PINECONE_API_CREDS


class Pinecone_Index:
    def __init__(self, creds:str = PINECONE_API_CREDS):
        self.pc = Pinecone(api_key = creds)
    

    def create_index(self, indexname:str):
        if indexname not in self.pc.list_indexes().names():
            self.pc.create_index(
                                name= indexname,
                                dimension=1536,
                                metric='cosine', #euclidean
                                spec=ServerlessSpec(
                                    cloud='aws',
                                    region='us-west-2'
                                                    )
                                )
    

if __name__ == "__main__":
    pc_index = Pinecone_Index()
    pc_index.create_index(indexname = "llmrag")
    print(pc_index.pc.list_indexes().names())
    