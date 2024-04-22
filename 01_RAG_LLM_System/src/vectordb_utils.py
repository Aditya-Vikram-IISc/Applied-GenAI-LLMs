from pinecone import Pinecone, ServerlessSpec
from config.config import CREDS


class Pinecone_Index:
    def __init__(self, key:str = CREDS["PINECONE_API_CREDS"]):
        self.pc = Pinecone(api_key = key)
    

    def create_index(self, indexname:str, dimension:int = 1536, metric:str = "cosine"):
        if indexname not in self.pc.list_indexes().names():
            self.pc.create_index(
                                name= indexname,
                                dimension=dimension,
                                metric=metric, #euclidean, cosine
                                spec=ServerlessSpec(
                                    cloud='aws',
                                    region='us-east-1'
                                                    )
                                )
    
    def upsert_embeddings_to_index(self, index_name:str, vectors:list[dict]):
        index = self.pc.Index(index_name)
        index.upsert(vectors = vectors)
        print("Vectors upserted succesfully !")









if __name__ == "__main__":
    pc_index = Pinecone_Index()
    pc_index.create_index(indexname = "llmrag")
    print(pc_index.pc.list_indexes().names())
    