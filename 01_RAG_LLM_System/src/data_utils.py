# src.data_utils
from tqdm.auto import tqdm
import joblib
from .llm_utils import OpenAI_MasterClass
from .vectordb_utils import Pinecone_Index
from config.config import RAG_INDEX

def create_data_for_uploading_to_PineconeDB(data: list[str]):
    # create an OpenAI_MasterClass instance
    client = OpenAI_MasterClass()
    vectors = []
    for index, text in tqdm(enumerate(data)):
        # Get embeddings for the data
        embeddings = client.create_embeddings(text)
        vectors.append({
            "id": str(index),
            "values" : embeddings,
            "metadata": {"text": text}
                    })
    
    ## sample vectors
    # vectors=[
    #     {
    #         "id": "vec1", 
    #         "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
    #         "metadata": {"genre": "drama"}
    #     }, {
    #         "id": "vec2", 
    #         "values": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
    #         "metadata": {"genre": "action"}
    #     }]

    return vectors



def postprocess_retrieval_data(retrieved_data, threshold: float = 0.8):
    response = ""
    for x in retrieved_data["matches"]:
        if x["score"] <= 0.8:
            continue
        response += x["metadata"]["text"]
    
    return response



if __name__ == "__main__":
    # read the data pickle
    data = joblib.load("text_data.pkl")

    # create an instance of pinecone 
    pc = Pinecone_Index()
    # create an index in PineCone of your choice
    index_name = RAG_INDEX
    pc.create_index(indexname = index_name)  # step gets skipped if indexname exists

    # create the data to be uploaded to PineCone
    vectors = create_data_for_uploading_to_PineconeDB(data = data[:5])

    # upsert the data to the index
    pc.upsert_embeddings_to_index(index_name= index_name, vectors = vectors)
    

    
    
