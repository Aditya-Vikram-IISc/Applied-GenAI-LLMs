from fastapi import FastAPI
from pydantic import BaseModel
from config.config import RAG_INDEX
from src import Pinecone_Index, OpenAI_MasterClass,postprocess_retrieval_data


# Create a Pydantic Baseclass
class RAGoutput(BaseModel):
    prompt : str


# Create an instance of app
app = FastAPI()


# Create instances of OpenAI_MasterClass, Pinecone_Index
openai_client = OpenAI_MasterClass()
pinecone_client = Pinecone_Index()



@app.get("/")
async def health_status():
    return {"text": "Health is up and running!"}


@app.post("/rag-based-output-generation", status_code=200)
async def rag_based_output_generation(data : RAGoutput):
    data = data.model_dump()
    prompt = data["prompt"]

    # Create Input Embeddings
    # create embeddings using OpenAI text-ada-002 model
    embeddings_vector = openai_client.create_embeddings(text = prompt)

    # Retrieve it using RAG i.e. search for similar text in VectorDB in Pinecone
    pc = Pinecone_Index()
    retrieved_data = pc.retrieve_data_from_vectorDB(vector=embeddings_vector, index_name= RAG_INDEX)
    rag = postprocess_retrieval_data(retrieved_data)

    # generate output
    output = openai_client.RAG_based_output_from_ChatGPT(user_prompt = prompt, rag_context = rag)
    return output


### run it by command: uvicorn main:app --host 127.0.0.1 --port 80 --reload