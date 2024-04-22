from config.config import CREDS, LLM_MODEL_PARAMS
from openai import OpenAI


class OpenAI_MasterClass:
    def __init__(self, embedding_model:str= "text-embedding-ada-002",
                 llm_model:str = "chatgpt",
                key = CREDS["OPENAI_API_CREDS"],
                ):
        
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.client = OpenAI(api_key= key)

    def create_embeddings(self, text:str):
        result = self.client.embeddings.create(
                            input=text,
                            model= self.embedding_model
                        )
        
        return result.data[0].embedding
    
    def RAG_based_output_from_ChatGPT(self, user_prompt:str, rag_context: str):
        SYSTEM_PROMPT = """You are a helpful, respectful and honest INTP-T AI Assistant named Gathnex AI. You are talking to a human User.
        Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
        You also have access to RAG vectore database access which has Indian Law data. Be careful when giving response, sometime irrelevent Rag content will be there so give response effectivly to user based on the prompt.
        You can speak fluently in English.
        Note: Sometimes the Context is not relevant to Question, so give Answer according to that sutiation.
        """
        response = self.client.chat.completions.create(
                    model = LLM_MODEL_PARAMS["model_name"],
                    messages = [
                        {f"role": "system", "content": SYSTEM_PROMPT},
                        {f"role": "user", "content": rag_context +", Prompt: "+ user_prompt}
                                ]
                    )

        return response.choices[0].message.content
    
