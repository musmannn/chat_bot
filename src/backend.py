import os
from prompt import rag_system_prompt,rag_history_prompt,chat_prompt
from langchain_community.chat_message_histories import ChatMessageHistory
import uuid
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import time
import json
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from functools import wraps
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

from utils import create_retriever
from langchain_core.messages import HumanMessage
# from utils import indexing,retrieval,time_it
import os
from langchain_core.globals import set_debug,get_debug

#Setting up Enviroment Variables
api_key = os.getenv('OPENAI_API_KEY')
set_debug(False)
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

"""
Todos:
    -- Use multithreading to speed up the process
    -- Add more comments
    -- add the ability to work with more resources
    -- how to handle chat history(if it just grows and grows)
    -- slinding context window?
    -- Streaming integration with langchain (this one is too slow and we shouldn't be doing this)
    -- Improve prompt to only use context if needed


"""

class Chat:
    """
    A class representing a chatbot.

    Attributes:
    - file_path (str): The path to the file containing the source data.
    - model (str): The name or ID of the Hugging Face model to use for chat generation.
    - embedding (str): The type of embedding to use for the chat model.
    - chunking_method (str): The method used for semantic chunking.

    Methods:
    - __init__(self, file_path=None, model="mistralai/Mistral-7B-Instruct-v0.2", embedding="", chunking_method="semantic_chunking"): Initializes a Chat instance.
    - add_source(self, file_path): Adds a source file for the chatbot to use.
    - get_session_history(self, session_id): Retrieves the chat message history for a given session ID.
    - run(self, query, username="Usman"): Runs the chatbot with the given query and username.
    """

    def __init__(self,file_path=None,model="mistralai/Mistral-7B-Instruct-v0.2",embedding="",chunking_method="semantic_chunking"):
        self.file_path = file_path
        self.llm = HuggingFaceEndpoint(repo_id=model,temperature=0.5,huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,max_new_tokens=4096)
        self.session_id = str(uuid.uuid4())
        self.store = {}
        self.file_uploaded = False
        self.current_file = None
        self.db = None
        self.retriever = None
        self.chunking_method = chunking_method
        self.chat_model =  ChatOpenAI(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            base_url="https://api-inference.huggingface.co/v1",
            api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN'),max_tokens=1000)
        self.retriever_chain = None
        self.chat_chain = None
    # add the source file and create the relevent chains 
    def add_source(self,file_path):
        print(f"Adding the following source: {file_path}")
        self.file_uploaded = True
        self.current_file = file_path
        self.qa_chain = create_stuff_documents_chain(self.chat_model,rag_system_prompt)
        data = create_retriever(file_path,self.chunking_method)
        self.retriever = data['retriever']
        self.db = data['vectordb']
        self.history_retriever = create_history_aware_retriever(self.llm,self.retriever,rag_history_prompt)
        self.source_chain = create_retrieval_chain(self.history_retriever,self.qa_chain)
    # Used to retrieve the chat session history
    def get_session_history(self,session_id):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    # Used to invoke the chain and generate response
    def run(self,query,username="Usman"):
        if self.file_uploaded == True:
            # Goes in this block if the user has uploaded the file/source
            print("RETURNED FROM FILE 1")
            response = self.source_chain.invoke({"input":query,'username':username})
            analysis  = []
            for data in response['context']:
                data = {'metadata':[data.metadata],'page_content':[data.page_content]}
                analysis.append(data)
            with open("documents.json", "w") as file:
                json.dump(analysis, file, indent=4)
            print(f"response returned is {response['context']}")
            return response['answer']
            
        else:
            #Change this so it only creates the chain if it hasnt created before
            if self.chat_chain:
                response = self.simple_chain.invoke({"input":query,'username':username},{'configurable':{"session_id":self.session_id}})
                print(response)
                return response.content
            else:
                self.chat_chain = chat_prompt | self.chat_model
                self.simple_chain = RunnableWithMessageHistory(
                    self.chat_chain,
                    self.get_session_history,
                    history_messages_key="messages"
                )
                response = self.simple_chain.invoke({"input":query,'username':username},{'configurable':{"session_id":self.session_id}})
                print(response)
                return response.content


if __name__ == "__main__":
    bot = Chat()
    
    user_inp = input("User:")
    while user_inp != "n":
        start_time = time.time()
        asd = bot.run(user_inp)
        end_time = time.time()
        response_time = end_time - start_time
        print(asd)
        print(f"(Response generated in {response_time:.2f} seconds)")
        user_inp = input("User:")
