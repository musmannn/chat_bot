import os
from prompt import rag_system_prompt,rag_history_prompt,chat_prompt
from langchain_community.chat_message_histories import ChatMessageHistory
import uuid
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import time
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
api_key = os.environ['OPENAI_API_KEY']
set_debug(False)
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

"""
What it should do:
    chat without a file present
    chat with a file present
    use the previous chat messages 
    

if there is no file:
    create  the same chain however the context should be a empty string
    
    

"""


class Chat:
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
    #Print all the attributes on every funciton call
    def __str__(self):
        return f"Chat(file_path={self.file_path},file_uploaded={self.file_uploaded},current_file={self.current_file})"
    def add_source(self,file_path):
        print("Adding source")
        self.file_uploaded = True
        print(f" in add source funcion {self.file_uploaded}")
        self.qa_chain = create_stuff_documents_chain(self.chat_model,rag_system_prompt)
        self.current_file = file_path
        data = create_retriever(file_path,self.chunking_method)
        self.retriever = data['retriever']
        self.db = data['vectordb']
        self.history_retriever = create_history_aware_retriever(self.llm,self.retriever,rag_history_prompt)
        self.source_chain = create_retrieval_chain(self.history_retriever,self.qa_chain)
    def get_session_history(self,session_id):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    def run(self,query,username="Usman"):
        
        
        # self.chat_model = ChatHuggingFace(llm=self.llm,max_new_tokens=1000)
        
        if self.file_uploaded == True:
            
            if self.file_path != self.current_file:
                print("RETURNED FROM FILE 1")
                response = self.source_chain.invoke({"input":query,'username':username})
                print(f"response returned is {response['context']}")
                return response['answer']
            else:
                print("RETURNED FROM FILE 2")
                response = self.source_chain.invoke({"input":query,'username':username})
                print(f"response returned is {response['context']}")
                return response['answer']
        else:
            # print(f"File_Uplaoded:{file_uploaded}")
            print("RETURNED FROM simple")
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
