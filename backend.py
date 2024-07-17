import os 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
import uuid
from langchain.chains import create_history_aware_retriever
import time 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from functools import wraps
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"





def time_it(method_name):
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Starting {method_name}")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"time for {method_name}: {elapsed_time:.6f} seconds")
            return result
        return wrapper
    return decorator

class Chat:
    def __init__(self,file_path,chunk_size,chunk_overlap,directory="db/chroma/"):
        self.file_path = file_path
        self.session_id = str(uuid.uuid4())
        self.model = HuggingFaceEndpoint(repo_id=REPO_ID, model_kwargs={'token':HUGGINGFACEHUB_API_TOKEN})
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.database = directory + self.session_id
        self.vectorstore = None
        self.chain = None
        self.retriever = None
        self.docs = None
        self.prompt_text = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise. "
            "Your answers shouldn't be in the format of 'Assistant: Answer'. "
            "Just give the answer directly without any prefix."
            "\n\n"
            "{context}"
        )
            
        self.history_prompt_text = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
        self.history_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.history_prompt_text),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.store = {}
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",self.prompt_text),
            ("human","{input}"),
        ])
    def get_session_history(self,session_id):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    @time_it("splitting")
    def splitting(self):
        loader = PyPDFLoader(self.file_path).load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap  = self.chunk_overlap
        )
        self.docs = text_splitter.split_documents(loader)
    @time_it("indexing")
    def indexing(self):
        
        self.vectorstore = Chroma.from_documents(documents=self.docs,embedding = self.embedding_model,persist_directory = self.database)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity",search_kwargs={'k':3})
    @time_it("retrieval")
    def retrieval(self):
        history_aware_retriever = create_history_aware_retriever(
            self.model, self.retriever, self.history_prompt
        )
        qa_chain = create_stuff_documents_chain(self.model,self.prompt)
        
        self.rachain = create_retrieval_chain(history_aware_retriever,qa_chain)
        self.chain = RunnableWithMessageHistory(
                self.rachain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        
        
        
        
        
        
if __name__ == "__main__":
        
    rag = Chat("documents/alchemist.pdf",500,50)
    rag.splitting()
    rag.indexing()
    rag.retrieval()
    end = True
    while end:
        question = input("ASK AWAY MY DEAR\n")
        response = rag.chain.invoke({"input":question},config={"configurable":{"session_id":rag.session_id}})
        print(response)
        print(response['answer'])
        # print(response['answer'])
        ask = input("do you want to coninue\n")
        if ask == "y":
            end = True
        else:
            end = False
            exit()