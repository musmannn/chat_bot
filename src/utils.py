from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
import uuid
import os
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

"""
Types of retrievers we can use:
    1)Query-Retriever
    2)Similarity Search
    3)BM25
    4)

"""



def create_retriever(source_file,method,embedding="all-MiniLM-L6-v2",persist_dir="db/chroma/"):
    persist_dir += str(uuid.uuid4())
    loader = PyPDFLoader(source_file).load()
    embedding = HuggingFaceEmbeddings(model_name=embedding)
    docs = SemanticChunker(embedding, breakpoint_threshold_type="percentile").split_documents(loader)
    analysis  = []
    bm25_retriever = BM25Retriever.from_documents(documents=docs,k=3)
    # docs = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap).split_documents(loader)
    vectorstore = Chroma.from_documents(documents=docs,embedding=embedding,persist_directory=persist_dir)
    similarity_retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={'k':3})
    # query_retriever = SelfQueryRetriever.from_llm(
    #     llm,
    #     vectorstore,
    #     document_content_description,
    #     metadata_field_info,
    #     verbose=True
    # )
    retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, similarity_retriever], weights=[0.5, 0.5]
    )
    # return docs
    return {'retriever':retriever,'vectordb':vectorstore,'path':persist_dir}

def file_worker(file_path):
    if not os.path.exists("temp"):
        os.makedirs("temp")
    file_src = os.path.join("temp", file_path.name)
    with open(file_src, "wb") as f:
        f.write(file_path.getbuffer())
    return "temp/"+file_path.name

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



if __name__ == "__main__":
    print("Hello WORLD")
    datas =  create_retriever("docs/ml_1.pdf","semantic_chunking")
    analysis  = []
    for data in datas:
        data = {'metadata':[data.metadata],'page_content':[data.page_content]}
        analysis.append(data)
    with open("documents.json", "w") as file:
        json.dump(analysis, file, indent=4)