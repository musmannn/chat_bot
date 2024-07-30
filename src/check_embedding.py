from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text1 = "How"
text2  = "Who is Ashutosh Saxena?"
embedding1 = embedding.embed_query(text1)
embedding2 = embedding.embed_query(text2)

print(np.dot(embedding1,embedding2))

