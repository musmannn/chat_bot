from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
import os
from langchain_openai import ChatOpenAI
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
question = "Hello "
import time
template = """
System: Answer the question below using only 10 words

Question: {question}

"""
template1 = ChatPromptTemplate.from_messages([
     (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.Only use 10 words or less for your response",
        ),
    ("human", "{user_input}"),
])

prompt_value = template1.invoke(
    {
        "name": "Bob",
        "user_input": "What is your name?",
        "username": "Usman"
    }
)
# print(prompt_value)
prompt = PromptTemplate.from_template(template)
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"





llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
chat_model  = ChatHuggingFace(
    llm=llm
)
# llm = 
# llm = ChatOpenAI(
#     model="mistralai/Mistral-7B-Instruct-v0.2",
# base_url="https://api-inference.huggingface.co/v1",
# api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN')
    # model="gpt-4o",
    # temperature=0,
    # max_tokens=None,
    # timeout=None,
    # max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
# )
llm_chain = template1 | chat_model
# # llm_chain = LLMChain(llm=llm, prompt=template1)
# # print(llm_chain.invoke(
    # {"question": question}))
user_inp = input("User:")
while user_inp != "n":
    start_time = time.time()
    asd = llm_chain.invoke({"name": "Bob",
        "user_input": user_inp,
        "username": "Usman"})
    end_time = time.time()
    response_time = end_time - start_time
    print(asd.content)
    print(f"(Response generated in {response_time:.2f} seconds)")
    user_inp = input("User:")
# asd = llm.invoke("Hello")
# print(asd)
