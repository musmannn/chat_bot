from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from simple_testing import questions
import time
from testing import models,test_inputs,test_prompts
from openai import OpenAI
from langchain_core.messages import AIMessage,HumanMessage
import os
from langchain_core.runnables.history import RunnableWithMessageHistory
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
store = {}
from langchain_openai import ChatOpenAI
import json



print(os.getenv('OPENAI_API_KEY'))

# models = []
# questions = []



# model = ChatOpenAI(
#     model="mistralai/Mistral-7B-Instruct-v0.2",
#     base_url="https://api-inference.huggingface.co/v1",
#     api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN'))
# # model = OpenAI(
# #   base_url="https://api-inference.huggingface.co/v1",
# #   api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN')
# # )
# # response  = model.chat

# # model = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={'token':HUGGINGFACEHUB_API_TOKEN})

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]


# with_message_history = RunnableWithMessageHistory(model, get_session_history)


# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Answer all questions to the best of your ability.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )
# config = {"configurable": {"session_id": "hello123"}}
# chain = prompt | model
# with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# response = with_message_history.invoke(
#     [HumanMessage(content="Hi! I'm Jim")],
#     config=config,
# )
def simple_test():
    results = []
    results_example = []
    for model in models:
        examples_text = []
        for prompt in test_prompts:
            examples_text.append(prompt['Example_text'])
        example_text = '\n'.join(examples_text)
        if model == "openai":
            chatmodel  = ChatOpenAI(model="gpt-3.5-turbo")
            
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"You are a chatbot which specializes in question answering\n ",
                    ),
                    ('human',"{input}")
                ]
            )
            chain  = prompt_template | chatmodel
            
            
            for question in questions:
                data = {'model':model,'user_input':None,'response':None,'time':None,'tokens_used':None}
                print(question)
                data['user_input'] = question
                a  = time.time()
                response =  chain.invoke({'input':data['user_input']})
                b = time.time()
                # print(response)
                data['response'] = response.content
                data['tokens_used'] = response.response_metadata['token_usage']['completion_tokens']
                data['time'] = b-a
                results.append(data)
        else:
            chatmodel  = ChatOpenAI( model="mistralai/Mistral-7B-Instruct-v0.2",
                base_url="https://api-inference.huggingface.co/v1",
                api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN'),max_tokens=6000)
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"You are a chatbot which specializes in question answering\n  ",
                    ),
                    ('human',"{input}")
                ]
            )

            chain  = prompt_template | chatmodel
            
            
            for question in questions:
                data = {'model':model,'user_input':None,'response':None,'time':None,'tokens_used':None}
                print(question)
                data['user_input'] = question
                a  = time.time()
                response =  chain.invoke({'input':data['user_input']})
                b = time.time()
                # print(response)
                data['response'] = response.content
                data['tokens_used'] = response.response_metadata['token_usage']['completion_tokens']
                data['time'] = b-a
                results.append(data)
    for model in models:
        examples_text = []
        for prompt in test_prompts:
            examples_text.append(prompt['Example_text'])
        example_text = '\n'.join(examples_text)
        if model == "openai":
            chatmodel  = ChatOpenAI(model="gpt-3.5-turbo")
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"You are a chatbot which specializes in question answering\n Use the following examples to structure your answer more appropriately\n {example_text} ",
                    ),
                    ('human',"{input}")
                ]
            )
            chain  = prompt_template | chatmodel

            for question in questions:
                data = {'model':model,'user_input':None,'response':None,'time':None,'tokens_used':None}
                print(question)
                data['user_input'] = question
                a  = time.time()
                response =  chain.invoke({'input':data['user_input']})
                b = time.time()
                # print(response)
                data['response'] = response.content
                data['tokens_used'] = response.response_metadata['token_usage']['completion_tokens']
                data['time'] = b-a
                results_example.append(data)
        else:
            chatmodel  = ChatOpenAI( model="mistralai/Mistral-7B-Instruct-v0.2",
        base_url="https://api-inference.huggingface.co/v1",
        api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN'),max_tokens=6000)
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"You are a chatbot which specializes in question answering\n Use the following examples to structure your answer more appropriately\n {example_text} ",
                    ),
                    ('human',"{input}")
                ]
            )
            chain  = prompt_template | chatmodel
            for question in questions:
                data = {'model':model,'user_input':None,'response':None,'time':None,'tokens_used':None}
                print(question)
                data['user_input'] = question
                a  = time.time()
                response =  chain.invoke({'input':data['user_input']})
                b = time.time()
                # print(response)
                data['response'] = response.content
                data['tokens_used'] = response.response_metadata['token_usage']['completion_tokens']
                data['time'] = b-a
                results_example.append(data)
    return results,results_example




file_name = "data_simple.json"
file_name_example = "data_example.json"
results,results_example = simple_test()

with open(file_name, "w") as json_file:
    json.dump(results, json_file, indent=4)
with open(file_name_example, "w") as json_file:
    json.dump(results_example, json_file, indent=4)