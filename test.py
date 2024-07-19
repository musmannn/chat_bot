import time
from typing import List, Tuple
import sys
from colorama import Fore, Style, init
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from operator import itemgetter
import os
api_key = os.environ['OPENAI_API_KEY']

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

model_name = "OpenAI"

def format_message(role: str, content: str) -> str:
    return f"{role.capitalize()}: {content}"

def get_chat_history(memory) -> List[Tuple[str, str]]:
    return [(msg.type, msg.content) for msg in memory.chat_memory.messages]

def print_typing_effect(text: str):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.05)
    print()

def run_chatbot(model_name, system_prompt='', history_window=30, temperature=0.3):
    if model_name == "OpenAI":
        model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=temperature)
    else:
        model = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={'token':HUGGINGFACEHUB_API_TOKEN}) 

    if system_prompt:
        SYS_PROMPT = system_prompt
    else:
        SYS_PROMPT = "Act as a helpful AI Assistant"
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', SYS_PROMPT),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{input}')
        ]
    )
    
    memory = ConversationBufferWindowMemory(k=history_window, return_messages=True)
    conversation_chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
        )
        | prompt
        | model
    )
    
    print_typing_effect(f"Hello, I am your friendly chatbot. Let's chat!")
    print("Type 'STOP' to end the conversation, 'HISTORY' to view chat history, or 'CLEAR' to clear the chat history.")
    
    while True:
        user_input = input('User: ')
        if user_input.strip().upper() == 'STOP':
            print_typing_effect('ChatGPT: Goodbye! It was a pleasure chatting with you.')
            break
        elif user_input.strip().upper() == 'HISTORY':
            chat_history = get_chat_history(memory)
            print("\n--- Chat History ---")
            for role, content in chat_history:
                print(format_message(role, content))
            print("--- End of History ---\n")
            continue
        elif user_input.strip().upper() == 'CLEAR':
            memory.clear()
            print_typing_effect('ChatGPT: Chat history has been cleared.')
            continue
        
        user_inp = {'input': user_input}
        start_time = time.time()
        reply = conversation_chain.invoke(user_inp)
        end_time = time.time()
        response_time = end_time - start_time
        print(f"(Response generated in {response_time:.2f} seconds)")
        if model_name == "OpenAI":
            print_typing_effect(f'ChatGPT: {reply.content}')
            memory.save_context(user_inp, {'output': reply.content})
        else:
            print_typing_effect(f'Mistral: {reply}')
            memory.save_context(user_inp, {'output': reply})    

if __name__ == "__main__":
    run_chatbot(model_name="Mistral",
        system_prompt="""You are a friendly chatbot named "Amal". Your expertise is 
            exclusively in providing information and advice about anything related to latest question. 
            This includes anything relevant to latest question. You do not provide information outside of this 
            scope. Make sure to answer as consicely as possible""",
        history_window=50,
        temperature=0.1
    )