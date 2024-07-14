
import os 
import streamlit as st
from backend import Chat


def generate_response(prompt,rag):
    response = rag.chain.invoke({"input":prompt},config={"configurable":{"session_id":rag.session_id}})
    print(response)
    return response['answer']
def file_handle(file_path):
    global rag
    uploaded_file = file_path
    print("File changed")
    if not os.path.exists("temp"):
        os.makedirs("temp")
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    print("Creating object")
    
    rag = Chat(file_path, 500, 50)
    rag.splitting()
    rag.indexing()
    rag.retrieval()
    return rag
    
    


if 'last_file' not in st.session_state:
    st.session_state['last_file'] = None

if 'rag' not in st.session_state:
    st.session_state['rag'] = None
st.title("ChatGPT-like clone")

# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
file_src = st.file_uploader("Upload file",['pdf'],False)
if file_src and st.session_state['last_file'] != file_src:
    print("Doing it again")
    print(f"last_file  == {st.session_state['last_file']} and file_src == {file_src}")
    st.session_state['last_file'] = file_src
    st.session_state['rag'] =  file_handle(file_src)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        response = generate_response(prompt,st.session_state['rag'])
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

