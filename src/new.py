import streamlit as st
import time
from utils import file_worker
from backend import Chat

st.set_page_config(
    page_title="PDF AI"
)
todos = '''
    Todos:
        -- Add the reset button 
        --

'''





st.title("PDF Assistant")

def stream(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def reset_state():
    st.session_state['call_rerun'] = True
    st.session_state['fileuploaded'] = False
    st.session_state['filemodel'] = None
    st.session_state['messages'] = [{'role':"AI","content":"Hello! How can i help you"}]
    st.session_state['bot'] = Chat()
    st.session_state['username'] = ""
    st.session_state['file_uploaded'] = False
    st.session_state['file_uploader_key'] += 1
    
username = None
def main():
    
    global username
    if "call_rerun" not in st.session_state:
        st.session_state['call_rerun'] = False
    if "bot" not in st.session_state:
        st.session_state['bot'] = Chat()
    if 'fileuploaded' not in st.session_state:
        st.session_state['fileuploaded'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = ""
    if 'filemodel' not in st.session_state:
        st.session_state['filemodel'] = None
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role':"AI","content":"Hello! How can i help you"}]
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
    with st.sidebar:
        placeholder = st.empty()
        st.session_state['username'] = placeholder.text_input("Enter your name")
        if st.session_state['username']:

            placeholder.empty()
            uploaded_file = st.file_uploader("Upload Source",["pdf"],False,key = st.session_state["file_uploader_key"])
            if uploaded_file:
                uploaded_file = file_worker(uploaded_file)
                print(f"uploaded_file is {uploaded_file}")
                print(f"filemodel is {st.session_state['filemodel']}")
                if uploaded_file != st.session_state['filemodel']:
                    st.session_state['filemodel'] = uploaded_file
                    st.session_state['file_uploaded'] = True
                    with st.spinner("Processing Documents"):
                        print("Creating the rag doc.uments")
                        st.session_state['bot'].add_source(uploaded_file)
                        print(f"STATE AFTER UPLOADING {st.session_state['bot'].file_uploaded}")
                        print(st.session_state['bot'])
                    placeholder_success = st.success("Done")
                    time.sleep(2)
                    placeholder_success.empty()
                placeholder_reset = st.button("Reset Chat",on_click=reset_state)
                if st.session_state['call_rerun'] == True:
                    st.session_state['call_rerun'] = False
                    st.rerun()

                else:
                    pass


    if st.session_state['username']:
        for message in st.session_state['messages']:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # print(f"File Uploaded in session state:{st.session_state['file_uploaded']}")
                print(f"STATE AFTER UPLOADING {st.session_state['bot'].file_uploaded}")
                print(st.session_state['bot'])
                response = st.session_state['bot'].run(prompt,st.session_state['username'])
                print(response)
                text = response
                st.write_stream(stream(text))
                st.session_state.messages.append({"role": "assistant", "content": text})
    else:
        print("No username :(")


if __name__ == "__main__":
    main()