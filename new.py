import streamlit as st
import time
st.set_page_config(
    page_title="PDF AI"
)
st.title("PDF Assistant")


def load_llm():
    pass
def load_rag():
    pass


def main():
    if 'fileuploaded' not in st.session_state:
        st.session_state['fileuploaded'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = ""
    if 'filemodel' not in st.session_state:
        st.session_state['filemodel'] = None
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role':"AI","content":"Hello! How can i help you"}]
    with st.sidebar:
        placeholder = st.empty()
        username = placeholder.text_input("Enter your name")
        if username:
            st.session_state['username'] = username
            placeholder.empty()
            uploaded_file = st.file_uploader("Upload Source",["pdf"],False)
            if uploaded_file:
                st.session_state['file_uploaded'] = True
                with st.spinner("Processing Documents"):
                    print("Creating the rag documents")
                    time.sleep(2)
                placeholder_success = st.success("Done")
                time.sleep(2)
                placeholder_success.empty()
            else:
                pass
    
    # if prompt := st.chat_input("What is up?"):
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     with st.chat_message("assistant"):
    #         if st.session_state['fileuploaded'] == False:
    #             pass
    #         else:
    #             # stream = client.chat.completions.create(
    #             #     model=st.session_state["openai_model"],
    #             #     messages=[
    #             #         {"role": m["role"], "content": m["content"]}
    #             #         for m in st.session_state.messages
    #             #     ],
    #             #     stream=True,
    #             # )
    #             response = st.write_stream(stream)
    #             st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == "__main__":
    main()