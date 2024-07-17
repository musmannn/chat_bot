import gradio as gr
from backend import Chat

"""
Todo:
    Add the streaming functionality
    Add the status thing like it should know what is happening
    
    

"""


class App:
    def __init__(self,):
        self.current_file = None
        self.modal = None
        self.cur_file = None
        self.ui = gr.ChatInterface(
            self.generate_data,
            multimodal=True,
            title="InsightBot",
            textbox=gr.MultimodalTextbox(file_types=["pdf"],file_count="single",interactive=True),
            description="Get Insights on your document using our  bot",
            theme="soft",
            retry_btn=None,
            undo_btn="Delete Previous",
            clear_btn="Clear",
            
        )
    def generate_data(self,input_data,history):
        print(f"Current File:{self.current_file}\n\n cur_file:{self.cur_file}")
        
        print(input_data)
        
        print(history)
        if input_data.files:
            text = input_data.text
            source_file = input_data.files[0]
            self.cur_file = source_file.url
            if self.current_file != self.cur_file:
                self.current_file = self.cur_file
                self.modal = Chat(self.cur_file,500,50)
                self.modal.splitting()
                self.modal.indexing()
                self.modal.retrieval()
                response = self.modal.chain.invoke({"input":text},config={"configurable":{"session_id":self.modal.session_id}})['answer']
                answer = ""
                for char in response:
                    answer += char
                    yield answer
                    
            else:
                print("Same file entered")
                response = self.modal.chain.invoke({"input":text},config={"configurable":{"session_id":self.modal.session_id}})['answer']
                answer = ""
                for char in response:
                    answer += char
                    yield answer
        elif self.modal:
            print("No file entered. Using previous file")
            text = input_data.text
            response = self.modal.chain.invoke({"input":text},config={"configurable":{"session_id":self.modal.session_id}})['answer']
            answer = ""
            for char in response:
                answer += char
                yield answer
        else:
            response =  "You haven't Uploaded any source"
            answer = ""
            for char in response:
                answer += char
                yield answer




if __name__ == "__main__":
    gui  = App()
    gui.ui.launch()
    

