from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

chat_system_text = (
    "You are a chatbot.Answer all questions to the best of your ability."
    "The person you are responding to is named {username}"
    "You have to complete your response and not stop in midway"
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            chat_system_text
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ]
)

rag_system_prompt_text = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to help you in answering "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise. "
            "The human you are responding to is named {username}"
            "{context}"
        )
rag_history_prompt_text = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
rag_history_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rag_history_prompt_text),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


rag_system_prompt = ChatPromptTemplate.from_messages([
    ("system",rag_system_prompt_text),
    ("human","{input}"),
])
