from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

load_dotenv()

model =  ChatOpenAI(model="gpt-3.5-turbo")
store ={}
def get_session_history(session_id:str) ->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        return  store[session_id]
prompt = ChatPromptTemplate.from_messages(
    [
        ()
    ]

)


if __name__ == "__main__":
    messages =[ HumanMessage(content="hello my name is cengizhan "),
               AIMessage(content='Hello Cengizhan, nice to meet you! How can I assist you today?')
               ,HumanMessage(content='what is my name'),
               ]
    response =model.invoke(messages)
    print(response.content)

