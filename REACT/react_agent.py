from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

# Ortam değişkenlerini yükleme
load_dotenv()

# Bellek için SQLite veritabanı oluşturma
memory = SqliteSaver.from_conn_string(":memory:")

# Araçları tanımlama
search = TavilySearchResults(max_results=1)
tools = [search]

# Prompt'u hub'dan çekme
prompt = hub.pull("hwchase17/react-chat")

# OpenAI modelini seçme
llm = OpenAI()

# ReAct ajanını oluşturma
agent = create_react_agent(llm, tools, prompt)

# Ajan yürütücüsünü oluşturma ve belleği (checkpoint) ekleme
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, checkpoint=memory)
config = {"configurable": {"thread_id": "abc123"}}

if __name__ == '__main__':
    chat_history = []

    while True:
        # Kullanıcıdan giriş alma
        user_input = input("> ")
        chat_history.append(f"Human: {user_input}")

        response = []
        try:
            for chunk in agent_executor.stream(
                    {
                        "input": user_input,
                        "chat_history": "\n".join(chat_history),
                    },
                    config
            ):
                if 'text' in chunk:
                    print(chunk['text'], end='')
                    response.append(chunk['text'])

            chat_history.append(f"AI: {''.join(response)}")
            print("\n----")
        except Exception as e:
            print(f"Error: {e}")
