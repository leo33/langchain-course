import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


tools = [ TavilySearch(TAVLILY_API_KEY=os.getenv("TAVILY_API_KEY")) ]
llm = ChatOpenAI(temperature=0, model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

def main():
    result = chain.invoke(
        input={
            "input": "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )   
    print(result)




if __name__ == "__main__":
    main()
