from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return {
        "content": f"It's always sunny in {city}!",
    }

def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]

agent = create_react_agent(
    model="groq:llama-3.3-70b-versatile",  # was: "anthropic:claude-3-7-sonnet-latest"
    tools=[get_weather],
    prompt=prompt
)
# Run the agent
agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]},
            config={"configurable": {"user_name": "John Smith", "max_iterations": 2}})