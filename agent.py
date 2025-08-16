from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict, Literal
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
load_dotenv()

# class SupportContext(TypedDict):
#     customer_tier: Literal["Free", "Pro", "Enterprise"]

stub_docs = {
    "tracing": "Tracing helps debug chains and agents by recording structured run data. Tracing is available for Pro and Enterprise tiers.",
    "self-hosting": "LangSmith self-hosting is available to Enterprise customers only. Please contact sales for deployment support.",
    "langgraph rag": "LangGraph can be used to build RAG agents with custom memory, state, and conditional branching.",
    "langsmith eval": "LangSmith supports LLM-as-a-judge, scoring functions, and dataset evaluation for agents and chains."
}

def init_state_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("INIT STATE DEBUG:", state)
    state = dict(state)
    tier = config["configurable"].get("customer_tier", "Free")
    return {
        **state,
        "question": state["messages"][-1].content,
         "customer_tier": tier
    }

def check_access(state: AgentState, config: RunnableConfig) -> AgentState:
    tier = config["configurable"].get("customer_tier", "Free")
    print("DEBUG: customer_tier =", tier)
    can_access = tier in ("Pro", "Enterprise")
    return {**state, "can_access": can_access}

def retrieve_doc(state: AgentState) -> AgentState:
    topic = state.get("topic")
    can_access = state.get("can_access", False)

    if not can_access or topic not in stub_docs:
        return {**state, "retrieved_doc": None}

    return {**state, "retrieved_doc": stub_docs[topic]}

def classify_topic(state: AgentState) -> AgentState:
    question = state.get("question", "").lower()

    if "trace" in question:
        topic = "tracing"
    elif "self-host" in question:
        topic = "self-hosting"
    elif "rag" in question:
        topic = "langgraph rag"
    elif "eval" in question or "evaluate" in question:
        topic = "langsmith eval"
    else:
        topic = "unknown"

    return {**state, "topic": topic}

def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
    user_name = config["configurable"].get("user_name")
    tier = config["configurable"].get("customer_tier", state.get("customer_tier", "Free")),
    topic = state.get("topic")
    can_access = state.get("can_access")
    stub_answer = state.get("retrieved_doc")

    system_msg = (
        f"You are a helpful support assistant for LangChain.\n"
        f"The user is named {user_name} and is on the {tier} tier.\n"
        f"If a feature is not available for their tier, politely explain that and suggest they contact support to upgrade.\n"
        f"Otherwise, answer clearly and concisely."
    )
    if topic and stub_answer:
        system_msg += f"\nThe user is asking about '{topic}'. Here's documentation to help: {stub_answer}"
    return [{"role": "system", "content": system_msg}] + state["messages"]

agent = create_react_agent(
    model="groq:llama-3.3-70b-versatile",  # was: "anthropic:claude-3-7-sonnet-latest"
    tools=[],
    prompt=prompt
)


# Create a new graph builder
builder = StateGraph(AgentState)
# Add a node to inject tier and extract question
builder.add_node("init", RunnableLambda(init_state_node))
# Add nodes
builder.add_node("agent", agent)
builder.add_node("retrieve_doc", retrieve_doc)
builder.add_node("check_access", RunnableLambda(check_access))
# Classify the topic
builder.add_node("classify_topic", classify_topic)
# Wire the nodes together
builder.set_entry_point("init")
builder.add_edge("init", "classify_topic")
builder.add_edge("classify_topic", "check_access")
builder.add_edge("check_access", "retrieve_doc")
builder.add_edge("retrieve_doc", "agent")
builder.add_edge("agent", END)

# Compile the runnable graph
graph = builder.compile()


# Run the agent
input_state = {
    "messages": [{"role": "user", "content": "How do I enable tracing in LangChain?"}],
    "customer_tier": "Pro"
}
result = graph.invoke(
     {
        "messages": [{"role": "user", "content": "How do I enable tracing in LangChain?"}]
    },
    config={"configurable": {"user_name": "John Smith", "customer_tier": "Pro"}}
)
print(result)