import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from insight_engine import summarize_metric, get_basic_info

load_dotenv()

# --- Define Tools (pandas helpers exposed to the agent) ---
@tool
def list_available_columns() -> dict:
    """List all available metrics and columns from the dataset."""
    return get_basic_info()

@tool
def analyze_metric(metric: str) -> dict:
    """Summarize week-to-week trend for a metric."""
    return summarize_metric(metric)

tools = [list_available_columns, analyze_metric]

# --- Initialize Azure OpenAI ---
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://psacodesprint2025.azure-api.net"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1-nano"),
    api_version=os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
)

# --- Define the reasoning prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are PSA’s data analytics assistant. "
               "Use the available tools to analyze port performance data "
               "and explain insights clearly in business terms."),
    MessagesPlaceholder(variable_name="messages"),
])

# --- Create agent (LangGraph runtime) ---
agent = create_react_agent(llm, tools=tools, prompt=prompt)

# In LangChain 1.0, the agent itself is Runnable.
# You can invoke it directly instead of wrapping in AgentExecutor.
def run_agentic_query(query: str):
    """Takes a user question and returns AI-generated insight."""
    try:
        result = agent.invoke({"messages": [("user", query)]})
        return result.get("output", "No response from agent.")
    except Exception as e:
        print("❌ Agent error:", e)
        return f"Error: {e}"
