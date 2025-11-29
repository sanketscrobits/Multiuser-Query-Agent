from src.schemas.response_schema import ResponseSchema
from settings import GOOGLE_API_KEY
from src.agents.query_agent import create_query_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Build the query agent once; uses the get_context tool under the hood
query_agent = create_query_agent(api_key=GOOGLE_API_KEY)

def retriver_agent(state: ResponseSchema, config: RunnableConfig) -> ResponseSchema:
    print(f"DEBUG: retriver_agent config: {config}")
    last_message = state["messages"][-1]
    user_query = last_message.content
    instruction = state["instruction"]
    modified_input = {"input": f"{user_query}\n\n{instruction}" if instruction else user_query}
    # Pass all messages except the last one (which is the current user query) as history
    chat_history = state["messages"][:-1]
    
    result = query_agent.invoke({
        "input": modified_input,
        "chat_history": chat_history
    }, config=config)
    response_str = result["output"]

    return {
        "messages": [AIMessage(content=response_str)],
        "evaluation_state": "",
        "retry_count": state["retry_count"] + 1,
        "instruction": instruction
    }
