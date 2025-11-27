from typing import TypedDict, Literal, Annotated, List
from langgraph.graph.message import add_messages

class ResponseSchema(TypedDict):
    user_query: str
    query_response: str
    evaluation_state: Literal["True", "False"]
    retry_count: int
    instruction: str
    messages: Annotated[List, add_messages]
    evaluation_state: Literal["True", "False"]
    retry_count: int
    instruction: str