from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.Workflow.workflow import workflow
from settings import MEMORY_LIMIT
import re

app = FastAPI()

class ChatRequest(BaseModel):
    user_message: str

class ChatResponse(BaseModel):
    response: str

class ChatRequest(BaseModel):
    user_message: str
    thread_id: str = "default_thread"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],  # important for POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # allow headers like Content-Type, Authorization
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Chatbot API running"}


@app.post("/chatbot", response_model=ChatResponse)
async def chatbot_endpoint(request: ChatRequest):
    
    try:
        user_input = request.user_message
        print(f"User message: {user_input}")

        from langchain_core.messages import HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "query_response": "",
            "evaluation_state": "",
            "retry_count": 0,
            "instruction": ""
        }

        thread_id = request.thread_id

        config = {
            "configurable": {"thread_id": thread_id},
            "verbose": True,
            "memory_limit": MEMORY_LIMIT
        }

        final_state = workflow.invoke(initial_state, config= config)
        
        print("Workflow final state:", final_state)

        query_response = final_state.get("query_response", "No response generated.")

        if "ValidationOutcome" in query_response:
           
            pattern = r'validated_output="((?:[^"\\]|\\.)*)"'
            match = re.search(pattern, query_response)
            if match:
                answer = match.group(1).replace('\\n', '\n').replace('\\"', '"').strip()
            else:
                
                fallback_pattern = r'validated_output=\'([^\']*)\'[, ]'
                fallback_match = re.search(fallback_pattern, query_response)
                if fallback_match:
                    answer = fallback_match.group(1).replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'").strip()
                else:
                    
                    start_idx = query_response.find("validated_output='") + len("validated_output='")
                    end_idx = query_response.find("',\n    reask=", start_idx)
                    if end_idx != -1:
                        raw_content = query_response[start_idx:end_idx]
                        answer = raw_content.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'").strip()
                    else:
                        answer = "Error parsing validated output."
        else:
            
            answer = query_response.replace("'", "").strip().strip("'").strip()

        
        answer = re.sub(r'\n+$', '', answer).strip()

        
        return ChatResponse(response=answer)

    except Exception as e:
        print("Error in /chatbot:", e)
        raise HTTPException(status_code=500, detail=str(e))
# python -m uvicorn src.main.chatbotapi:app --reload