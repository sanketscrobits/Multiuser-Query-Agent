from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.Workflow.workflow import workflow
from settings import MEMORY_LIMIT, USERS_CONFIG
from src.utils.vector_db.loader_strategies.local_loader import LocalLoader
from src.utils.vector_db.index_strategies.pinecone_vector_index import PineconeVectorIndex
from src.utils.vector_db.vector_store_singleton import VectorStoreSingleton
from langchain_huggingface import HuggingFaceEmbeddings
import re

app = FastAPI()

class ChatResponse(BaseModel):
    response: str

class ChatRequest(BaseModel):
    user_message: str
    thread_id: str = "default_thread"
    user_id: str = "user1" # Default to user1 for testing

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
        user_id = request.user_id
        
        # Validate user and get namespace
        if user_id not in USERS_CONFIG:
            raise HTTPException(status_code=400, detail=f"Invalid user_id. Available users: {list(USERS_CONFIG.keys())}")
            
        # Unify identity: 
        # 1. Namespace comes from config (based on user_id)
        # 2. Thread ID is derived from user_id (or passed explicitly, but defaults to user-specific)
        
        namespace = USERS_CONFIG[user_id]["namespace"]
        
        # If thread_id is default, bind it to the user_id to ensure isolation
        if request.thread_id == "default_thread":
            thread_id = f"{user_id}_thread"
        else:
            thread_id = request.thread_id

        print(f"User message: {user_input} | User: {user_id} | Namespace: {namespace} | Thread: {thread_id}")

        from langchain_core.messages import HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "query_response": "",
            "evaluation_state": "",
            "retry_count": 0,
            "instruction": ""
        }

        config = {
            "configurable": {
                "thread_id": thread_id,
                "namespace": namespace
            },
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

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    try:
        if user_id not in USERS_CONFIG:
            raise HTTPException(status_code=400, detail=f"Invalid user_id. Available users: {list(USERS_CONFIG.keys())}")
            
        namespace = USERS_CONFIG[user_id]["namespace"]
        
        # Read file content
        content = await file.read()
        text_content = content.decode("utf-8") # Assuming text/markdown file for now
        
        # Initialize VectorStoreSingleton (if not already)
        embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        document_loader_strategy = LocalLoader()
        vector_index_strategy = PineconeVectorIndex(embeddings=embeddings_model)
        
        vector_store = VectorStoreSingleton(
            embeddings_model=embeddings_model,
            document_loader_strategy=document_loader_strategy,
            vector_index_strategy=vector_index_strategy,
        )
        
        # Ingest document
        vector_store.ingest_document(text_content, namespace=namespace)
        
        return {"status": "ok", "message": f"Document uploaded and ingested for user {user_id} in namespace {namespace}"}
        
    except Exception as e:
        print("Error in /upload:", e)
        raise HTTPException(status_code=500, detail=str(e))
# python -m uvicorn src.main.chatbotapi:app --reload