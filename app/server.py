from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any
import uuid

app = FastAPI()

# In-memory storage for users and chats
users_db = {}
chats_db = {}

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Template setup
templates = Jinja2Templates(directory="templates")

class User(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    message: str

@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    user = users_db.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    # Generate a token (for simplicity, using username as token)
    token = username
    return {"access_token": token, "token_type": "bearer"}

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    if username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    users_db[username] = {"username": username, "password": password}
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="message", value="Registration successful!", max_age=10)
    return response

@app.post("/chat")
async def create_chat(token: str = Depends(oauth2_scheme)):
    chat_id = str(uuid.uuid4())
    chats_db[chat_id] = {"user": token, "messages": []}
    return RedirectResponse(url=f"/chat/{chat_id}/messages", status_code=303)

@app.post("/chat/{chat_id}/message")
async def send_message(chat_id: str, chat_message: ChatMessage, token: str = Depends(oauth2_scheme)):
    chat = chats_db.get(chat_id)
    if not chat or chat["user"] != token:
        raise HTTPException(status_code=404, detail="Chat not found or access denied")
    
    # Process the message with DynamicAgent
    agent = app.state.agent
    result = await agent.process_task(chat_message.message)
    
    # Store the message and response
    chat["messages"].append({"user": chat_message.message, "agent": result})
    return {"response": result}

@app.get("/chat/{chat_id}/messages", response_class=HTMLResponse)
async def get_messages(request: Request, chat_id: str, token: str = Depends(oauth2_scheme)):
    chat = chats_db.get(chat_id)
    if not chat or chat["user"] != token:
        raise HTTPException(status_code=404, detail="Chat not found or access denied")
    return templates.TemplateResponse("chat.html", {"request": request, "chat_id": chat_id, "messages": chat["messages"]})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, token: str = Depends(oauth2_scheme)):
    user_chats = [chat_id for chat_id, chat in chats_db.items() if chat["user"] == token]
    return templates.TemplateResponse("dashboard.html", {"request": request, "chat_ids": user_chats})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)