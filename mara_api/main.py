from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from celery.result import AsyncResult
from tasks import run_full_analysis
from celery_worker import celery_app

# --- Security and Authentication Setup ---
INTERNAL_SECRET_KEY = os.getenv("INTERNAL_SECRET_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

class User(BaseModel):
    id: int
    email: str
    name: str | None = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
api_key_header = APIKeyHeader(name="X-Internal-Secret", auto_error=True)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = int(payload.get("sub"))
        email: str = payload.get("email")
        name: str = payload.get("name")
        if user_id is None or email is None: raise credentials_exception
    except (JWTError, ValueError, TypeError):
        raise credentials_exception
    return User(id=user_id, email=email, name=name)

async def check_internal_secret(internal_secret: str = Security(api_key_header)):
    if internal_secret != INTERNAL_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid secret key")

# --- Application Setup ---
app = FastAPI()
origins = [ "https://aaronhanto-nyozw.com", "https://timothy-han.com", "https://aaronhanto-nyozw.wpcomstaging.com", "http://localhost:3000" ]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models for API ---
class Query(BaseModel):
    message: str

# --- API Endpoints ---
@app.post("/auth/issue-wordpress-token", dependencies=[Depends(check_internal_secret)])
async def issue_wordpress_token(user: User):
    expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(data={"sub": str(user.id), "email": user.email, "name": user.name}, expires_delta=expires)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/chat")
async def start_chat_task(query: Query, current_user: User = Depends(get_current_user)):
    print(f"WEB SERVER: Received chat request from {current_user.email}. Dispatching to worker.")
    task = run_full_analysis.delay(query.message)
    return {"task_id": task.id}

@app.get("/tasks/status/{task_id}")
async def get_task_status(task_id: str, current_user: User = Depends(get_current_user)):
    task_result = AsyncResult(task_id, app=celery_app)
    result = None
    info = None
    
    if task_result.state == 'PROGRESS':
        info = task_result.info
    elif task_result.successful():
        result = task_result.get()
    elif task_result.failed():
        result = {"error": "Task failed. Check server logs for details."}
        
    return {
        "task_id": task_id,
        "status": task_result.state,
        "info": info, # This will contain the progress message
        "result": result
    }