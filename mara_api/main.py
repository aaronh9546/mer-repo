from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import enum
import json
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
import re
from typing import Optional

# STEP 1: VERIFICATION PRINT STATEMENT
print("--- RUNNING LATEST VERSION WITH ENUM FIX: SEPTEMBER 21 ---")

# --- Security and Authentication Setup ---

INTERNAL_SECRET_KEY = os.getenv("INTERNAL_SECRET_KEY", "YOUR_SUPER_SECRET_PRE_SHARED_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a_different_strong_secret_for_jwt")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week expiration for tokens

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
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=401, detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = int(payload.get("sub"))
        email: str = payload.get("email")
        name: str = payload.get("name")
        if user_id is None or email is None:
            raise credentials_exception
    except (JWTError, ValueError, TypeError):
        raise credentials_exception
    
    user = User(id=user_id, email=email, name=name)
    return user

async def check_internal_secret(internal_secret: str = Security(api_key_header)):
    if internal_secret != INTERNAL_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid secret key for internal communication")

# --- Application Setup ---
chat_sessions = {}
client = None
gemini_model = "gemini-1.5-pro"
common_persona_prompt = "You are a senior data analyst with a specialty in meta-analysis."
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global client
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("FATAL: GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    client = genai.GenerativeModel(gemini_model)
    print("✅ GenAI Client configured and initialized successfully.")

origins = [
    "https://myeducationresearcher.com", "https://timothy-han.com",
    "https://jsdean1517-pdkfw.wpcomstaging.com", "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Pydantic Models ---
class InitialQuery(BaseModel):
    message: str

class FindStudiesResponse(BaseModel):
    session_id: str
    studies: list[str]

class ExtractRequest(BaseModel):
    session_id: str
    study_title: str

class ExtractedData(BaseModel):
    study: str
    treatment_n: int
    comparison_n: int
    effect_size: float
    design: str

class AnalyzeRequest(BaseModel):
    session_id: str
    collected_data: list[ExtractedData]

class AnalysisDetails(BaseModel):
    regression_models: str
    process: str
    plots: Optional[str] = None

class AnalysisResponse(BaseModel):
    summary: str
    # STEP 2: SIMPLIFIED THE SCHEMA. Replaced the Enum with a simple string.
    confidence: str 
    details: AnalysisDetails

# --- Batching API Endpoints ---
@app.post("/find-studies", response_model=FindStudiesResponse)
async def find_studies_api(query: InitialQuery, current_user: User = Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    print(f"Starting new session {session_id} for user {current_user.email}")
    raw_studies_text = await _get_studies_list(query.message)
    studies_list = [re.sub(r'^\d+\.\s*', '', line).strip() for line in raw_studies_text.strip().split('\n') if line.strip()]
    chat_sessions[session_id] = {
        "user_id": current_user.id, "original_query": query.message,
        "studies_list": studies_list, "status": "pending"
    }
    return FindStudiesResponse(session_id=session_id, studies=studies_list)

@app.post("/extract-single-study", response_model=ExtractedData)
async def extract_single_study_api(request: ExtractRequest, current_user: User = Depends(get_current_user)):
    session = chat_sessions.get(request.session_id)
    if not session or session["user_id"] != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found or access denied.")
    print(f"Extracting data for '{request.study_title[:50]}...' in session {request.session_id}")
    return await _extract_single_study_data(request.study_title)
    
@app.post("/analyze-data", response_model=AnalysisResponse)
async def analyze_data_api(request: AnalyzeRequest, current_user: User = Depends(get_current_user)):
    session = chat_sessions.get(request.session_id)
    if not session or session["user_id"] != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found or access denied.")
    print(f"Analyzing collected data for session {request.session_id}")
    markdown_table = "| Study | Treatment N | Comparison N | Effect Size | Design |\n|---|---|---|---|---|\n"
    for item in request.collected_data:
        markdown_table += f"| {item.study} | {item.treatment_n} | {item.comparison_n} | {item.effect_size} | {item.design} |\n"
    analysis_result = await _analyze_studies_from_data(markdown_table)
    session['status'] = 'complete'
    session['analysis'] = analysis_result.model_dump()
    return analysis_result

# --- Authentication Endpoint ---
@app.post("/auth/issue-wordpress-token")
async def issue_wordpress_token(user: User, internal_secret: str = Security(api_key_header)):
    if internal_secret != INTERNAL_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid secret key for internal communication")
    print(f"Issuing token for WordPress user: {user.email} (ID: {user.id})")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "name": user.name},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- Core Logic and Prompt Composition ---
def _compose_find_studies_query(user_query: str) -> str:
    return (f"{common_persona_prompt}\nFind me up to 30 high-quality studies about: {user_query}\n"
            "Return the results as a simple numbered list with 'Title, Authors, Date Published.'\n"
            "Do not include any other text.")

async def _get_studies_list(user_query: str) -> str:
    prompt = _compose_find_studies_query(user_query)
    response = await client.generate_content_async(prompt)
    return response.text

def _compose_extract_one_query(study_title: str) -> str:
    return (f"{common_persona_prompt}\nFor the study titled '{study_title}', generate realistic data based on the title.\n"
            "You MUST return a single, raw JSON object with the following keys:\n"
            "- `study`: A short name for the study (e.g., 'Author et al. (Year)').\n"
            "- `treatment_n`: An integer for the treatment group size.\n"
            "- `comparison_n`: An integer for the comparison group size.\n"
            "- `effect_size`: A float for the standardized mean difference.\n"
            "- `design`: A string, either 'Randomized Controlled Trial' or 'Quasi-Experimental'.\n"
            "CRITICAL: Your entire response must be ONLY the JSON object.")

async def _extract_single_study_data(study_title: str) -> ExtractedData:
    prompt = _compose_extract_one_query(study_title)
    generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
    try:
        response = await client.generate_content_async(prompt, generation_config=generation_config)
        return ExtractedData.model_validate_json(response.text)
    except Exception as e:
        print(f"🔴 Failed to extract valid JSON for '{study_title}'. Error: {e}. Returning placeholder.")
        return ExtractedData(study=f"{study_title[:30]}... (Error)", treatment_n=0, comparison_n=0, effect_size=0.0, design="Extraction Failed")

def _compose_raw_analysis_query(data_table: str) -> str:
    confidence_description = "GREEN for strong experimental evidence, YELLOW for strong quasi-experimental evidence, RED for correlational or weaker evidence."
    return (f"{common_persona_prompt}\nUsing this dataset:\n{data_table}\n"
            "Perform a meta-analysis. In your response, you must include sections covering these topics:\n"
            f"1. A 'Summary' of the final conclusion.\n"
            f"2. A 'Confidence' level (a single word: GREEN, YELLOW, or RED) based on: {confidence_description}.\n"
            f"3. A 'Process' section describing the analysis process.\n"
            f"4. A 'Regression Models' section showing the models produced.\n"
            f"5. A 'Plots' section describing any corresponding plots.\n"
            "Write the response in clear, simple text. Do not use JSON.")

def _compose_json_formatting_query(raw_analysis_text: str) -> str:
    return (f"Take the following text-based analysis and convert it into a valid JSON object. "
            f"The JSON object must strictly adhere to the required schema. The root object must have keys 'summary', 'confidence', and 'details'. "
            f"The 'confidence' value must be a single string: 'GREEN', 'YELLOW', or 'RED'. " # Updated instruction for the formatter
            f"The 'details' object must contain the keys 'process', 'regression_models', and 'plots'.\n\n"
            f"--- TEXT TO CONVERT ---\n{raw_analysis_text}\n\n"
            f"--- END TEXT --- \n\n"
            f"Now, provide only the JSON object.")

async def _analyze_studies_from_data(data_table: str) -> AnalysisResponse:
    try:
        print("--- Analysis Step 1: Generating raw text analysis ---")
        raw_analysis_prompt = _compose_raw_analysis_query(data_table)
        raw_response = await client.generate_content_async(raw_analysis_prompt)
        print("--- Analysis Step 1 Complete ---")

        print("--- Analysis Step 2: Formatting text into JSON ---")
        json_format_prompt = _compose_json_formatting_query(raw_response.text)
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=AnalysisResponse,
        )
        json_response = await client.generate_content_async(json_format_prompt, generation_config=generation_config)
        print("--- Analysis Step 2 Complete ---")
        
        return AnalysisResponse.model_validate_json(json_response.text)
    except Exception as e:
        print(f"🔴 A critical error occurred during the two-step analysis process: {e}")
        raise HTTPException(status_code=500, detail=f"Failed during the analysis and formatting steps. Last error: {e}")