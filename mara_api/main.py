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

# --- Security and Authentication Setup ---

INTERNAL_SECRET_KEY = os.getenv("INTERNAL_SECRET_KEY", "YOUR_SUPER_SECRET_PRE_SHARED_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a_different_strong_secret_for_jwt")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week expiration for tokens

class TokenData(BaseModel):
    user_id: int | None = None

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
        status_code=401,
        detail="Could not validate credentials",
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

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)

client = None
gemini_model = "gemini-2.5-pro"
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
    "https://myeducationresearcher.com",
    "https://timothy-han.com",
    "https://jsdean1517-pdkfw.wpcomstaging.com",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schema (Pydantic Models) ---
class Query(BaseModel):
    message: str

class FollowupQuery(BaseModel):
    conversation_id: str
    message: str

class Confidence(enum.Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    
    @staticmethod
    def get_description():
        return (
            "GREEN - If the research on the topic has a well-conducted, randomized study showing a statistically significant positive effect on at least one outcome measure (e.g., state test or national standardized test) analyzed at the proper level of clustering (class/school or student) with a multi-site sample of at least 350 participants. Strong evidence from at least one well-designed and wellimplemented experimental study."
            + "\nYELLOW - If it meets all standards for “green” stated above, except that instead of using a randomized design, qualifying studies are prospective quasi-experiments (i.e., matched studies). Quasiexperimental studies (e.g., Regression Discontinuity Design) are those in which students have not been randomly assigned to treatment or control groups, but researchers are using statistical matching methods that allow them to speak with confidence about the likelihood that an intervention causes an outcome."
            + "\nRED - The topic has a study that would have qualified for “green” or “yellow” but did not because it failed to account for clustering (but did obtain significantly positive outcomes at the student level) or did not meet the sample size requirements. Post-hoc or retrospective studies may also qualify."
        )

class AnalysisDetails(BaseModel):
    regression_models: str
    process: str
    plots: str

class AnalysisResponse(BaseModel):
    summary: str
    confidence: Confidence
    details: AnalysisDetails

# --- API Endpoints ---

@app.post("/auth/issue-wordpress-token", dependencies=[Depends(check_internal_secret)])
async def issue_wordpress_token(user: User):
    print(f"Issuing token for WordPress user: {user.email} (ID: {user.id})")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "name": user.name},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat")
async def chat_api(query: Query, current_user: User = Depends(get_current_user)):
    print(f"Authenticated request from user: {current_user.email}")
    user_query = query.message

    async def event_generator():
        if not client:
            error_data = {"type": "error", "content": "Server error: AI client not initialized."}
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        try:
            yield f"data: {json.dumps({'type': 'update', 'content': 'Finding relevant studies...'})}\n\n"
            step_1_result = await get_studies(user_query)
            yield f"data: {json.dumps({'type': 'step_result', 'step': 1, 'content': step_1_result})}\n\n"

            yield f"data: {json.dumps({'type': 'update', 'content': 'Extracting study data...'})}\n\n"
            step_2_result = await extract_studies_data(step_1_result)

            # --- ADDED LINE TO MEASURE DATA SIZE ---
            print(f"✅ Size of Step 2 Result: {len(step_2_result)} characters")

            yield f"data: {json.dumps({'type': 'step_result', 'step': 2, 'content': step_2_result})}\n\n"

            yield f"data: {json.dumps({'type': 'update', 'content': 'Analyzing study data...'})}\n\n"
            analysis_result = await analyze_studies(step_2_result)
            
            conversation_id = str(uuid.uuid4())
            chat_sessions[conversation_id] = {
                "user_id": current_user.id,
                "original_query": user_query,
                "studies_list": step_1_result,
                "studies_data": step_2_result,
                "analysis": analysis_result.model_dump(),
                "history": [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": analysis_result.summary}
                ]
            }

            result_data = {"type": "result", "content": analysis_result.model_dump()}
            yield f"data: {json.dumps(result_data, cls=CustomEncoder)}\n\n"
            
            yield f"data: {json.dumps({'type': 'conversation_id', 'content': conversation_id})}\n\n"

            suggestions_content = (
                "Analysis complete. Here are some things you can do next:\n"
                "- Show me a forest plot of the results\n"
                "- Show me the procedures used\n"
                "- Show me a list of included citations\n"
                "- Show me the meta-analytic models\n"
                "- Ask another question"
            )
            yield f"data: {json.dumps({'type': 'update', 'content': suggestions_content})}\n\n"
        except Exception as e:
            print(f"An error occurred in the stream: {e}")
            error_data = {"type": "error", "content": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/followup")
async def followup_api(query: FollowupQuery, current_user: User = Depends(get_current_user)):
    conversation_id = query.conversation_id
    session_data = chat_sessions.get(conversation_id)

    if not session_data:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if session_data.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have permission to access this conversation.")

    async def followup_generator():
        try:
            yield f"data: {json.dumps({'type': 'update', 'content': 'Analyzing followup...'})}\n\n"
            
            followup_prompt = compose_followup_query(session_data, query.message)
            
            response_stream = await client.generate_content_async(followup_prompt, stream=True)
            
            full_response = ""
            async for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    yield f"data: {json.dumps({'type': 'message', 'content': chunk.text})}\n\n"
            
            session_data["history"].append({"role": "user", "content": query.message})
            session_data["history"].append({"role": "assistant", "content": full_response})
            chat_sessions[conversation_id] = session_data

        except Exception as e:
            print(f"An error occurred in the followup stream: {e}")
            error_data = {"type": "error", "content": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(followup_generator(), media_type="text/event-stream")

# --- MARA Logic and Prompt Composition ---

def compose_followup_query(session_data: dict, new_message: str) -> str:
    history_summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in session_data.get("history", [])])
    
    return (
        f"{common_persona_prompt}\n"
        f"You are continuing a conversation with a user. Below is the context from their original query and the chat history so far.\n\n"
        f"--- Original Context ---\n"
        f"Original Question: {session_data['original_query']}\n"
        f"Analysis Summary: {session_data['analysis']['summary']}\n"
        f"--- End Original Context ---\n\n"
        f"--- Conversation History ---\n"
        f"{history_summary}\n"
        f"--- End History ---\n\n"
        f"Now, please answer the user's latest message based on all the information provided.\n"
        f"User's new message: \"{new_message}\""
    )

async def get_studies(user_query: str) -> str:
    if not user_query:
        raise ValueError("Step 1: user_query is empty.")
    
    step_1_query = compose_step_one_query(user_query)
    response = await client.generate_content_async(step_1_query)
    
    if not response.text:
        raise ValueError("Step 1: No response from Gemini.")
        
    return response.text

def compose_step_one_query(user_query: str) -> str:
    return (
        common_persona_prompt
        + " Find me high-quality studies that look into the question of: " + user_query
        + "\nOptimize your search per the following constraints: "
        + "\n1. Search online databases that index published literature, as well as sources such as Google Scholar."
        + "\n2. Find studies per retrospective reference harvesting and prospective forward citation searching."
        + "\n3. Attempt to identify unpublished literature such as dissertations and reports from independent research firms."
        + "\nExclude any studies which either:"
        + "\n1. lack a comparison or control group,"
        + "\n2. are purely correlational, that do not include either a randomized-controlled trial, quasi-experimental design, or regression discontinuity"
        + "\nFinally, return these studies in a list of highest quality to lowest, formatting that list by: 'Title, Authors, Date Published.' "
        + "\nInclude 7 high-quality studies, or if fewer than 7, the max available." 
        + "\nDo not add any explanatory text."
    )

async def extract_studies_data(step_1_result: str) -> str:
    if not step_1_result:
        raise ValueError("Step 2: step_1_result is empty.")
    step_2_query = compose_step_two_query(step_1_result)
    response = await client.generate_content_async(step_2_query)
    if not response.text:
        raise ValueError("Step 2: No response from Gemini.")
    return response.text

def compose_step_two_query(step_1_result: str) -> str:
    return (
        common_persona_prompt
        + " For each study in this list, look up the paper:\n" + step_1_result
        + "\nThen, extract the following data into a spreadsheet format: "
        + "\n1. Sample size of treatment and comparison groups"
        + "\n2. Cluster sample sizes (i.e. size of classroom/school)"
        + "\n3. Intraclass correlation coefficient (ICC). If not provided, impute 0.20."
        + "\n4. Effect size for each outcome (standardized mean difference, adjusted for pre-test if possible)."
        + "\n5. Study design (RCT, quasi-experimental, or RDD)."
        + "\nReturn only the spreadsheet data and nothing else."
    )

# --- UPDATED FUNCTION ---
async def analyze_studies(step_2_result: str, max_retries: int = 2) -> AnalysisResponse:
    if not step_2_result:
        raise ValueError("Step 3: step_2_result is empty.")

    original_prompt = compose_step_three_query(step_2_result)
    current_prompt = original_prompt
    
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=AnalysisResponse,
        temperature=0.1
    )
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            print(f"--- Step 3: Analysis, Attempt {attempt + 1}/{max_retries + 1} ---")
            
            # MODIFICATION 1: Add a request timeout to prevent indefinite hanging.
            request_options = {"timeout": 180} # Timeout in seconds (e.g., 3 minutes)
            
            response = await client.generate_content_async(
                current_prompt, 
                generation_config=generation_config,
                request_options=request_options
            )
            
            # MODIFICATION 2: Explicitly check if the response was blocked by safety filters.
            if not response.parts:
                block_reason = "Unknown"
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason.name
                raise ValueError(f"Request failed: The API returned an empty response, possibly due to safety filters. Block Reason: {block_reason}")

            print(f"🔎 Raw Response (Attempt {attempt + 1}): {response.text}")
            
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            parsed_response = AnalysisResponse.model_validate_json(cleaned_text)
            
            print("✅ Successfully parsed valid JSON from Gemini.")
            return parsed_response

        except (ValidationError, json.JSONDecodeError) as e:
            print(f"🔴 Attempt {attempt + 1} failed due to invalid JSON. Error: {e}")
            last_error = e
            # ... (rest of the retry logic is the same)
            
        except Exception as e:
            print(f"🔴 An unexpected error occurred in attempt {attempt + 1}: {e}")
            last_error = e
            if attempt >= max_retries:
                raise ValueError(f"Step 3 failed after {max_retries + 1} attempts. Last error: {last_error}")

    raise ValueError(f"Step 3 failed after {max_retries + 1} attempts. Last error: {last_error}")

def compose_step_three_query(step_2_result: str) -> str:
    # By providing a clear JSON structure example, we guide the model to produce the correct output format reliably.
    json_structure_example = """
{
  "summary": "A one or two sentence summary of the analysis conclusion.",
  "confidence": "GREEN",
  "details": {
    "process": "A description of the meta-analysis process used.",
    "regression_models": "The specific meta-regression models produced, including coefficients and statistics.",
    "plots": "A textual description of relevant plots, such as a forest plot or funnel plot."
  }
}
"""

    return (
        common_persona_prompt
        + "\nUsing this dataset: " + step_2_result
        + "\nPerform a meta-analysis using a multivariate meta-regression model and return the results as a valid JSON object."
        + "\n\n**CRITICAL REQUIREMENT:** Your response MUST be a single, valid JSON object that strictly adheres to the following structure and schema. Do not include any text, markdown formatting, or explanations outside of the JSON object itself."
        + f"\n\nHere is an example of the required JSON structure:\n```json\n{json_structure_example}\n```"
        + "\n\nNow, populate this exact JSON structure based on your analysis:"
        + "\n1. For the `summary` field: Write a one or two sentence summary of your conclusion."
        + "\n2. For the `confidence` field: Determine the confidence level (GREEN, YELLOW, or RED) based on these criteria: " + Confidence.get_description()
        + "\n3. For the nested `details.process` field: Describe the analysis process you used."
        + "\n4. For the nested `details.regression_models` field: Show the regression models produced."
        + "\n5. For the nested `details.plots` field: Describe any corresponding plots."
    )