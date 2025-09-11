from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)

client = None
gemini_model = "gemini-1.5-pro-latest" 
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

origins = [ "https://aaronhanto-nyozw.com", "https://timothy-han.com", "https://aaronhanto-nyozw.wpcomstaging.com", "http://localhost:3000" ]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Schema (Pydantic Models) ---
class Query(BaseModel):
    message: str

class Confidence(enum.Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    @staticmethod
    def get_description():
        return (
            "GREEN - If the research on the topic has a well-conducted, randomized study showing a statistically significant positive effect on at least one outcome measure (e.g., state test or national standardized test) analyzed at the proper level of clustering (class/school or student) with a multi-site sample of at least 350 participants. Strong evidence from at least one well-designed and wellimplemented experimental study. Experimental studies were used to answer this question. Experimental studies are those in which students are randomly assigned to treatment or control groups, allowing researchers to speak with confidence about the likelihood that an intervention causes an outcome. Well-designed and well implemented experimental studies. The research studies use large (larger than 350 participants), multi-site samples. No other experimental or quasiexperimental research shows that the intervention negatively affects the outcome. Researchers have found that the intervention improves outcomes for the specific student subgroups that the district or school intends to support with the intervention."
            + "\nYELLOW - If it meets all standards for “green” stated above, except that instead of using a randomized design, qualifying studies are prospective quasi-experiments (i.e., matched studies). Quasiexperimental studies (e.g., Regression Discontinuity Design) are those in which students have not been randomly assigned to treatment or control groups, but researchers are using statistical matching methods that allow them to speak with confidence about the likelihood that an intervention causes an outcome. The research studies use large, multi-site samples. No other experimental or quasiexperimental research shows that the intervention negatively affects the outcome. Researchers have found that the intervention improves outcomes for the specific student subgroups that the district or school intends to support with the intervention."
            + "\nRED - The topic has a study that would have qualified for “green” or “yellow” but did not because it failed to account for clustering (but did obtain significantly positive outcomes at the student level) or did not meet the sample size requirements. Post-hoc or retrospective studies may also qualify. Correlational studies (e.g., studies that can show a relationship between the intervention and outcome but cannot show causation) have found that the intervention likely improves a relevant student outcome (e.g., reading scores, attendance rates). The studies do not have to be based on large, multi-site samples. No other experimental or quasiexperimental research shows that the intervention negatively affects the outcome."
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
    expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(data={"sub": str(user.id), "email": user.email, "name": user.name}, expires_delta=expires)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/chat")
async def chat_api(query: Query, current_user: User = Depends(get_current_user)):
    user_query = query.message

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'update', 'content': 'Finding relevant studies...'})}\n\n"
            step_1_result = await get_studies(user_query)
            yield f"data: {json.dumps({'type': 'step_result', 'step': 1, 'content': step_1_result})}\n\n"

            yield f"data: {json.dumps({'type': 'update', 'content': 'Extracting study data...'})}\n\n"
            step_2_result = await extract_studies_data(step_1_result)
            yield f"data: {json.dumps({'type': 'step_result', 'step': 2, 'content': step_2_result})}\n\n"

            yield f"data: {json.dumps({'type': 'update', 'content': 'Analyzing study data...'})}\n\n"
            analysis_result = await analyze_studies(step_2_result)
            
            result_data = {"type": "result", "content": analysis_result.model_dump()}
            yield f"data: {json.dumps(result_data, cls=CustomEncoder)}\n\n"

        except Exception as e:
            print(f"An error occurred in the stream: {e}")
            error_data = {"type": "error", "content": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- MARA Logic and Prompt Composition ---
async def get_studies(user_query: str) -> str:
    step_1_query = compose_step_one_query(user_query)
    response = await client.generate_content_async(step_1_query)
    return response.text

async def extract_studies_data(step_1_result: str) -> str:
    step_2_query = compose_step_two_query(step_1_result)
    response = await client.generate_content_async(step_2_query)
    return response.text

async def analyze_studies(step_2_result: str) -> AnalysisResponse:
    step_3_query = compose_step_three_query(step_2_result)
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=AnalysisResponse,
    )
    response = await client.generate_content_async(step_3_query, generation_config=generation_config)
    return AnalysisResponse.model_validate_json(response.text)

def compose_step_one_query(user_query: str) -> str:
    return (
        common_persona_prompt
        + " Find me high-quality studies that look into the question of: " + user_query
        + "\nPlease optimize your search per the following constraints: "
        + "\n1. Search online databases that index published literature, as well as sources such as Google Scholar."
        + "\n2. Find studies per retrospective reference harvesting and prospective forward citation searching."
        + "\n3. Attempt to identify unpublished literature such as dissertations and reports from independent research firms."
        + "\nExclude any studies which either:"
        + "\n1. lack a comparison or control group."
        + "\n2. are purely correlational, that do not include either a randomized-controlled trial, quasi-experimental design, or regression discontinuity"
        + "\nFinally, return these studies in a list of highest quality to lowest, formatting that list by: 'Title, Authors, Date Published.' "
        + "\nInclude at least 30 studies, or if fewer than 30 the max available."
        + "\nKeep your response brief, only including that raw list and nothing more."
    )

def compose_step_two_query(step_1_result: str) -> str:
    return (
        common_persona_prompt
        + " First, lookup the papers for each of the studies in this list.\n" + step_1_result
        + "\n Then, extract the following data to compile into a spreadsheet."
        + "\nSpecifically, organize the data for each study into the following columns: "
        + "\n1. Sample size of treatment and comparison groups"
        + "\n2. Cluster sample sizes (i.e. size of the classroom or school of however the individuals are clustered)"
        + "\n3. Intraclass correlation coefficient (ICC; when available) will be coded for cluster studies. When the ICC estimates are not provided, impute a constant value of 0.20."
        + "\n4. Effect size for each outcome analysis will be calculated and recorded. These should be the standardized mean difference between the treatment and control group at post-test, ideally adjusted for pre-test differences."
        + "\nAuthors can report this in varying ways. The preference is for adjusted effects, found in a linear regression. If adjusted effects are unavailable, raw means and standard deviations can be used."
        + "\n6. Study design (i.e., randomized controlled trial, quasi-experimental, or regression discontinuity)"
        + "\nReturn the results in a spreadsheet, where each row is for each study and each column is for each column feature in the above list."
        + "\nKeep your response brief, only including those spreadsheet rows and nothing more."
    )

def compose_step_three_query(step_2_result: str) -> str:
   return (
       common_persona_prompt
       + "\nUsing this dataset: " + step_2_result
       + "\ncreate a simple model with only the impact of the main predictor of interest. Specifically, use a multivariate meta-regression model to conduct the meta-analysis."
       + "\nDetermine the Confidence level per the following criteria: " + Confidence.get_description()
       + "Return this in the Confidence enum."
       + "\nGenerate an overview summarizing the analysis conclusion, in one or two sentences. Return this in the response Summary."
       + "\nInclude all other details in the response Details, making sure to include a description of the analysis process used, the regression models produced, and any correpsonding plots, in the corresponding AnalysisDetails fields."
   )