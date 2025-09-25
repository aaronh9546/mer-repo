from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import enum
import json
import asyncio
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone

# --- Environment Variable & API Client Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)

gemini_model_name = "gemini-1.5-pro-latest"
model = genai.GenerativeModel(gemini_model_name)

common_persona_prompt = "You are a senior data analyst with a specialty in meta-analysis."

# --- Authentication Code ---
CML_SHARED_SECRET_KEY = os.getenv("CML_SHARED_SECRET_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not CML_SHARED_SECRET_KEY or not JWT_SECRET_KEY:
    raise RuntimeError("CML_SHARED_SECRET_KEY and JWT_SECRET_KEY must be set in your Render environment.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

async def verify_internal_secret(x_internal_secret: str = Header(None)):
    if not x_internal_secret or x_internal_secret != CML_SHARED_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing internal secret")

class WordPressUser(BaseModel):
    id: int
    email: str
    name: str

class Token(BaseModel):
    access_token: str
    token_type: str

app = FastAPI()

# --- CORS Middleware Setup ---
origins = [
    "https://myeducationresearcher.com",
    "https://timothy-han.com",
    "https://aaronhanto-nyozw.wpcomstaging.com",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas for Chat API ---
class Query(BaseModel):
    message: str

class Confidence(str, enum.Enum):
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

# Use Field(alias='...') to map the AI's capitalized keys to our lowercase variables.
class AnalysisDetails(BaseModel):
    regression_models: str = Field(alias="RegressionModel")
    process: str = Field(alias="AnalysisProcess")
    plots: str = Field(alias="Plots")

class AnalysisResponse(BaseModel):
    summary: str = Field(alias="Summary")
    confidence: Confidence = Field(alias="Confidence")
    details: AnalysisDetails = Field(alias="Details")

# --- Authentication Endpoint ---
@app.post("/auth/issue-wordpress-token", response_model=Token, dependencies=[Depends(verify_internal_secret)])
async def issue_wordpress_token(user: WordPressUser):
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode = {"sub": str(user.id), "exp": expire}
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": encoded_jwt, "token_type": "bearer"}

# --- Helper Function ---
def extract_text(response):
    try:
        if response and hasattr(response, 'text'): return response.text
        if response and hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].content.parts[0].text
    except (AttributeError, IndexError) as e:
        print(f"⚠️ Error during text extraction: {e}. Raw response: {response}")
    return None

# --- Main Chat API Endpoint ---
@app.post("/chat")
async def chat_api(query: Query):
    user_query = query.message
    if not user_query: raise HTTPException(status_code=400, detail="Query message is required.")
    async def event_generator():
        try:
            print(f"LOG: Starting investigation for query: '{user_query}'")
            yield "data: Finding relevant studies...\n\n"
            step_1_result = step_one_find_studies(user_query)
            print("LOG: Step 1 complete.")
            yield "data: Found relevant studies.\n\n"
            await asyncio.sleep(0.1)
            print("LOG: Starting Step 2.")
            yield "data: Extracting study data...\n\n"
            step_2_result = step_two_extract_data(step_1_result)
            print("LOG: Step 2 complete.")
            yield "data: Extracted study data.\n\n"
            await asyncio.sleep(0.1)
            print("LOG: Starting Step 3.")
            yield "data: Analyzing study data...\n\n"
            step_3_result = step_three_analyze_data(step_2_result)
            print("LOG: Step 3 complete.")
            yield "data: Analyzed study data.\n\n"
            final_json = step_3_result.model_dump_json()
            yield f"event: result\n"
            yield f"data: {final_json}\n\n"
        except Exception as e:
            print(f"ERROR: An error occurred during the stream: {e}")
            error_message = json.dumps({"error": f"An internal error occurred: {e}"})
            yield "event: error\n"
            yield f"data: {error_message}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -------------------------------------------------------------------
# MARA Analysis Steps
# -------------------------------------------------------------------

def step_one_find_studies(user_query: str) -> str:
    if not user_query: raise ValueError("Step 1: user_query cannot be empty.")
    step_1_query = compose_step_one_query(user_query)
    step_1_response = model.generate_content(contents=step_1_query)
    text = extract_text(step_1_response)
    if not text: raise ValueError("Step 1: Failed to get a valid text response from the API.")
    return text

def compose_step_one_query(user_query: str) -> str:
    return (
        common_persona_prompt
        + " Find me high-quality studies that look into the question of: "
        + user_query
        + "\nPlease optimize your search per the following constraints: "
        + "\n1. Search online databases that index published literature, as well as sources such as Google Scholar."
        + "\n2. Find studies per retrospective reference harvesting and prospective forward citation searching."
        + "\n3. Attempt to identify unpublished literature such as dissertations and reports from independent research firms."
        + "\nExclude any studies which either:"
        + "\n1. lack a comparison or control group."
        + "\n2. are purely correlational, that do not include either a randomized-controlled trial, quasi-experimental design, or regression discontinuity"
        + "\nFinally, return these studies in a list of highest quality to lowest, formatting that list by: 'Title, Authors, Date Published.' "
        + "\nKeep your response brief, only including that raw list and nothing more."
    )

def step_two_extract_data(step_1_result: str) -> str:
    if not step_1_result: raise ValueError("Step 2: Input from step 1 cannot be empty.")
    step_2_query = compose_step_two_query(step_1_result)
    step_2_response = model.generate_content(contents=step_2_query)
    text = extract_text(step_2_response)
    if not text: raise ValueError("Step 2: Failed to get a valid text response from the API.")
    return text

def compose_step_two_query(step_1_result: str) -> str:
    return (
        common_persona_prompt
        + " First, lookup the papers for each of the studies in this list."
        + "\n"
        + step_1_result
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

def step_three_analyze_data(step_2_result: str) -> AnalysisResponse:
    if not step_2_result: raise ValueError("Step 3: Input from step 2 cannot be empty.")
    step_3_query = compose_step_three_query(step_2_result)
    step_3_response = model.generate_content(
        contents=step_3_query,
        generation_config={"response_mime_type": "application/json"},
    )
    try:
        parsed_response = AnalysisResponse.model_validate_json(step_3_response.text)
    except Exception as e:
        # Raise a more informative error that includes the raw text for debugging
        raise ValueError(f"Step 3: Failed to parse JSON response from AI. Error: {e}. Raw text: {step_3_response.text}")
    return parsed_response

def compose_step_three_query(step_2_result: str) -> str:
    # Changed "...in the corresponding AnalysisDetails fields" to "...in the corresponding Details fields"
    # to prevent extra nesting.
    return (
        common_persona_prompt
        + "\nUsing this dataset: "
        + step_2_result
        + "\ncreate a simple model with only the impact of the main predictor of interest. Specifically, use a multivariate meta-regression model to conduct the meta-analysis."
        + "\nDetermine the Confidence level per the following criteria: "
        + Confidence.get_description()
        + "Return this in the Confidence enum."
        + "\nGenerate an overview summarizing the analysis conclusion, in one or two sentences. Return this in the response Summary."
        + "\nInclude a description of the analysis process, the regression models produced, and any corresponding plots, in the corresponding Details fields."
    )