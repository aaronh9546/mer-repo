from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import enum

# --- Environment Variable & API Client Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=GEMINI_API_KEY)
gemini_model = "gemini-1.5-pro-latest" # Using the latest recommended model
common_persona_prompt = "You are a senior data analyst with a specialty in meta-analysis."

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

# --- Pydantic Schemas for API Input and Output ---

# Input schema for the user's initial question
class Query(BaseModel):
    message: str

# Schemas for the structured JSON response from Step 3
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

class AnalysisDetails(BaseModel):
    regression_models: str
    process: str
    plots: str

class AnalysisResponse(BaseModel):
    summary: str
    confidence: Confidence
    details: AnalysisDetails


# --- Helper Function for Text Extraction (Steps 1 & 2) ---
def extract_text(response):
    """Safely extract text from a Gemini response object."""
    try:
        if response and hasattr(response, 'text'):
            return response.text
        if response and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content.parts:
                return candidate.content.parts[0].text
    except (AttributeError, IndexError) as e:
        print(f"⚠️ Error during text extraction: {e}. Raw response: {response}")
    return None


# --- Main API Endpoint ---
@app.post("/chat", response_model=AnalysisResponse)
def chat_api(query: Query):
    user_query = query.message
    if not user_query:
        raise HTTPException(status_code=400, detail="Query message is required.")

    print("Starting an investigation into:", user_query)

    try:
        # Step 1: Find relevant studies (returns plain text)
        step_1_result = step_one_find_studies(user_query)

        # Step 2: Extract data from studies (returns plain text)
        step_2_result = step_two_extract_data(step_1_result)

        # Step 3: Analyze data and get structured JSON response
        step_3_result = step_three_analyze_data(step_2_result)
        
        print("Investigation complete. Returning structured analysis.")
        return step_3_result

    except ValueError as ve:
        print(f"Data processing error: {ve}")
        raise HTTPException(status_code=500, detail=f"An error occurred during data processing: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected internal error occurred: {e}")


# -------------------------------------------------------------------
# MARA Analysis Steps
# -------------------------------------------------------------------

def step_one_find_studies(user_query: str) -> str:
    """Finds a list of high-quality studies based on the user's query."""
    if not user_query:
        raise ValueError("Step 1: user_query cannot be empty.")
    
    print("Step 1: Finding relevant studies...")
    step_1_query = compose_step_one_query(user_query)
    step_1_response = client.models.generate_content(
        model=gemini_model,
        contents=step_1_query,
    )
    text = extract_text(step_1_response)
    if not text:
        raise ValueError("Step 1: Failed to get a valid text response from the API.")
    print("Step 1: Found studies.")
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
    """Extracts structured data from the list of studies found in step one."""
    if not step_1_result:
        raise ValueError("Step 2: Input from step 1 cannot be empty.")
    
    print("Step 2: Extracting study data...")
    step_2_query = compose_step_two_query(step_1_result)
    step_2_response = client.models.generate_content(
        model=gemini_model,
        contents=step_2_query,
    )
    text = extract_text(step_2_response)
    if not text:
        raise ValueError("Step 2: Failed to get a valid text response from the API.")
    print("Step 2: Extracted data.")
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
    """Performs meta-analysis and returns a structured JSON object."""
    if not step_2_result:
        raise ValueError("Step 3: Input from step 2 cannot be empty.")
    
    print("Step 3: Analyzing data for final report...")
    step_3_query = compose_step_three_query(step_2_result)
    step_3_response = client.models.generate_content(
        model=gemini_model,
        contents=step_3_query,
        config={
            "response_mime_type": "application/json",
            "response_schema": AnalysisResponse,
        },
    )
    if not step_3_response or not hasattr(step_3_response, 'parsed'):
        raise ValueError(f"Step 3: Failed to get a valid parsed JSON response from the API. Raw response: {step_3_response}")
    
    parsed_response: AnalysisResponse = step_3_response.parsed
    print("Step 3: Analysis complete.")
    return parsed_response

def compose_step_three_query(step_2_result: str) -> str:
    return (
        common_persona_prompt
        + "\nUsing this dataset: "
        + step_2_result
        + "\ncreate a simple model with only the impact of the main predictor of interest. Specifically