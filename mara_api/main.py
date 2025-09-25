from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse # Used for streaming updates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import enum
import json # Used to format the final JSON payload in the stream
import asyncio # Used for non-blocking operations in async functions

# --- Environment Variable & API Client Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=GEMINI_API_KEY)
gemini_model = "gemini-1.5-pro-latest"
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

# --- Main API Endpoint (Updated for Streaming) ---
@app.post("/chat")
async def chat_api(query: Query): # 👈 Changed to async def
    user_query = query.message
    if not user_query:
        raise HTTPException(status_code=400, detail="Query message is required.")

    async def event_generator():
        """This is the generator that yields updates to the client."""
        try:
            # --- Step 1: Find Studies ---
            print(f"LOG: Starting investigation for query: '{user_query}'")
            yield "data: Finding relevant studies...\n\n"
            step_1_result = step_one_find_studies(user_query)
            print("LOG: Step 1 complete. Found studies list.")
            yield "data: Found relevant studies.\n\n"
            
            # A small delay to ensure messages are sent and processed
            await asyncio.sleep(0.1)

            # --- Step 2: Extract Data ---
            print("LOG: Starting Step 2 - Extracting data from studies.")
            yield "data: Extracting study data...\n\n"
            step_2_result = step_two_extract_data(step_1_result)
            print("LOG: Step 2 complete. Extracted study data.")
            yield "data: Extracted study data.\n\n"
            
            await asyncio.sleep(0.1)

            # --- Step 3: Analyze Data ---
            print("LOG: Starting Step 3 - Analyzing data.")
            yield "data: Analyzing study data...\n\n"
            step_3_result = step_three_analyze_data(step_2_result)
            print("LOG: Step 3 complete. Analysis finished.")
            yield "data: Analyzed study data.\n\n"

            # --- Send Final Result ---
            print("LOG: Sending final analysis response to client.")
            final_json = step_3_result.model_dump_json()
            yield f"event: result\n" # Use a custom event name for the final data
            yield f"data: {final_json}\n\n"

        except Exception as e:
            # --- Handle Errors in the Stream ---
            print(f"ERROR: An error occurred during the stream: {e}")
            error_message = json.dumps({"error": f"An internal error occurred: {e}"})
            yield "event: error\n"
            yield f"data: {error_message}\n\n"
            
    # Return the generator in a StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -------------------------------------------------------------------
# MARA Analysis Steps (No changes to logic needed)
# -------------------------------------------------------------------

def step_one_find_studies(user_query: str) -> str:
    # ... (code is identical to previous version)
    if not user_query:
        raise ValueError("Step 1: user_query cannot be empty.")
    step_1_query = compose_step_one_query(user_query)
    step_1_response = client.models.generate_content(
        model=gemini_model, contents=step_1_query,
    )
    text = extract_text(step_1_response)
    if not text:
        raise ValueError("Step 1: Failed to get a valid text response from the API.")
    return text

def compose_step_one_query(user_query: str) -> str:
    # ... (code is identical to previous version)
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
    # ... (code is identical to previous version)
    if not step_1_result:
        raise ValueError("Step 2: Input from step 1 cannot be empty.")
    step_2_query = compose_step_two_query(step_1_result)
    step_2_response = client.models.generate_content(
        model=gemini_model, contents=step_2_query,
    )
    text = extract_text(step_2_response)
    if not text:
        raise ValueError("Step 2: Failed to get a valid text response from the API.")
    return text

def compose_step_two_query(step_1_result: str) -> str:
    # ... (code is identical to previous version)
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
    # ... (code is identical to previous version)
    if not step_2_result:
        raise ValueError("Step 3: Input from step 2 cannot be empty.")
    step_3_query = compose_step_three_query(step_2_result)
    step_3_response = client.models.generate_content(
        model=gemini_model,
        contents=step_3_query,
        config={"response_mime_type": "application/json", "response_schema": AnalysisResponse},
    )
    if not step_3_response or not hasattr(step_3_response, 'parsed'):
        raise ValueError(f"Step 3: Failed to get a valid parsed JSON response from the API. Raw response: {step_3_response}")
    parsed_response: AnalysisResponse = step_3_response.parsed
    return parsed_response

def compose_step_three_query(step_2_result: str) -> str:
    # ... (code is identical to previous version)
    return (
        common_persona_prompt
        + "\nUsing this dataset: "
        + step_2_result
        + "\ncreate a simple model with only the impact of the main predictor of interest. Specifically, use a multivariate meta-regression model to conduct the meta-analysis."
        + "\nDetermine the Confidence level per the following criteria: "
        + Confidence.get_description()
        + "Return this in the Confidence enum."
        + "\nGenerate an overview summarizing the analysis conclusion, in one or two sentences. Return this in the response Summary."
        + "\nInclude all other details in the response Details, making sure to include a description of the analysis process used, the regression models produced, and any correpsonding plots, in the corresponding AnalysisDetails fields."
    )