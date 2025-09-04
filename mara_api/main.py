from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import enum
import asyncio

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

client = genai.GenerativeModel("gemini-1.5-pro-latest")
common_persona_prompt = "You are a senior data analyst with a specialty in meta-analysis."

app = FastAPI()

# --- CORS setup ---
origins = [
    "https://aaronhanto-nyozw.com",
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

# --- Schema (no changes) ---
class Query(BaseModel):
    message: str

class Confidence(enum.Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    
    @staticmethod
    def get_description():
        return (
            "GREEN - If the research on the topic has a well-conducted, randomized study..."
            # (rest of description is unchanged)
        )

class AnalysisDetails(BaseModel):
    regression_models: str
    process: str
    plots: str

class AnalysisResponse(BaseModel):
    summary: str
    confidence: Confidence
    details: AnalysisDetails


# --- API endpoint (no changes) ---
@app.post("/chat")
async def chat_api(query: Query):
    user_query = query.message
    
    async def event_generator():
        try:
            # Step 1
            yield f"data: {json.dumps({'type': 'update', 'content': 'Finding relevant studies...'})}\n\n"
            step_1_result = await get_studies(user_query)

            # Step 2
            yield f"data: {json.dumps({'type': 'update', 'content': 'Extracting study data...'})}\n\n"
            step_2_result = await extract_studies_data(step_1_result)

            # Step 3
            yield f"data: {json.dumps({'type': 'update', 'content': 'Analyzing study data...'})}\n\n"
            analysis_result = await analyze_studies(step_2_result)
            
            # Final result
            result_data = {"type": "result", "content": analysis_result.model_dump()}
            yield f"data: {json.dumps(result_data)}\n\n"
            
            yield f"data: {json.dumps({'type': 'update', 'content': 'Analysis complete. Goodbye from MARA!'})}\n\n"

        except Exception as e:
            print(f"An error occurred in the stream: {e}")
            error_data = {"type": "error", "content": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ------------------------
# MARA Steps
# ------------------------
async def get_studies(user_query: str) -> str:
    if not user_query:
        raise ValueError("Step 1: user_query is empty.")
    step_1_query = compose_step_one_query(user_query)
    
    response = await client.generate_content_async(step_1_query)
    
    print(f"🔎 Step 1 Raw Response: {response}")

    if not response.text:
        raise ValueError("Step 1: No response from Gemini.")
    return response.text

# UPDATED PROMPT
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
        + "\nInclude at least 30 studies, or if fewer than 30 the max available."
        # ADDED: Stricter instruction
        + "\nIMPORTANT: Your entire response must ONLY be the raw list of studies. Do NOT include any preamble, postamble, notes, explanations, or any other conversational text."
    )

async def extract_studies_data(step_1_result: str) -> str:
    if not step_1_result:
        raise ValueError("Step 2: step_1_result is empty.")
    step_2_query = compose_step_two_query(step_1_result)

    response = await client.generate_content_async(step_2_query)
    
    print(f"🔎 Step 2 Raw Response: {response}")

    if not response.text:
        raise ValueError("Step 2: No response from Gemini.")
    return response.text

# UPDATED PROMPT
def compose_step_two_query(step_1_result: str) -> str:
    return (
        common_persona_prompt
        + " First, Lookup the papers for each of the studies in this list."
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
        # ADDED: Stricter instruction
        + "\nIMPORTANT: Your entire response must ONLY be the raw spreadsheet data. Do NOT include any preamble, notes, explanations, or any other conversational text."
    )

async def analyze_studies(step_2_result: str) -> AnalysisResponse:
    if not step_2_result:
        raise ValueError("Step 3: step_2_result is empty.")
    step_3_query = compose_step_three_query(step_2_result)

    response = await client.generate_content_async(
        step_3_query,
        generation_config={"response_mime_type": "application/json"},
    )
    
    print(f"🔎 Step 3 Raw Response: {response}")

    # Add a try-except block for more robust JSON parsing
    try:
        parsed_json = json.loads(response.text)
        return AnalysisResponse(**parsed_json)
    except json.JSONDecodeError as e:
        print(f"🔴 FAILED TO PARSE JSON FROM GEMINI. Raw text was: {response.text}")
        raise ValueError(f"Step 3 failed because the API did not return valid JSON. Error: {e}")


# UPDATED PROMPT
def compose_step_three_query(step_2_result: str) -> str:
    return (
        common_persona_prompt
        + "\nUsing this dataset: "
        + step_2_result
        + "\ncreate a simple model with only the impact of the main predictor of interest. Specifically, use a multivariate meta-regression model to conduct the meta-analysis."
        + "\nDetermine the Confidence level per the following criteria: "
        + Confidence.get_description()
        + "\nReturn this in the Confidence enum."
        + "\nGenerate an overview summarizing the analysis conclusion, in one or two sentences. Return this in the response Summary."
        + "\nInclude all other details in the response Details, making sure to include a description of the analysis process used, the regression models produced, and any correpsonding plots, in the corresponding AnalysisDetails fields."
        # ADDED: Stricter instruction
        + "\nIMPORTANT: Your entire response must ONLY be the raw JSON object that conforms to the schema. Do not wrap it in markdown, or include any other text."
    )