from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import enum
import asyncio

# --- Client Initialization ---
client = None
common_persona_prompt = "You are a senior data analyst with a specialty in meta-analysis."

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Initializes the Gemini client when the application starts.
    """
    global client
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("FATAL: GEMINI_API_KEY environment variable not set.")
    
    genai.configure(api_key=GEMINI_API_KEY)
    client = genai.GenerativeModel("gemini-1.5-pro-latest")
    print("✅ GenAI Client configured and initialized successfully.")


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
        if not client:
            error_data = {"type": "error", "content": "Server error: AI client not initialized."}
            yield f"data: {json.dumps(error_data)}\n\n"
            return
            
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
# get_studies and its prompt are unchanged
async def get_studies(user_query: str) -> str:
    if not user_query:
        raise ValueError("Step 1: user_query is empty.")
    step_1_query = compose_step_one_query(user_query)
    response = await client.generate_content_async(step_1_query)
    print(f"🔎 Step 1 Raw Response: {response}")
    if not response.text:
        raise ValueError("Step 1: No response from Gemini.")
    return response.text

def compose_step_one_query(user_query: str) -> str:
    return (
        common_persona_prompt
        + " Find me high-quality studies that look into the question of: "
        + user_query
        + "\nPlease optimize your search per the following constraints: "
        # ... (rest of prompt is unchanged) ...
        + "\nIMPORTANT: Your entire response must ONLY be the raw list of studies. Do NOT include any preamble, postamble, notes, explanations, or any other conversational text."
    )

# extract_studies_data and its prompt are unchanged
async def extract_studies_data(step_1_result: str) -> str:
    if not step_1_result:
        raise ValueError("Step 2: step_1_result is empty.")
    step_2_query = compose_step_two_query(step_1_result)
    response = await client.generate_content_async(step_2_query)
    print(f"🔎 Step 2 Raw Response: {response}")
    if not response.text:
        raise ValueError("Step 2: No response from Gemini.")
    return response.text

def compose_step_two_query(step_1_result: str) -> str:
    return (
        common_persona_prompt
        + " First, Lookup the papers for each of the studies in this list."
        + "\n"
        + step_1_result
        # ... (rest of prompt is unchanged) ...
        + "\nIMPORTANT: Your entire response must ONLY be the raw spreadsheet data. Do NOT include any preamble, notes, explanations, or any other conversational text."
    )


# UPDATED FUNCTION: Added a safeguard to handle casing issues
async def analyze_studies(step_2_result: str) -> AnalysisResponse:
    if not step_2_result:
        raise ValueError("Step 3: step_2_result is empty.")
    step_3_query = compose_step_three_query(step_2_result)

    response = await client.generate_content_async(
        step_3_query,
        generation_config={"response_mime_type": "application/json"},
    )
    
    print(f"🔎 Step 3 Raw Response: {response}")

    try:
        raw_dict = json.loads(response.text)
        
        # ADDED SAFEGUARD: Convert all keys to lowercase before validation.
        sanitized_dict = {k.lower(): v for k, v in raw_dict.items()}
        
        return AnalysisResponse(**sanitized_dict)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"🔴 FAILED TO PARSE/VALIDATE JSON FROM GEMINI. Raw text was: {response.text}")
        raise ValueError(f"Step 3 failed because the API did not return valid JSON. Error: {e}")


# UPDATED PROMPT: More explicit instructions for JSON keys
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
        # ADDED: Stricter instruction about JSON format and keys
        + "\nIMPORTANT: Your entire response must ONLY be a raw JSON object. The keys MUST be exactly `summary`, `confidence`, and `details` in all lowercase. Do not wrap the JSON in markdown or include any other text."
    )