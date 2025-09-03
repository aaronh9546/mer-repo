from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.generativeai.types import GenerationConfig # Import GenerationConfig
import os
import enum

# --- Environment and API Client Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# CORRECTED: Use genai.configure and genai.GenerativeModel
genai.configure(api_key=GEMINI_API_KEY)
# Initialize the model once to be reused
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

common_persona_prompt = "You are a senior data analyst with a specialty in meta-analysis."

app = FastAPI()

# --- CORS Middleware ---
origins = [
    "https://timothy-han.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas (Data Models) ---

class Query(BaseModel):
    """Schema for the initial user query."""
    message: str

class Confidence(str, enum.Enum):
    """Enum for the confidence level of the analysis."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

    @staticmethod
    def get_description() -> str:
        return (
            "GREEN - If the research on the topic has a well-conducted, randomized study showing a statistically significant positive effect on at least one outcome measure (e.g., state test or national standardized test) analyzed at the proper level of clustering (class/school or student) with a multi-site sample of at least 350 participants. Strong evidence from at least one well-designed and wellimplemented experimental study..."
            "\nYELLOW - If it meets all standards for “green” stated above, except that instead of using a randomized design, qualifying studies are prospective quasi-experiments (i.e., matched studies)..."
            "\nRED - The topic has a study that would have qualified for “green” or “yellow” but did not because it failed to account for clustering (but did obtain significantly positive outcomes at the student level) or did not meet the sample size requirements..."
        )

class AnalysisDetails(BaseModel):
    """Schema for the detailed components of the analysis."""
    regression_models: str
    process: str
    plots: str

class AnalysisResponse(BaseModel):
    """Schema for the final, structured analysis from Step 3."""
    summary: str
    confidence: Confidence
    details: AnalysisDetails

class FollowupQuery(BaseModel):
    """Schema for a follow-up question."""
    followup_message: str
    step_2_data: str
    step_3_analysis: str

# --- Helper Function ---

def extract_text(response) -> str | None:
    """Safely extract text from a Gemini response object for Steps 1 and 2."""
    try:
        if response and hasattr(response, 'text'):
            return response.text
    except Exception as e:
        print(f"⚠️ Error extracting text: {e}")
    return None

# --- API Endpoints ---

@app.get("/")
def read_root():
    """Provides a welcome message and example questions."""
    return {
        "message": "Hello! I’m MARA, ready to help answer your questions about what works in education with strong evidence and systematic meta-analytic methods.",
        "examples": [
            "What is the overall effect of direct instruction on student achievement?",
            "What is the impact of homework on academic performance by grade level?",
            "Do educational technology tools improve student achievement?",
            "What is the effect of reduced class sizes on student achievement?",
        ]
    }

@app.post("/chat", response_model=AnalysisResponse)
def chat_api(query: Query):
    """Main endpoint to perform the three-step meta-analysis."""
    user_query = query.message
    if not user_query:
        raise HTTPException(status_code=400, detail="Query message is required.")

    print(f"🚀 Starting investigation for: '{user_query}'")

    try:
        step_1_result = get_studies(user_query)
        print("✅ Step 1: Found relevant studies.")

        step_2_result = extract_studies_data(step_1_result)
        print("✅ Step 2: Extracted data from studies.")

        step_3_result = analyze_studies(step_2_result)
        print("✅ Step 3: Completed analysis.")

        print("🎉 Investigation complete. Sending response.")
        return step_3_result

    except Exception as e:
        print(f"🔥 An error occurred during the process: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.post("/followup")
def followup_api(query: FollowupQuery):
    """Endpoint to handle follow-up questions based on the initial analysis."""
    print("🔎 Handling a follow-up question...")
    try:
        prompt = (
            f"Given the initial dataset:\n---DATA---\n{query.step_2_data}\n\n"
            f"And the initial analysis:\n---ANALYSIS---\n{query.step_3_analysis}\n\n"
            f"Please answer the following follow-up question: '{query.followup_message}'"
        )
        
        # CORRECTED: Call generate_content on the model instance
        response = gemini_model.generate_content(prompt)
        
        reply = extract_text(response)
        if not reply:
            raise ValueError("Failed to generate a follow-up response.")
            
        print("✅ Follow-up response generated.")
        return {"reply": reply}
        
    except Exception as e:
        print(f"🔥 An error occurred during the follow-up: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process follow-up: {e}")

# --- Step 1: Get Studies ---
def get_studies(user_query: str) -> str:
    prompt = compose_step_one_query(user_query)
    # CORRECTED: Call generate_content on the model instance
    response = gemini_model.generate_content(prompt)
    text = extract_text(response)
    if not text:
        raise ValueError("Step 1: Failed to get a valid response from Gemini for finding studies.")
    return text

def compose_step_one_query(user_query: str) -> str:
    # This function is correct, no changes needed
    return (
        f"{common_persona_prompt} "
        f"Find me high-quality studies that look into the question of: {user_query}\n"
        "Please optimize your search per the following constraints:\n"
        "1. Search online databases that index published literature, as well as sources such as Google Scholar.\n"
        "2. Find studies per retrospective reference harvesting and prospective forward citation searching.\n"
        "3. Attempt to identify unpublished literature such as dissertations and reports from independent research firms.\n"
        "Exclude any studies which either:\n"
        "1. lack a comparison or control group.\n"
        "2. are purely correlational, that do not include either a randomized-controlled trial, quasi-experimental design, or regression discontinuity.\n"
        "Finally, return these studies in a list of highest quality to lowest, formatting that list by: 'Title, Authors, Date Published.'\n"
        "Include at least 30 studies, or if fewer than 30 the max available.\n"
        "Keep your response brief, only including that raw list and nothing more."
    )

# --- Step 2: Extract Data ---
def extract_studies_data(step_1_result: str) -> str:
    prompt = compose_step_two_query(step_1_result)
    # CORRECTED: Call generate_content on the model instance
    response = gemini_model.generate_content(prompt)
    text = extract_text(response)
    if not text:
        raise ValueError("Step 2: Failed to get a valid response from Gemini for data extraction.")
    return text

def compose_step_two_query(step_1_result: str) -> str:
    # This function is correct, no changes needed
    return (
        f"{common_persona_prompt} "
        f"First, lookup the papers for each of the studies in this list:\n{step_1_result}\n"
        "Then, extract the following data to compile into a spreadsheet.\n"
        "Specifically, organize the data for each study into the following columns:\n"
        "1. Sample size of treatment and comparison groups\n"
        "2. Cluster sample sizes (i.e. size of the classroom or school of however the individuals are clustered)\n"
        "3. Intraclass correlation coefficient (ICC; when available) will be coded for cluster studies. When the ICC estimates are not provided, impute a constant value of 0.20.\n"
        "4. Effect size for each outcome analysis will be calculated and recorded. These should be the standardized mean difference between the treatment and control group at post-test, ideally adjusted for pre-test differences. Authors can report this in varying ways. The preference is for adjusted effects, found in a linear regression. If adjusted effects are unavailable, raw means and standard deviations can be used.\n"
        "5. Study design (i.e., randomized controlled trial, quasi-experimental, or regression discontinuity)\n"
        "Return the results in a spreadsheet, where each row is for each study and each column is for each column feature in the above list.\n"
        "Keep your response brief, only including those spreadsheet rows and nothing more."
    )

# --- Step 3: Analyze Studies (with JSON output) ---
def analyze_studies(step_2_result: str) -> AnalysisResponse:
    prompt = compose_step_three_query(step_2_result)
    
    # THIS IS THE KEY FIX
    # We create a GenerationConfig object and pass it to the 'generation_config' parameter.
    config = GenerationConfig(
        response_mime_type="application/json",
        response_schema=AnalysisResponse,
    )
    
    # CORRECTED: Call generate_content on the model instance with the config object
    response = gemini_model.generate_content(
        prompt,
        generation_config=config
    )
    
    if not hasattr(response, 'text') or not response.text: # Check .text for JSON string
        raise ValueError("Step 3: Failed to get a valid JSON response from Gemini for analysis.")
    
    # The SDK handles parsing when a schema is provided. Access it via response.text
    # and Pydantic will validate it in the FastAPI response model.
    # The .candidates[0].content.parts[0].text route is also valid.
    return AnalysisResponse.parse_raw(response.text)


def compose_step_three_query(step_2_result: str) -> str:
    # This function is correct, no changes needed
    return (
        f"{common_persona_prompt}\n"
        f"Using this dataset:\n{step_2_result}\n"
        "Create a simple model with only the impact of the main predictor of interest. Specifically, use a multivariate meta-regression model to conduct the meta-analysis.\n"
        f"Determine the Confidence level per the following criteria:\n{Confidence.get_description()}\n"
        "Return this in the Confidence enum.\n"
        "Generate an overview summarizing the analysis conclusion, in one or two sentences. Return this in the response Summary.\n"
        "Include all other details in the response Details, making sure to include a description of the analysis process used, the regression models produced, and any corresponding plots, in the corresponding AnalysisDetails fields."
    )