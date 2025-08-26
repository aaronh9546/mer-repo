import time
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
gemini_model = "gemini-2.5-pro"
common_persona_prompt = "You are a senior data analyst with a specialty in meta-analysis."

app = FastAPI()

# --- CORS setup ---
origins = [
    "https://aaronhanto-nyozw.com",
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

# --- Schema ---
class Query(BaseModel):
    message: str

# --- Helper: retry wrapper (2 retries) ---
def call_gemini_with_retry(contents, retries=2, delay=2):
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(model=gemini_model, contents=contents)
            return response.text
        except Exception as e:
            print(f"Gemini attempt {attempt} failed:", e)
            if attempt < retries:
                time.sleep(delay)
            else:
                raise  # re-raise after last attempt

# --- API endpoint with streaming-like JSON response ---
@app.post("/chat")
def chat_api(query: Query):
    user_query = query.message
    print("Starting an investigation into:", user_query)

    step_results = {}

    try:
        # Step 1
        step_results['step_1'] = step_one(user_query)

        # Step 2
        step_results['step_2'] = step_two(step_results['step_1'])

        # Step 3
        step_results['step_3'] = step_three(step_results['step_2'])

        # Final reply = step 3
        step_results['final_reply'] = step_results['step_3']

    except Exception as e:
        print("Gemini error after retries:", e)
        step_results['final_reply'] = "Sorry, there was an error talking to Gemini. Please try again later."

    return step_results

# ------------------------
# MARA Steps with retry + debug prints
# ------------------------
def step_one(user_query):
    step_1_query = compose_step_one_query(user_query)
    print("Step 1: Finding relevant studies...")
    step_1_response = call_gemini_with_retry(step_1_query)
    print("Step 1 result (first 500 chars):", step_1_response[:500], "...")
    return step_1_response

def compose_step_one_query(user_query):
    return (
        common_persona_prompt
        + "Find me high-quality studies that look into the question of: "
        + user_query
        + "\nPlease optimize your search per the following constraints: "
        + "\n1. Search online databases that index published literature, as well as sources such as Google Scholar."
        + "\n2. Find studies per retrospective reference harvesting and prospective forward citation searching."
        + "\n3. Attempt to identify unpublished literature such as dissertations and reports from independent research firms."
        + "\nExclude any studies which either:"
        + "\n1. lack a comparison or control group."
        + "\n2. are purely correlational, that do not include either a randomized-controlled trial, quasi-experimental design, or regression discontinuity"
        + "\nFinally, return these studies in a list of highest quality to lowest, formatting that list by: 'Title, Authors, Date Published.' "
    )

def step_two(step_1_result):
    step_2_query = compose_step_two_query(step_1_result)
    print("Step 2: Extracting study data...")
    step_2_response = call_gemini_with_retry(step_2_query)
    print("Step 2 result (first 500 chars):", step_2_response[:500], "...")
    return step_2_response

def compose_step_two_query(step_1_result):
    return (
        common_persona_prompt
        + "First, Lookup the papers for each of the studies in this list."
        + "\n"
        + step_1_result
        + "\n Then, extract the following data to compile into a spreadsheet."
        + "\nSpecifically, organize the data for each study into the following columns: "
        + "\n1. Sample size of treatment and comparison groups"
        + "\n2. Cluster sample sizes (i.e. size of the classroom or school of however the individuals are clustered)"
        + "\n3. Intraclass correlation coefficient (ICC; when available) will be coded for cluster studies. When the ICC estimates are not provided, impute a constant value of 0.20."
        + "\n4. Effect size for each outcome analysis will be calculated and recorded. These should be the standardized mean difference between the treatment and control group at post-test, ideally adjusted for pre-test differences."
        + " Authors can report this in varying ways. The preference is for adjusted effects, found in a linear regression. If adjusted effects are unavailable, raw means and standard deviations can be used."
        + "\n6. Study design (i.e., randomized controlled trial, quasi-experimental, or regression discontinuity)"
        + "\nReturn the results in a spreadsheet, where each row is for each study and each column is for each column feature in the above list."
    )

def step_three(step_2_result):
    step_3_query = compose_step_three_query(step_2_result)
    print("Step 3: Analyzing study data...")
    step_3_response = call_gemini_with_retry(step_3_query)
    print("Step 3 result (first 500 chars):", step_3_response[:500], "...")
    return step_3_response

def compose_step_three_query(step_2_result):
    return (
        common_persona_prompt
        + "\nUsing this dataset: "
        + step_2_result
        + "\ncreate a simple model with only the impact of the main predictor of interest. Specifically, use a multivariate meta-regression model to conduct the meta-analysis."
    )