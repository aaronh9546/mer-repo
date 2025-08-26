from fastapi import FastAPI, HTTPException
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


# 🟢 Helper: safely extract Gemini text
def extract_text(response):
    """Safely extract text from a Gemini response object."""
    if not response:
        return None
    # Try direct .text
    if hasattr(response, "text") and response.text:
        return response.text
    # Fallback: dig into candidates
    try:
        return response.candidates[0].content.parts[0].text
    except (AttributeError, IndexError, KeyError):
        return None


# --- API endpoint ---
@app.post("/chat")
def chat_api(query: Query):
    user_query = query.message
    if not user_query:
        raise HTTPException(status_code=400, detail="Query message is required.")

    print("Starting an investigation into:", user_query)

    try:
        step_1_result = step_one(user_query)
        step_2_result = step_two(step_1_result)
        step_3_result = step_three(step_2_result)
        reply = step_3_result or "No reply generated."
    except Exception as e:
        print("Gemini error:", e)
        reply = f"Sorry, there was an error talking to Gemini: {e}"

    print("Goodbye from MARA!")
    return {"reply": reply}


# ------------------------
# MARA Steps
# ------------------------
def step_one(user_query: str) -> str:
    if not user_query:
        raise ValueError("Step 1: user_query is empty.")
    step_1_query = compose_step_one_query(user_query)
    step_1_response = client.models.generate_content(
        model=gemini_model,
        contents=step_1_query,
    )
    text = extract_text(step_1_response)
    if not text:
        raise ValueError(f"Step 1: No response from Gemini. Raw: {step_1_response}")
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
    )


def step_two(step_1_result: str) -> str:
    if not step_1_result:
        raise ValueError("Step 2: step_1_result is empty.")
    step_2_query = compose_step_two_query(step_1_result)
    step_2_response = client.models.generate_content(
        model=gemini_model,
        contents=step_2_query,
    )
    text = extract_text(step_2_response)
    if not text:
        raise ValueError(f"Step 2: No response from Gemini. Raw: {step_2_response}")
    return text


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
    )


def step_three(step_2_result: str) -> str:
    if not step_2_result:
        raise ValueError("Step 3: step_2_result is empty.")
    step_3_query = compose_step_three_query(step_2_result)
    step_3_response = client.models.generate_content(
        model=gemini_model,
        contents=step_3_query,
    )
    text = extract_text(step_3_response)
    if not text:
        raise ValueError(f"Step 3: No response from Gemini. Raw: {step_3_response}")
    return text


def compose_step_three_query(step_2_result: str) -> str:
    return (
        common_persona_prompt
        + "\nUsing this dataset: "
        + step_2_result
        + "\ncreate a simple model with only the impact of the main predictor of interest. Specifically, use a multivariate meta-regression model to conduct the meta-analysis."
    )