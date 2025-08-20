from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from mara_secrets import GEMINI_API_KEY

"""
in: question on relationship (model) of several variables. A proposed model from these variables would constitute a hypothesis.
out: analysis on that relationship, which would prove/disprove the veracity of a given hypothesis
"""

client = genai.Client(api_key=GEMINI_API_KEY)
gemini_model = "gemini-2.5-pro"
common_persona_prompt = "You are a senior data analyst with a specialty in meta-analysis."

app = FastAPI()

# Update this to your WordPress domain
origins = [
    "https://aaronhanto-nyozw.wpcomstaging.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat_api(query: Query):
    user_query = query.message
    print("Starting an investigation into: " + user_query)

    step_1_result = step_one(user_query)
    step_2_result = step_two(step_1_result)
    step_3_result = step_three(step_2_result)

    print("Goodbye from MARA!")
    return {"reply": step_3_result}


# ------------------------
# MARA Steps (keep all your comments)
# ------------------------
def step_one(user_query):
    step_1_query = compose_step_one_query(user_query)
    step_1_response = client.models.generate_content(
        model=gemini_model,
        contents=step_1_query,
    )
    return step_1_response.text

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
    step_2_response = client.models.generate_content(
        model=gemini_model,
        contents=step_2_query,
    )
    return step_2_response.text

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
    step_3_response = client.models.generate_content(
        model=gemini_model,
        contents=step_3_query,
    )
    return step_3_response.text

def compose_step_three_query(step_2_result):
    return (
        common_persona_prompt
        + "\nUsing this dataset: "
        + step_2_result
        + "\ncreate a simple model with only the impact of the main predictor of interest. Specifically, use a multivariate meta-regression model to conduct the meta-analysis."
    )