import os
import enum
import json
import google.generativeai as genai
from pydantic import BaseModel
from celery_worker import celery_app

# --- AI Client Initialization for the Worker ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
client = genai.GenerativeModel("gemini-1.5-pro-latest") 
common_persona_prompt = "You are a senior data analyst with a specialty in meta-analysis."


# --- Pydantic Schemas ---
class Confidence(enum.Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    
    @staticmethod
    def get_description():
        # REVERTED to your original, full description
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


# --- Synchronous Helper Functions for the Celery Task ---
def get_studies_sync(user_query: str) -> str:
    step_1_query = compose_step_one_query(user_query)
    response = client.generate_content(step_1_query)
    print(f"WORKER LOG: Step 1 Raw Response: {response.text}")
    return response.text

def extract_studies_data_sync(step_1_result: str) -> str:
    step_2_query = compose_step_two_query(step_1_result)
    response = client.generate_content(step_2_query)
    print(f"WORKER LOG: Step 2 Raw Response: {response.text}")
    return response.text

def analyze_studies_sync(step_2_result: str) -> AnalysisResponse:
    step_3_query = compose_step_three_query(step_2_result)
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=AnalysisResponse,
    )
    response = client.generate_content(step_3_query, generation_config=generation_config)
    print(f"WORKER LOG: Step 3 Raw Response: {response.text}")
    return AnalysisResponse.model_validate_json(response.text)


# --- REVERTED PROMPT FUNCTIONS ---
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
        + "\nKeep your response brief, only including that raw list and nothing more."
    )

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
        + " Authors can report this in varying ways. The preference is for adjusted effects, found in a linear regression. If adjusted effects are unavailable, raw means and standard deviations can be used."
        + "\n6. Study design (i.e., randomized controlled trial, quasi-experimental, or regression discontinuity)"
        + "\nReturn the results in a spreadsheet, where each row is for each study and each column is for each column feature in the above list."
        + "\nKeep your response brief, only including those spreadsheet rows and nothing more."
    )

def compose_step_three_query(step_2_result: str) -> str:
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


# --- THE CELERY TASK ---
@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def run_full_analysis(self, user_query: str):
    try:
        print(f"WORKER: Starting task {self.request.id}")
        
        # Step 1
        self.update_state(state='PROGRESS', meta={'status': 'Finding relevant studies...'})
        step_1_result = get_studies_sync(user_query)
        
        # Step 2
        self.update_state(state='PROGRESS', meta={'status': 'Extracting study data...'})
        step_2_result = extract_studies_data_sync(step_1_result)
        
        # Step 3
        self.update_state(state='PROGRESS', meta={'status': 'Analyzing study data...'})
        analysis_result = analyze_studies_sync(step_2_result)
        
        print(f"WORKER: Finished task {self.request.id}")
        return {
            "step_1_result": step_1_result,
            "step_2_result": step_2_result,
            "analysis_result": analysis_result.model_dump()
        }
    except Exception as exc:
        print(f"WORKER: Task {self.request.id} failed. Error: {exc}")
        raise