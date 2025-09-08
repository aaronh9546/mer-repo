from google import genai


from mara_secrets import GEMINI_API_KEY
from pydantic import BaseModel
import enum


"""
in: question on relationship (model) of several variables. A proposed model from these variables would constitute a hypothesis.
out: analysis on that relationship, which would prove/disprove the veracity of a given hypothesis
"""


# eg: What is the relationship between free school lunches and test scores?


client = genai.Client(api_key=GEMINI_API_KEY)
gemini_model = "gemini-2.5-pro" # "gemini-2.0-flash"  #   # "gemini-2.5-flash"  # "gemini-2.5-pro" # "gemini-2.0-flash"
common_persona_prompt = (
   "You are a senior data analyst with a specialty in meta-analysis."
)
# https://ai.google.dev/gemini-api/docs/models
# Note: Gemini 2.5 Pro and 2.5 Flash come with thinking on by default.
# https://ai.google.dev/gemini-api/docs/thinking
# https://ai.google.dev/gemini-api/docs/document-processing
# https://ai.google.dev/gemini-api/docs/function-calling?example=meeting
# https://ai.google.dev/gemini-api/docs/text-generation




def main():
   # step 0: user question, on relationship of several variables. Corresponds to hypothesis, proposing relationship or lack thereof.
   hello = (
       "Hello!"
       + "\nI’m here to help answer your questions about what works in education with strong evidence and systematic meta-analytic methods. "
       + "\nIf there isn’t enough strong evidence to answer your question, or conflicting evidence, I will let you know."
       + "\nHere are some examples of questions I can answer: "
       + "\n - What is the overall effect of a specific teaching method (e.g., direct instruction, inquiry-based learning, project-based learning) on student achievement?"
       + "\n - What is the impact of homework on academic performance by grade level?"
       + "\n - Do educational technology tools (e.g., adaptive learning software, gamification, online learning) improve student achievement?"
       + "\n - What is the effectiveness of blended learning compared to fully face-to-face instruction?"
       + "\n - What is the effect of reduced class sizes on student achievement?"
       + "\n - What is the effect of school start times on academic performance and attendance?"
       + "\n - Do charter schools perform better than traditional public schools?"
       + "\n[enter your query]: "
   )
   user_query = input(
       hello
       # "Enter your question. This question should ask about the relationship of variables, such that a model capturing that relationship would constitute a hypothesis.\n [enter your question]: "
   )


   print("Starting an investigation into: " + user_query)


   # step 1: compile list of research / studies from which analysis will be drawn
   # step 1.5: limit to higher-quality research, as determined per research features
   step_1_result = get_studies(user_query)
   # step 2: extract underlying data of that research, preserving which research corresponds to what data
   step_2_result = extract_studies_data(step_1_result)
   # step 3: perform novel & independent analysis on that underlying data, to yield a model that captures the relationship of those variables, per the given initial features.
   step_3_result = analyze_studies(step_2_result)


   followup(step_3_result, step_2_result)


   print("Goodbye from MARA!")




def get_studies(user_query):
   """
   # step 1: compile list of research / studies from which analysis will be drawn
   # step 1.5: limit to higher-quality research, as determined per research features
   """
   step_1_query = compose_step_one_query(user_query)


   print("Finding relevant studies...")
   step_1_response = client.models.generate_content(
       model=gemini_model,
       contents=step_1_query,
   )
   print("Found relevant studies. ")
   print(step_1_response.text)
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
       + "\nInclude at least 30 studies, or if fewer than 30 the max available."
       + "\nKeep your response brief, only including that raw list and nothing more."
   )




def extract_studies_data(step_1_result):
   """
   # step 2: extract underlying data of that research, preserving which research corresponds to what data
   """
   step_2_query = compose_step_two_query(step_1_result)


   print("Extracting study data...")
   step_2_response = client.models.generate_content(
       model=gemini_model,
       contents=step_2_query,
   )
   print("Extracted study data.")
   print(step_2_response.text)




   return step_2_response.text




def compose_step_two_query(step_1_result):
   return (
       common_persona_prompt
       + "First, lookup the papers for each of the studies in this list."
       + "\n"
       + step_1_result
       + "\nThen, extract the following data to compile into a spreadsheet."
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




class Confidence(enum.Enum):
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




def analyze_studies(step_2_result):
   """
   # step 3: perform novel & independent analysis on that underlying data, to yield a model that captures the relationship of those variables, per the given initial features.
   """
   step_3_query = compose_step_three_query(step_2_result)


   print("Analyzing study data...")
   step_3_response = client.models.generate_content(
       model=gemini_model,
       contents=step_3_query,
       config={
           "response_mime_type": "application/json",
           "response_schema": AnalysisResponse,
       },
   )
   parsed_step_3_response : AnalysisResponse = step_3_response.parsed # as AnalysisResponse
   print("Analyzed study data.")
   print("Analysis overview: " + parsed_step_3_response.summary)
   print("Analysis confidence: " + str(parsed_step_3_response.confidence))


   # print("step_3_response.text: ")
   # print(step_3_response.text)
   return step_3_response.text




def compose_step_three_query(step_2_result):
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




def followup(step_3_result, step_2_result):
   user_has_followup = True


   while user_has_followup:
       user_followup_input = input(
           "Do you have additional followup questions?"
           + "\nFor example, whether you'd like to know more about the meta-regression models used, the process by which this analysis was done, the number of studies included, etc, please just ask."
           +"\n[Enter any followup question here, or decline]: "
       )
       user_has_followup = client.models.generate_content(
           model=gemini_model,
           contents="Return true if the following text either indicates a 'yes' or asks a question making a request for analysis: "
           + user_followup_input,
           config={
               "response_mime_type": "application/json",
               "response_schema": bool,
           },
       )
       if not user_has_followup.parsed:
           return


       print("Looking into that followup...")
       followup_response = client.models.generate_content(
           model=gemini_model,
           contents="Answer this question: "
           + user_followup_input
           + ". Use the both the analysis here: "
           + "\n1. "
           + step_3_result
           + "and the data here: "
           + "\n2. "
           + step_2_result
       )


       print(followup_response.text)




if __name__ == "__main__":
   main()