import os
import enum
import json
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, List
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import redis

from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
from pydantic import BaseModel, ValidationError
from jose import JWTError, jwt
import google.generativeai as genai

# --- Sentry Initialization ---
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"), 
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0,
)

# --- Pydantic Models (Data Schemas) ---

class User(BaseModel):
    id: int
    email: str
    name: str | None = None

class Query(BaseModel):
    message: str

class FollowupQuery(BaseModel):
    conversation_id: str
    message: str

class Confidence(enum.Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    
    @staticmethod
    def get_description():
        return (
            "HIGH - If the research on the topic has a well-conducted, randomized study showing a statistically significant positive effect on at least one outcome measure (e.g., state test or national standardized test) analyzed at the proper level of clustering (class/school or student) with a multi-site sample of at least 350 participants. Strong evidence from at least one well-designed and wellimplemented experimental study."
            + "\nMODERATE - If it meets all standards for “HIGH” stated above, except that instead of using a randomized design, qualifying studies are prospective quasi-experiments (i.e., matched studies). Quasiexperimental studies (e.g., Regression Discontinuity Design) are those in which students have not been randomly assigned to treatment or control groups, but researchers are using statistical matching methods that allow them to speak with confidence about the likelihood that an intervention causes an outcome."
            + "\nLOW - The topic has a study that would have qualified for “HIGH” or “MODERATE” but did not because it failed to account for clustering (but did obtain significantly positive outcomes at the student level) or did not meet the sample size requirements. Post-hoc or retrospective studies may also qualify."
        )

class AnalysisDetails(BaseModel):
    regression_models: Any | None = None
    process: str
    plots: Any | None = None

class AnalysisResponse(BaseModel):
    summary: str
    confidence: Confidence
    details: AnalysisDetails

# --- Application Setup ---

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=[
    "https://myeducationresearcher.com",
    "https://timothy-han.com",
    "https://jsdean1517-pdkfw.wpcomstaging.com",
    "http://localhost:3000",
])

# --- Security and Authentication Configuration ---
INTERNAL_SECRET_KEY = os.getenv("INTERNAL_SECRET_KEY", "YOUR_SUPER_SECRET_PRE_SHARED_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a_different_strong_secret_for_jwt")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

# --- Redis and Rate Limiter Setup ---
REDIS_URL = os.getenv("RATELIMIT_STORAGE_URI", "redis://localhost:6379")

def get_user_id_from_context():
    """Get the user ID from the Flask global `g` object after authentication."""
    try:
        return g.current_user.id
    except AttributeError:
        return get_remote_address

limiter = Limiter(
    get_user_id_from_context,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=REDIS_URL,
)

gemini_model = "gemini-2.5-pro"
common_persona_prompt = "You are a senior data analyst with a specialty in meta-analysis."

def initialize_client():
    """Helper function to configure and return the GenAI client."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("FATAL: GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ GenAI Client configured and initialized successfully.")
    return genai.GenerativeModel(gemini_model)

client = initialize_client()
redis_client = redis.from_url(REDIS_URL)

# --- Authentication Logic (Flask Decorators) ---

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            try:
                token = request.headers['Authorization'].split(" ")[1]
            except IndexError:
                return jsonify({"message": "Bearer token malformed"}), 401
        if not token:
            return jsonify({"message": "Token is missing!"}), 401
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
            user_data = { "id": int(payload.get("sub")), "email": payload.get("email"), "name": payload.get("name") }
            user = User.model_validate(user_data)
            g.current_user = user
        except (JWTError, ValueError, TypeError, ValidationError):
            return jsonify({"message": "Token is invalid or expired"}), 401
        return f(*args, **kwargs)
    return decorated

def internal_secret_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        internal_secret = request.headers.get("X-Internal-Secret")
        if not internal_secret or internal_secret != INTERNAL_SECRET_KEY:
            return jsonify({"message": "Invalid secret key for internal communication"}), 403
        return f(*args, **kwargs)
    return decorated

# --- Helper Function for Streaming ---
def stream_event(data: dict) -> str:
    """Robustly formats a dictionary into a Server-Sent Event string."""
    return f"data: {json.dumps(data)}\n\n"

# --- API Endpoints ---

@app.route("/auth/issue-wordpress-token", methods=['POST'])
@internal_secret_required
def issue_wordpress_token():
    try:
        user_data = request.json
        token_data_for_jwt = { "sub": str(user_data.get("id")), "email": user_data.get("email"), "name": user_data.get("name") }
        print(f"Issuing token for WordPress user: {token_data_for_jwt['email']} (ID: {token_data_for_jwt['sub']})")
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data=token_data_for_jwt, expires_delta=access_token_expires)
        return jsonify({"access_token": access_token, "token_type": "bearer"})
    except (ValidationError, TypeError, AttributeError):
        return jsonify({"message": "Invalid user data"}), 400

@app.route("/chat", methods=['POST'])
@token_required
@limiter.limit("1 per 5 minutes")
def chat_api():
    current_user = g.current_user
    print(f"Authenticated request from user: {current_user.email}")
    
    try:
        query = Query.model_validate(request.json)
        user_query = query.message
    except (ValidationError, TypeError):
        return jsonify({"message": "Invalid request body"}), 400

    def event_generator():
        try:
            conversation_id = str(uuid.uuid4())

            yield stream_event({'type': 'update', 'content': 'Finding relevant studies...'})
            step_1_result = get_studies(user_query)
            step_1_result_id = f"{conversation_id}:step1"
            redis_client.set(f"result:{step_1_result_id}", step_1_result, ex=600)
            yield stream_event({'type': 'fetch_result', 'url': f'/results/{step_1_result_id}'})

            yield stream_event({'type': 'update', 'content': 'Extracting study data...'})
            step_2_result = extract_studies_data(step_1_result)
            step_2_result_id = f"{conversation_id}:step2"
            redis_client.set(f"result:{step_2_result_id}", step_2_result, ex=600)
            yield stream_event({'type': 'fetch_result', 'url': f'/results/{step_2_result_id}'})
            
            yield stream_event({'type': 'update', 'content': 'Compacting data for analysis...'})
            step_2_5_compact_data = summarize_data_for_analysis(step_2_result)
            
            yield stream_event({'type': 'update', 'content': 'Analyzing study data...'})
            analysis_result = analyze_studies(step_2_5_compact_data)
            
            analysis_dict = analysis_result.model_dump(mode='json')
            
            session_data_to_store = {
                "user_id": current_user.id,
                "original_query": user_query,
                "studies_data": step_2_result,
                "analysis_data_str": json.dumps(analysis_dict)
            }
            redis_client.set(f"session:{conversation_id}", json.dumps(session_data_to_store), ex=3600)

            result_data = {"type": "result", "content": analysis_dict}
            yield stream_event(result_data)
            
            yield stream_event({'type': 'conversation_id', 'content': conversation_id})
            
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"An error occurred in the stream: {e}")
            yield stream_event({"type": "error", "content": f"An error occurred: {str(e)}"})

    return Response(event_generator(), mimetype='text-event-stream')

@app.route("/results/<path:result_id>")
@token_required
def get_result(result_id):
    try:
        safe_result_id = f"result:{result_id.replace('..', '')}"
        result_data = redis_client.get(safe_result_id)
        if result_data:
            return Response(result_data, mimetype='text/plain')
        else:
            return "Result not found or expired.", 404
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return "An error occurred while fetching the result.", 500

@app.route("/followup", methods=['POST'])
@token_required
@limiter.limit("15 per hour")
def followup_api():
    current_user = g.current_user
    print(f"Follow-up request from user: {current_user.email}")
    
    try:
        data = request.json
        conversation_id = data.get("conversation_id")
        user_message = data.get("message")
        if not conversation_id or not user_message:
            raise ValueError("Missing conversation_id or message")
    except (ValidationError, TypeError, ValueError):
        return jsonify({"message": "Invalid request body"}), 400

    session_json = redis_client.get(f"session:{conversation_id}")
    if not session_json:
        def error_generator():
            yield stream_event({'type': 'error', 'content': 'Conversation not found or has expired.'})
        return Response(error_generator(), mimetype='text-event-stream')
    
    session_data = json.loads(session_json)

    if session_data.get("user_id") != current_user.id:
        def error_generator():
            yield stream_event({'type': 'error', 'content': 'Access denied to this conversation.'})
        return Response(error_generator(), mimetype='text-event-stream')

    def event_generator():
        try:
            followup_prompt = compose_followup_query(session_data, user_message)
            
            input_tokens = client.count_tokens(followup_prompt)
            print(f"🪙 Followup Input Tokens: {input_tokens.total_tokens}")
            
            response_stream = client.generate_content(followup_prompt, stream=True)
            
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    yield stream_event({'type': 'message', 'content': chunk.text})
            
            output_tokens = client.count_tokens(full_response)
            print(f"🪙 Followup Output Tokens: {output_tokens.total_tokens}")

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"An error occurred in the followup stream: {e}")
            yield stream_event({"type": "error", "content": f"An error occurred: {str(e)}"})

    return Response(event_generator(), mimetype='text-event-stream')

# --- MARA Logic (Synchronous Versions) ---

def get_studies(user_query: str) -> str:
    step_1_query = compose_step_one_query(user_query)
    
    input_tokens = client.count_tokens(step_1_query)
    print(f"🪙 Step 1 Input Tokens: {input_tokens.total_tokens}")

    response = client.generate_content(step_1_query, request_options={"timeout": 300})
    
    output_tokens = client.count_tokens(response.text)
    print(f"🪙 Step 1 Output Tokens: {output_tokens.total_tokens}")
    
    cleaned_text = response.text.replace('"', "'")
    return cleaned_text

def extract_studies_data(step_1_result: str) -> str:
    step_2_query = compose_step_two_query(step_1_result)
    
    input_tokens = client.count_tokens(step_2_query)
    print(f"🪙 Step 2 Input Tokens: {input_tokens.total_tokens}")

    response = client.generate_content(step_2_query, request_options={"timeout": 300})
    
    output_tokens = client.count_tokens(response.text)
    print(f"🪙 Step 2 Output Tokens: {output_tokens.total_tokens}")
    
    cleaned_text = response.text.replace('\n', ' ').replace('\r', ' ')
    return cleaned_text

def summarize_data_for_analysis(step_2_markdown: str) -> str:
    print("--- Step 2.5: Summarizing data for analysis ---")
    
    summarization_prompt = compose_step_two_point_five_query(step_2_markdown)
    
    input_tokens = client.count_tokens(summarization_prompt)
    print(f"🪙 Step 2.5 Input Tokens: {input_tokens.total_tokens}")
    
    response = client.generate_content(summarization_prompt, request_options={"timeout": 300})

    output_tokens = client.count_tokens(response.text)
    print(f"🪙 Step 2.5 Output Tokens: {output_tokens.total_tokens}")
    
    print("✅ Data summarization complete.")
    cleaned_response = response.text.replace('\n', ' ').replace('\r', ' ').replace('"', "'")
    return cleaned_response

def analyze_studies(step_2_5_compact_data: str, max_retries: int = 1) -> AnalysisResponse:
    step_3_query = compose_step_three_query(step_2_5_compact_data)
    
    input_tokens = client.count_tokens(step_3_query)
    print(f"🪙 Step 3 Input Tokens: {input_tokens.total_tokens}")
    
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json"
    )
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            print(f"--- Step 3: Analysis, Attempt {attempt + 1}/{max_retries + 1} ---")
            response = client.generate_content(
                step_3_query,
                generation_config=generation_config,
                request_options={"timeout": 300}
            )
            
            cleaned_json_string = response.text.replace('\n', ' ').replace('\r', ' ')

            output_tokens = client.count_tokens(cleaned_json_string)
            print(f"🪙 Step 3 Output Tokens: {output_tokens.total_tokens}")

            response_json = json.loads(cleaned_json_string)
            return AnalysisResponse.model_validate(response_json)
        except Exception as e:
            print(f"🔴 Attempt {attempt + 1} failed. Error: {e}")
            last_error = e
            if attempt < max_retries:
                print("Retrying...")
            else:
                raise ValueError(f"Step 3 failed after {max_retries + 1} attempts. Last error: {last_error}")

# --- Prompt Composition Functions ---

def compose_followup_query(session_data: dict, new_message: str) -> str:
    step_3_result = session_data.get('analysis_data_str', '{}')
    step_2_result = session_data.get('studies_data', 'No data available.')

    return (
        "Answer this question: "
        + new_message
        + ". Use both the analysis here: "
        + "\n1. "
        + step_3_result
        + " and the data here: "
        + "\n2. "
        + step_2_result
    )

def compose_step_one_query(user_query: str) -> str:
    return (
        common_persona_prompt
        + " Find me high-quality studies that look into the question of: " + user_query
        + "\nOptimize your search per the following constraints: "
        + "\n1. Search online databases that index published literature, as well as sources such as Google Scholar."
        + "\n2. Find studies per retrospective reference harvesting and prospective forward citation searching."
        + "\n3. Attempt to identify unpublished literature such as dissertations and reports from independent research firms."
        + "\nExclude any studies which either:"
        + "\n1. lack a comparison or control group,"
        + "\n2. are purely correlational, that do not include either a randomized-controlled trial, quasi-experimental design, or regression discontinuity"
        + "\nFinally, return these studies in a list of highest quality to lowest, formatting that list by: 'Title, Authors, Date Published.' "
        + "\nInclude 30 high-quality studies, or if fewer than 30, the max available." 
        + "\nDo not add any explanatory text."
    )

def compose_step_two_query(step_1_result: str) -> str:
    return (
        common_persona_prompt
        + " You have been provided with a definitive list of studies below. **Do not search for any other studies or add any studies not on this exact list.**"
        + " For **only** the studies in this list, look up each paper:\n" + step_1_result
        + "\nThen, extract the following data into a markdown table format: "
        + "\n1. Sample size of treatment and comparison groups"
        + "\n2. Cluster sample sizes (i.e. size of classroom/school)"
        + "\n3. Intraclass correlation coefficient (ICC). If not provided, impute 0.20."
        + "\n4. Hedges' g effect size for each outcome (standardized mean difference, adjusted for pre-test if possible)."
        + "\n5. Study design (RCT, quasi-experimental, or RDD)."
        + "\nReturn only the markdown table and nothing else. **Ensure there is one entry per study from the provided list and no duplicates.**"
    )

def compose_step_two_point_five_query(step_2_markdown: str) -> str:
    return (
        "You are an expert data processing agent. You have been given a markdown table containing data about academic studies. "
        "Your task is to convert this table into a compact, machine-readable CSV (Comma-Separated Values) format. "
        "Do not lose any information. Ensure the header row is simple and all subsequent rows contain the corresponding data points. "
        "Return only the raw CSV data and nothing else.\n\n"
        "Here is the markdown table:\n"
        f"{step_2_markdown}"
    )

def compose_step_three_query(step_2_result: str) -> str:
    json_structure_example = """
{
  "summary": "A one or two sentence summary of the analysis conclusion.",
  "confidence": "HIGH",
  "details": {
    "process": "A description of the meta-analysis process used.",
    "regression_models": "The specific meta-regression models produced, including coefficients and statistics.",
    "plots": "A textual description of relevant plots, such as a forest plot or funnel plot."
  }
}
"""
    return (
        common_persona_prompt
        + "\nUsing this CSV dataset of academic studies: \n" + step_2_result
        + "\nPerform a meta-analysis using a multivariate meta-regression model and return the results as a valid JSON object."
        + "\n\n**CRITICAL REQUIREMENT:** Your response MUST be a single, valid JSON object that strictly adheres to the following structure and schema. Do not include any text, markdown formatting, or explanations outside of the JSON object itself."
        + f"\n\nHere is an example of the required JSON structure:\n```json\n{json_structure_example}\n```"
        + "\n\nNow, populate this exact JSON structure based on your analysis:"
        + "\n1. For the `summary` field: Write a one or two sentence summary of your conclusion."
        + "\n2. For the `confidence` field: Determine the confidence level (HIGH, MODERATE, or LOW) based on these criteria: " + Confidence.get_description()
        + "\n3. For the nested `details.process` field: Describe the analysis process you used."
        + "\n4. For the nested `details.regression_models` field: Show the regression models produced."
#        + "\n5. For the nested `details.plots` field: Describe any corresponding plots."
    )

if __name__ == "__main__":
    app.run(debug=True, port=8005)