import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import math
import requests
import random
from datetime import date as dt_date
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# --- Imports for Model Loading ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor

# --- Page Configuration ---
st.set_page_config(
    page_title="Crop Sense",
    page_icon="🌾",
    layout="wide"
)

# --- Load API Keys ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Keras Model Definition (Required for loading the model) ---
def create_model(n_features_in, optimizer='adam', dropout_rate=0.2, hidden_layer_sizes=(64, 32)):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape=(n_features_in,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_layer_sizes[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    return model

# --- Caching Functions to Improve Performance ---
@st.cache_resource
def load_model(path):
    """Loads the trained model pipeline."""
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading the model from {path}. Details: {e}")
        return None

@st.cache_data
def load_data(path):
    """Loads and preprocesses the base data."""
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        for col in ['Crop', 'State', 'Season']:
            if col in df.columns:
                df[col] = df[col].str.strip()
        df['Yield'] = df['Yield'].fillna(df['Production'] / df['Area'].replace(0, np.nan))
        df = df.dropna(subset=['Yield', 'Crop', 'State', 'Season'])
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at {path}.")
        return None

# --- Helper Functions ---
STATE_COORDINATES = {
    "Odisha": (20.9517, 85.0985)
}

@st.cache_data
def fetch_rainfall_from_api(state: str, year: int):
    if state not in STATE_COORDINATES:
        st.sidebar.error(f"Coordinates for '{state}' not found.")
        return np.nan
    lat, lon = STATE_COORDINATES[state]
    start_date_str = f"{year}-01-01"
    end_date_str = f"{year}-12-31"
    current_year = dt_date.today().year
    if year == current_year:
        end_date_str = dt_date.today().isoformat()
    elif year > current_year:
        st.sidebar.error("Cannot fetch rainfall data for a future year.")
        return np.nan
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date_str}&end_date={end_date_str}"
        f"&daily=precipitation_sum&timezone=Asia/Kolkata"
    )
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if "daily" in data and "precipitation_sum" in data["daily"]:
            daily_rainfall = data["daily"]["precipitation_sum"]
            total_rainfall = sum(precip for precip in daily_rainfall if precip is not None)
            return total_rainfall
        else:
            st.sidebar.warning(f"Rainfall data not available for {state}, {year}.")
            return np.nan
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Network error fetching data: {e}")
        return np.nan

@st.cache_data
def get_llm_suggestions(api_key: str, predicted_yield: float, farmer_profile: dict):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are an expert agricultural advisor based in India. Your task is to provide concise, actionable advice to a farmer.
        **FARMER'S DATA:**
        - Crop: {farmer_profile.get('Crop', 'N/A')}
        - State: {farmer_profile.get('State', 'N/A')}
        - Season: {farmer_profile.get('Season', 'N/A')}
        - Annual Rainfall: {farmer_profile.get('Annual_Rainfall', 0.0):.2f} mm
        **MODEL PREDICTION:**
        - Predicted Yield: {predicted_yield:.2f} tons/hectare
        **INSTRUCTIONS:**
        Provide 2-3 clear, simple, and actionable suggestions in a bulleted list. Keep the tone helpful and easy to understand.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating AI suggestions: {e}"

@st.cache_data
def translate_text_gemini(text_to_translate: str, target_language: str, api_key: str):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Translate the following English text to {target_language}. Provide only the direct translation:\n\n---\n\n{text_to_translate}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during translation: {e}"

def save_guest_data(profile_data):
    try:
        profile_data['timestamp'] = datetime.now().isoformat()
        guest_df = pd.DataFrame([profile_data])
        file_exists = os.path.isfile("guest_predictions.csv")
        guest_df.to_csv("guest_predictions.csv", mode='a', header=not file_exists, index=False)
        st.toast("This prediction data has been saved anonymously.", icon="📝")
    except Exception as e:
        st.warning(f"Could not save guest data: {e}")

# --- Functions for Page Rendering ---

def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- NEW: Function for animated login page background ---
def set_login_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://i.pinimg.com/originals/b7/65/9d/b7659d95086e4a29a3416e7886a113b2.gif");
            background-size: cover;
            background-position: center;
        }}
        /* Style for the semi-transparent container */
        div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 2rem;
            border-radius: 10px;
            color: white;
        }}
        /* Make text white for readability */
        div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] h1,
        div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] p,
        div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] label {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def navigate_to_page(page_name):
    st.session_state.page = page_name

def render_login_page():
    """Renders the login page with an animated background."""
    set_login_bg()
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        # Using st.container() to group elements for styling
        with st.container():
            st.title("🌾 Welcome to Crop Sense")
            st.markdown("Please log in or continue as a guest to get started.")

            with st.form("login_form"):
                st.markdown("Enter Phone Number")
                c1, c2 = st.columns([0.3, 0.7])
                with c1:
                    st.text_input("Country Code", value="+91", disabled=True, label_visibility="collapsed")
                with c2:
                    st.text_input(
                        "Mobile Number",
                        placeholder="9876543210",
                        key="phone_number",
                        max_chars=10,
                        label_visibility="collapsed"
                    )

                send_otp_button = st.form_submit_button("Send OTP")
                if send_otp_button:
                    otp = random.randint(100000, 999999)
                    st.session_state.correct_otp = otp
                    st.session_state.otp_sent = True
                    st.success(f"📬 Mock OTP Sent! Your OTP is: **{otp}**")
                
                if st.session_state.get('otp_sent', False):
                    user_otp = st.text_input("Enter OTP", placeholder="Enter 6-digit OTP")
                    login_button = st.form_submit_button("Login")
                    if login_button:
                        if user_otp == str(st.session_state.get('correct_otp', '')):
                            st.session_state.logged_in = True
                            st.session_state.page = "cover"
                            st.rerun()
                        else:
                            st.error("Invalid OTP. Please try again.")
            
            st.divider()
            
            if st.button("Continue without Login", use_container_width=True):
                st.session_state.logged_in = False
                navigate_to_page("cover")
                st.rerun()

def render_cover_page():
    image_url = "https://images.unsplash.com/photo-1560493676-04071c5f467b?q=80&w=1974&auto=format&fit=crop"
    set_bg_image(image_url)
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="background-color: rgba(0, 0, 0, 0.6); padding: 30px; border-radius: 10px; text-align: center; color: white;">
                <h1 style="font-size: 3rem; font-weight: bold;">Yield Prediction & Optimisation</h1>
                <p style="font-size: 1.25rem;">Data-driven insights for modern Indian farming</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.button(
            "Let's Predict",
            on_click=navigate_to_page,
            args=("main_app",),
            use_container_width=True,
            type="primary"
        )

def render_main_app(df, model_pipeline):
    st.title("🌾 Crop Yield Prediction and Advisory System")
    st.markdown("Enter the details of your farm to get a yield prediction and tailored recommendations.\nଅମଳ ପୂର୍ବାନୁମାନ ଏବ... (rest of your function remains the same)")
    
    # --- ALL THE CODE FROM YOUR render_main_app FUNCTION GOES HERE ---
    # --- I've omitted it for brevity, but you should paste your original code back in ---
    crop_options = sorted(df['Crop'].unique())
    state_options = sorted(df['State'].unique())
    season_options = sorted(df['Season'].unique())
    features = model_pipeline.named_steps['preprocessor'].feature_names_in_

    st.sidebar.header("Farmer Input Parameters/କୃଷକ ଇନପୁଟ୍ ପାରାମିଟରଗୁଡିକ")

    w_crop = st.sidebar.selectbox("Crop/ଫସଲ:", options=crop_options)
    w_state = st.sidebar.selectbox("State/ରାଜ୍ୟ :", options=state_options, index=state_options.index("Odisha"))
    w_season = st.sidebar.selectbox("Season/ଋତୁ:", options=season_options, index=season_options.index("Kharif"))
    w_year = st.sidebar.number_input("Year/ବର୍ଷ:", min_value=1997, max_value=2050, value=dt_date.today().year)
    w_area = st.sidebar.number_input("Area (in Hectares)/କ୍ଷେତ୍ରଫଳ (ହେକ୍ଟରରେ):", min_value=0.1, value=float(df['Area'].median()), format="%.2f")
    w_fert = st.sidebar.number_input("Fertilizer (kg/ha)/ସାର (କେଜି/ହେକ୍ଟର):", min_value=0.0, value=float(df['Fertilizer'].median()), format="%.2f")
    w_pest = st.sidebar.number_input("Pesticide (g/ha)/କୀଟନାଶକ (ଗ୍ରାମ/ହେକ୍ଟର):", min_value=0.0, value=float(df['Pesticide'].median()), format="%.2f")

    if st.sidebar.button("Fetch Rainfall for Year & State/ବର୍ଷ ଏବଂ ରାଜ୍ୟ ପାଇଁ ବର୍ଷା ଆଣନ୍ତୁ", use_container_width=True):
        with st.spinner("Fetching rainfall data...\nବର୍ଷାର ତଥ୍ୟ ଆଣୁଛି..."):
            rain_val = fetch_rainfall_from_api(w_state, w_year)
        if not math.isnan(rain_val):
            st.session_state.rainfall = rain_val
            st.sidebar.success(f"Fetched: {rain_val:.1f} mm")
    
    w_rain = st.sidebar.number_input(
        "Annual Rainfall (mm)/ବାର୍ଷିକ ବର୍ଷା (ମିମି):", 
        min_value=0.0, 
        value=st.session_state.get('rainfall', float(df['Annual_Rainfall'].median())),
        format="%.2f"
    )

    if st.button("Get Recommendations/ସୁପାରିଶଗୁଡିକ ପ୍ରାପ୍ତ କରନ୍ତୁ", use_container_width=True, type="primary"):
        if 'english_advice' in st.session_state: del st.session_state.english_advice
        if 'odia_advice' in st.session_state: del st.session_state.odia_advice
        
        profile = {
            'Crop': w_crop, 'Crop_Year': w_year, 'Season': w_season,
            'State': w_state, 'Area': w_area, 'Annual_Rainfall': w_rain,
            'Fertilizer': w_fert, 'Pesticide': w_pest,
        }
        
        display_profile = {
            'Crop/ଫସଲ': w_crop, 'Crop_Year/ଫସଲ ବର୍ଷ': w_year, 'Season/ଋତୁ': w_season,
            'State/ରାଜ୍ୟ': w_state, 'Area/କ୍ଷେତ୍ର': w_area, 'Annual_Rainfall/ବାର୍ଷିକ ବର୍ଷା': w_rain,
            'Fertilizer/ସାର': w_fert, 'Pesticide/କୀଟନାଶକ': w_pest,
        }

        input_df = pd.DataFrame([profile])
        for col in features:
            if col not in input_df.columns:
                input_df[col] = np.nan
                
        with st.spinner("Analyzing data and predicting yield..."):
            y_pred = model_pipeline.predict(input_df)[0]
            st.session_state.prediction = y_pred
            st.session_state.profile_for_display = display_profile
            st.session_state.profile_for_llm = profile

            if not st.session_state.get('logged_in', False):
                save_guest_data(profile)

        if GEMINI_API_KEY:
            with st.spinner("Generating personalized advice with Gemini AI..."):
                llm_advice = get_llm_suggestions(
                    api_key=GEMINI_API_KEY,
                    predicted_yield=y_pred,
                    farmer_profile=st.session_state.profile_for_llm
                )
                st.session_state.english_advice = llm_advice

    if 'prediction' in st.session_state:
        st.subheader("📋 Farmer Profile/କୃଷକ ପ୍ରୋଫାଇଲ୍")
        st.table(pd.DataFrame([st.session_state.profile_for_display]))
        
        st.subheader("📈 Predicted Yield/ପୂର୍ବାନୁମାନିତ ଅମଳ")
        st.metric(label="Predicted Crop Yield/ପୂର୍ବାନୁମାନିତ ଫସଲ ଅମଳ", value=f"{st.session_state.prediction:.2f} tons/hectare")

    if 'english_advice' in st.session_state:
        st.subheader("💡 AI Advisor's Summary")
        st.markdown(st.session_state.english_advice)
        
        if st.button("Translate to Odia/ଓଡିଆକୁ ଅନୁବାଦ କରନ୍ତୁ", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("Cannot translate without a GEMINI_API_KEY.")
            else:
                with st.spinner("Translating..."):
                    st.session_state.odia_advice = translate_text_gemini(
                        text_to_translate=st.session_state.english_advice,
                        target_language="Odia",
                        api_key=GEMINI_API_KEY
                    )

    if 'odia_advice' in st.session_state:
        st.subheader("ଓଡ଼ିଆରେ ପରାମର୍ଶ (Advice in Odia)")
        st.markdown(st.session_state.odia_advice)

    if not GEMINI_API_KEY and 'prediction' in st.session_state:
        st.warning("`GEMINI_API_KEY` not found. Please create a `.env` file to enable AI-powered advice and translation.")


# --- Main App Logic: Load data once and route pages ---

if "page" not in st.session_state:
    st.session_state.page = "login"

# Load essential data and model files once at the start
model_pipeline = load_model(os.path.join("artifacts", "yield_model_pipeline.joblib"))
df = load_data("sorted_data.csv")

if st.session_state.page != "login" and (model_pipeline is None or df is None):
    st.error("Failed to load essential model or data files. The application cannot start.")
    st.stop()

# Page router
if st.session_state.page == "login":
    render_login_page()
elif st.session_state.page == "cover":
    render_cover_page()
elif st.session_state.page == "main_app":
    render_main_app(df, model_pipeline)