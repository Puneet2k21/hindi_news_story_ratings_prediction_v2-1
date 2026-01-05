import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import yaml
import streamlit_authenticator as stauth
import datetime
import gspread
from google.oauth2.service_account import Credentials
import pytz

# -------------------------------
# 1. Load Authentication Config
# -------------------------------
with open("allowed_users.yaml") as file:
    config = yaml.safe_load(file)

# -------------------------------
# 2. Setup Google Sheet for Login Logs
# -------------------------------
def init_google_sheet():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    service_account_info = st.secrets["service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("Streamlit_login_track").worksheet("hindi_news_app")
    return sheet

def log_user_login(username):
    sheet = init_google_sheet()
    ist = pytz.timezone('Asia/Kolkata')
    login_time = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    new_row = [username, login_time]
    sheet.append_row(new_row)

# -------------------------------
# 3. Login Authentication
# -------------------------------
authenticator = stauth.Authenticate(
    config['credentials'],
    'news_app_cookie_test',
    'abc123',
    cookie_expiry_days=7
)

login_result = authenticator.login()

# -------------------------------
# 4. After Successful Login
# -------------------------------
if st.session_state['authentication_status']:
    log_user_login(st.session_state["username"])
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')

    # App Title
    st.title("Hindi News Story Rating Prediction based on Machine Learning model")

    # ----------------------------------------
    # 5. Define Dropdown Input Options
    # ----------------------------------------
    genre_options = ["ASTROLOGY", "CAREER/EDUCATION", "CRIME/LAW & ORDER", "ENTERTAINMENT NEWS",
                     "EVENT/CELEBRATIONS", "FINANCIAL NEWS", "HEALTH", "INDIA-PAK",
                     "MISHAPS/FAILURE OF MACHINERY", "NATIONAL THREAT/DEFENCE NEWS",
                     "POLITICAL NEWS/GOVERNMENT NEWS", "RELIGIOUS / FAITH", "SCIENCE/SPACE",
                     "SPORTS NEWS", "WAR", "WEATHER/ENVIRONMENT", "OTHER"]

    geography_options = ["INDIAN", "INTERNATIONAL", "BIHAR", "CHANDIGARH", "CHHATTISGARH", "DELHI",
                         "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU AND KASHMIR", "JHARKHAND",
                         "KARNATAKA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "RAJASTHAN",
                         "TELANGANA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "OTHER"]

    popularity_options = ["H", "M", "L"]

    personality_genre_options = ["Astrologer", "Cricketer", "Defense", "Entertainer", "International", "Religious", "AAP", "AIMIM",
                                 "Bajrang Dal", "BJP", "BSP", "DMK", "INC", "JDU", "JMM", "NC", "NCP", "RJD", "RSS-VHP",
                                 "SBSP", "SP", "SS", "TMC", "OTHER"]

    logistics_options = ["ON LOCATION", "IN STUDIO", "BOTH"]
    story_format_options = ["DEBATE OR DISCUSSION", "INTERVIEW", "NEWS REPORT"]

    # ----------------------------------------
    # 6. Input Fields
    # ----------------------------------------
    genre = st.selectbox("Genre", genre_options)
    geography = st.selectbox("Geography", geography_options)
    personality_popularity = st.selectbox("Personality Popularity (H/M/L)", popularity_options)
    personality_genre = st.selectbox("Personality Genre", personality_genre_options)
    logistics = st.selectbox("Logistics", logistics_options)
    story_format = st.selectbox("Story Format", story_format_options)

    if st.button("Predict Tier"):

        # ----------------------------------------
        # 7. Prepare DataFrame from Input
        # ----------------------------------------
        new_data = pd.DataFrame({
            'Genre': [genre],
            'Geography': [geography],
            'Personality Popularity': [personality_popularity],
            'Personality-Genre': [personality_genre],
            'Logistics': [logistics],
            'Story_Format': [story_format]
        })

        new_data['Personality Popularity Ord'] = new_data['Personality Popularity'].map({'H': 2, 'M': 1, 'L': 0})

        categorical_columns = ['Genre', 'Geography', 'Personality Popularity', 'Personality-Genre',
                               'Logistics', 'Story_Format']

        def df_to_input_dict(df, columns):
            return {f"{col}_input": df[col].values for col in columns}

        # ----------------------------------------
        # 8. Load Label Encoders
        # ----------------------------------------
        with open("label_encoders_model3.pkl", "rb") as f:
            label_encoders_model3 = pickle.load(f)
        with open("label_encoders_model4.pkl", "rb") as f:
            label_encoders_model4 = pickle.load(f)

        encoded_data_model3 = new_data.copy()
        encoded_data_model4 = new_data.copy()

        for col in categorical_columns:
            encoded_data_model3[col] = label_encoders_model3[col].transform(encoded_data_model3[col])
            encoded_data_model4[col] = label_encoders_model4[col].transform(encoded_data_model4[col])

        input_dict_model3 = df_to_input_dict(encoded_data_model3, categorical_columns)
        input_dict_model4 = df_to_input_dict(encoded_data_model4, categorical_columns)

        # ----------------------------------------
        # 9. Load Models
        # ----------------------------------------
        model3_paths = [f"model3_fold{i}_best.keras" for i in range(1, 6)]
        model4_paths = [f"model4_fold{i}_best.keras" for i in range(1, 6)]
        models = [load_model(path) for path in model3_paths + model4_paths]

        # ----------------------------------------
        # 10. Weighted Soft Voting
        # ----------------------------------------
        weights = [0.0838, 0.0720, 0.1210, 0.0938, 0.1080,
                   0.0780, 0.0907, 0.1180, 0.1121, 0.1227]  # From your F1 scores

        soft_preds = None
        for model, weight, input_dict in zip(models[:5] + models[5:], weights,
                                             [input_dict_model3]*5 + [input_dict_model4]*5):
            probs = model.predict(input_dict, verbose=0)
            soft_preds = probs * weight if soft_preds is None else soft_preds + probs * weight

        final_pred = np.argmax(soft_preds, axis=1)[0]

        # ----------------------------------------
        # 11. Display Result
        # ----------------------------------------
        tier_map = {
            0: 'Minimal viewership',
            1: 'Low viewership',
            2: 'Average viewership',
            3: 'High viewership',
            4: 'Max viewership'
        }

        st.success(f"Predicted Tier: {final_pred} - {tier_map[final_pred]}")

        # Note with Markdown formatting

        note = (
        "The predicted value tier is determined based on a five-point scale, ranging from lowest to highest. "
        "The tiers are categorized as follows:\n\n"
        
        "• **Minimal Viewership**: Less than 213 TVTs  \n"
        "• **Low Viewership**: 213 to 244 TVTs  \n"
        "• **Average Viewership**: 244 to 276 TVTs  \n"
        "• **High Viewership**: 277 to 318 TVTs  \n"
        "• **Maximum Viewership**: 319 TVTs and above."
)
        st.markdown(note)


# -------------------------------
# 12. Login Failure Handling
# -------------------------------
elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')

# -------------------------------
# 13. Footer
# -------------------------------
st.write("""
---
**Note**: This app leverages machine learning to predict news ratings, offering insights based on historical data. 
Predictions should be combined with domain expertise. The developer is not responsible for outcomes based solely on the app's predictions. 
For technical details on ML models employed and error metrics, contact: 
**Puneet Sah**  
📧 puneet2k21@gmail.com  
📞 9820615085
""")
