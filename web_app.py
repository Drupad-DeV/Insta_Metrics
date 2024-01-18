import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, precision_score
import streamlit as st
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Instagram data.csv', header=0, encoding='latin1')
df = df.dropna()

# Assume 'features' are your input features and 'target' is your target variable
features = df[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
target = df['Impressions']

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=42)

# Passive Agressive Regresion Model
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
par_prediction = model.predict(xtest)

#Evaluating the Perfomance for the Model 
par_r2 = r2_score(ytest, par_prediction)
par_mse = mean_squared_error(ytest, par_prediction)
par_mae = mean_absolute_error(ytest, par_prediction)
par_ac = model.score(xtest, ytest)

# Random Forest Regression model
rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
rfr_model.fit(xtrain, ytrain)
rfr_predictions = rfr_model.predict(xtest)

rfr_r2 = r2_score(ytest, rfr_predictions)
rfr_mse = mean_squared_error(ytest, rfr_predictions)
rfr_mae = mean_absolute_error(ytest, rfr_predictions)
rfr_ac = rfr_model.score(xtest, ytest)

# Multiple Linear Regression model
mlr_model = LinearRegression()
mlr_model.fit(xtrain, ytrain)
mlr_predictions = mlr_model.predict(xtest)

mlr_r2 = r2_score(ytest, mlr_predictions)
mlr_mse = mean_squared_error(ytest, mlr_predictions)
mlr_mae = mean_absolute_error(ytest, mlr_predictions)
mlr_ac = mlr_model.score(xtest, ytest)



#main


st.set_page_config(
    page_title="Insta Metrics",
    page_icon="📊",  # You can use emojis as icons
    layout="centered",  # "wide" or "centered"
)

# Add cover image and header
st.markdown(
    """
    <style>
    body {
        background-size: cover;
    }
    .stApp {
        max-width: 1200px;  /* Set maximum width for the content */
        margin: auto;  /* Center the content */
        color: #fff;  /* Set text color to white */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar menu
selected_option = st.sidebar.selectbox("Select Tracking Mode", ["About US","Live Impression Tracking", "Manual Impression Tracking"])

st.image("https://github.com/Drupad-DeV/Insta_Metrics/assets/100958162/3733d8a0-a1a6-4af9-9224-b7d0903b73a1")




# Display content based on selected option
if selected_option == "Live Impression Tracking":
    # Add content for live tracking (leave as is)
    st.write("Live Impression Tracking Content Goes Here")
elif selected_option == "Manual Impression Tracking":
    likes = st.slider('Likes:', min_value=0, max_value=1000, value=500)
    saves = st.slider('Saves:', min_value=0, max_value=1000, value=250)
    comments = st.slider('Comments:', min_value=0, max_value=100, value=25)
    shares = st.slider('Shares:', min_value=0, max_value=1000, value=10)
    profile_visits = st.slider('Profile Visits:', min_value=0, max_value=1000, value=100)
    follows = st.slider('Follows:', min_value=0, max_value=1000, value=10)

    # Make Prediction Using Model 
    prediction_pa = model.predict([[likes, saves, comments, shares, profile_visits, follows]])
    prediction_rf = rfr_model.predict([[likes, saves, comments, shares, profile_visits, follows]])
    prediction_mlr = mlr_model.predict([[likes, saves, comments, shares, profile_visits, follows]])

    # Display predictions
    st.subheader('Predictions:')
    st.write(f'Passive Aggressive Regressor Prediction: {prediction_pa[0]:,.2f} Impressions')
    st.write(f'Random Forest Regressor Prediction: {prediction_rf[0]:,.2f} Impressions')
    st.write(f'Linear Regression Prediction: {prediction_mlr[0]:,.2f} Impressions')
else:
    # Add content for advanced analytics (customize as needed)
    st.title('InstaMetrics Predictor')
    st.subheader('Predict Instagram Impressions')
    st.subheader('Ippo Kittile Harin aee??')
