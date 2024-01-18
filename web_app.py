import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, precision_score
import matplotlib.pyplot as plt
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
    st.write(
        """
        Welcome to Your Streamlit App, where we understand the dynamic landscape of social media, especially Instagram. 
        As a content creator, staying ahead of changes is crucial for sustained success on this platform. Our mission is to 
        empower you with the tools to adapt and thrive in the ever-evolving world of Instagram. Whether you're promoting 
        your business, building a portfolio, or expressing your creativity through content, our Instagram Reach Analysis 
        using Python provides valuable insights. Explore how data science can help you understand and navigate Instagram's 
        changes, allowing you to optimize your reach and performance in the long run. Join us in this journey of data-driven 
        success on Instagram.
        """
    )

    st.subheader('Models Used: ')
    
    st.markdown("We employed a diverse set of machine learning models in our analysis, each chosen for its specific strengths and capabilities. "
             "The models include <strong style='color:red;'>Passive Aggressive Regression</strong>, known for its adaptability to changing "
             "data streams, <strong style='color:red;'>Multi Linear Regression</strong>, which explores linear relationships among multiple "
             "features, and <strong style='color:red;'>Random Forest</strong>, a versatile ensemble method. Each model underwent rigorous "
             "testing and tuning to ensure optimal performance in predicting Instagram impressions.", unsafe_allow_html=True)

    st.markdown("<strong style='color:red;'>Passive Aggressive Regression</strong> excels in scenarios where data distribution may change over time, "
             "making it suitable for dynamic social media platforms like Instagram. <strong style='color:red;'>Multi Linear Regression</strong>, "
             "on the other hand, provides insights into how different features collectively influence impression metrics. The "
             "<strong style='color:red;'>Random Forest</strong> model, being an ensemble of decision trees, brings robustness and the ability "
             "to capture non-linear relationships within the data.", unsafe_allow_html=True)

    st.markdown("Our approach involves leveraging the strengths of each model to create a more comprehensive understanding of Instagram reach dynamics. "
             "By combining the predictive power of these models, we aim to assist content creators in making informed decisions to enhance the "
             "effectiveness of their Instagram content strategy.", unsafe_allow_html=True)

    st.subheader('Model Scores: ')
    # Model Scores (Pie Chart)
    models = ['Passive Aggressive', 'Multi Linear', 'Random Forest']

    rmse_scores = [par_r2, mlr_r2, rfr_r2]
    mae_scores = [par_mae, mlr_mae, rfr_mae]
    mse_scores = [par_mse, mlr_mse, rfr_mse]
    accuracy_scores = [par_ac, mlr_ac, rfr_ac]

    scores = [
    rmse_scores,  # Replace with your actual scores for Model 1
    mae_scores,  # Replace with your actual scores for Model 2
    mse_scores,  # Replace with your actual scores for Model 3
    accuracy_scores,  # Replace with your actual scores for Model 4
]

   # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plotting Pie Charts
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            model_index = i * 2 + j
            ax.pie(scores[model_index], labels=models, autopct='%1.1f%%',
                colors=['skyblue', 'lightgreen', 'lightcoral'], startangle=90)
            ax.set_title(f'Model {model_index + 1} Scores')

    # Adjust layout
    plt.tight_layout()

    # Display Subplots using Streamlit
    st.pyplot(fig)

    # Display Table with Scores
    st.markdown("## Model Scores Table")

    # Create a table
    score_table_data = {
        'Model': models,
        'R2 Score': rmse_scores,
        'MAE': mae_scores,
        'MSE': mse_scores,
        'Accuracy': accuracy_scores
}

    # Display the table
    st.table(score_table_data)
