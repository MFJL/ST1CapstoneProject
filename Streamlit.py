# Created by Myeisha Foo u3241507

import streamlit as st
import pandas as pd
import seaborn as sns
import requests
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from PIL import Image
from Model import Prediction

# Create page
st.set_page_config(page_title="Lung Cancer Webpage", page_icon=":shooting_star:", layout="wide")

# Create navigation bar
selected = option_menu(
    menu_title=None,
    options=["Home", "Data", "About"],
    icons=["house", "clipboard-data", "person"],
    default_index=0,
    orientation="horizontal"
)


# Create gif function
def load_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Create home page
def home():
    with st.container():
        # Create 2 columns
        left_column, right_column = st.columns([2, 3], gap="medium")
        # Create title with description of heart attacks
        with left_column:
            st.title("Lung Cancer")
            st.write(
                "")

            st.write("")
            st.write("")

        # Display GIF
        with right_column:
            asteroid_gif2 = load_url("https://lottie.host/b6523534-9d4f-42bb-8147-e7e540597070/5t9FP3Cp7Y.json")
            st_lottie(asteroid_gif2, height=500, key="coding2")


# Create drop down menu to select EDA and PDA pages
def pages():
    page = st.sidebar.selectbox("Select Predict or Explore", ("Predict", "Explore"))
    if page == "Predict":
        predict_page()
    if page == "Explore":
        explore()


# Create prediction page(PDA)
# Mapping for categorical variables
gender_mapping = {'Male': 1, 'Female': 2}
yes_no_mapping = {'Yes': 2, 'No': 1}

# Create prediction page(PDA)
def predict_page():
    df = pd.read_csv("surveylungcancer.csv")

    # Collect user input
    st.sidebar.header('Check Your Lung Cancer Risk')
    user_age = st.sidebar.slider('Your Age', min(df['AGE']), max(df['AGE']), min(df['AGE']))
    user_gender = st.sidebar.selectbox('Your Gender', ['Male', 'Female'])
    user_smoking = st.sidebar.selectbox('Do You Smoke?', ['Yes', 'No'])
    user_yellow_fingers = st.sidebar.selectbox('Do You Have Yellow Fingers?', ['Yes', 'No'])
    user_anxiety = st.sidebar.selectbox('Do You Have Anxiety?', ['Yes', 'No'])
    user_peer_pressure = st.sidebar.selectbox('Are You Experiencing Peer Pressure?', ['Yes', 'No'])
    user_chronic_disease = st.sidebar.selectbox('Do You Have Chronic Disease?', ['Yes', 'No'])
    user_fatigue = st.sidebar.selectbox('Are You Experiencing Fatigue?', ['Yes', 'No'])
    user_allergy = st.sidebar.selectbox('Do You Have Allergies?', ['Yes', 'No'])
    user_wheezing = st.sidebar.selectbox('Are You Experiencing Wheezing?', ['Yes', 'No'])
    user_alcohol = st.sidebar.selectbox('Do You Consume Alcohol?', ['Yes', 'No'])
    user_coughing = st.sidebar.selectbox('Are You Experiencing Coughing?', ['Yes', 'No'])
    user_shortness_of_breath = st.sidebar.selectbox('Are You Experiencing Shortness of Breath?', ['Yes', 'No'])
    user_swallowing_difficulty = st.sidebar.selectbox('Do You Have Swallowing Difficulty?', ['Yes', 'No'])
    user_chest_pain = st.sidebar.selectbox('Are You Experiencing Chest Pain?', ['Yes', 'No'])

    # Convert user inputs to numerical values
    user_data = {
        "AGE": user_age,
        "GENDER": gender_mapping[user_gender],
        "SMOKING": yes_no_mapping[user_smoking],
        "YELLOW_FINGERS": yes_no_mapping[user_yellow_fingers],
        "ANXIETY": yes_no_mapping[user_anxiety],
        "PEER_PRESSURE": yes_no_mapping[user_peer_pressure],
        "CHRONIC DISEASE": yes_no_mapping[user_chronic_disease],
        "FATIGUE": yes_no_mapping[user_fatigue],
        "ALLERGY": yes_no_mapping[user_allergy],
        "WHEEZING": yes_no_mapping[user_wheezing],
        "ALCOHOL": yes_no_mapping[user_alcohol],
        "COUGHING": yes_no_mapping[user_coughing],
        "SHORTNESS_OF_BREATH": yes_no_mapping[user_shortness_of_breath],
        "SWALLOWING_DIFFICULTY": yes_no_mapping[user_swallowing_difficulty],
        "CHEST_PAIN": yes_no_mapping[user_chest_pain],
    }

    features = pd.DataFrame(user_data, index=[0])

    # Create headings
    st.title("Lung Cancer Prediction 🫁")
    st.subheader(f"This prediction has an accuracy of: {Prediction.model_accuracy:.0%}")

    # Show user input from sliders
    st.subheader("User Input")
    st.write("Key => 1 : No  |  2 : Yes")
    df = features
    st.write(df)

    # Predict outcome
    risk_info = list(user_data.values())
    risk_prediction = Prediction.best_model.predict([risk_info])
    if risk_prediction == [2]:
        st.success("Prediction: Warning! You are at risk of Lung Cancer!!!")
    else:
        st.error("Prediction: You are not at risk of Lung Cancer.")


# Create EDA page
def explore():
    # Create title
    st.title("Exploratory Data Analysis")
    df = pd.read_csv("surveylungcancer.csv")

    st.sidebar.header('Filters')

    age_range = st.sidebar.slider('Select Age Range', min(df['AGE']), max(df['AGE']), (min(df['AGE']), max(df['AGE'])))
    gender_filter = st.sidebar.selectbox('Select Gender', ['All', 'Male', 'Female'])
    lung_cancer_filter = st.sidebar.selectbox('Select Lung Cancer Status', ['All', 'Yes', 'No'])

    # Filter the data
    filtered_df = df[(df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])]
    if gender_filter != 'All':
        filtered_df = filtered_df[filtered_df['GENDER'] == (1 if gender_filter == 'Male' else 2)]
    if lung_cancer_filter != 'All':
        filtered_df = filtered_df[filtered_df['LUNG_CANCER'] == (1 if lung_cancer_filter == 'Yes' else 0)]

    # Display dataframe
    st.subheader("Lung Cancer Data frame")
    st.dataframe(df)

    # Question 1: Distribution of lung cancer cases
    st.write('### Question 1:  What is the distribution of lung cancer cases in the dataset?')
    custom_palette = {'NO': '#d3a0f0', 'YES': '#8a2be2'}
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=filtered_df, x='LUNG_CANCER', palette=custom_palette, ax=ax1)
    st.pyplot(fig1)
    st.write('Findings : In the dataset there are more people with lung cancer than without')

    # Question 2: Age vs. lung cancer
    st.write('### Question 2: How does age related to the likelihood of lung cancer?')
    fig2, ax2 = plt.subplots(figsize=(15, 5), dpi=80)
    sns.kdeplot(data=filtered_df, x='AGE', hue="LUNG_CANCER", shade=True, common_norm=False,
                palette=['#ff9f9f', '#a3a3ec'], ax=ax2)
    st.pyplot(fig2)
    st.write('Findings : People who are typically in the age 20 range tend to have healthy lungs. '
             'The age ranges between 30 to 90 it can vary whether you have healthy or unhealthy lungs.')

    # Question 3: Correlation heatmap
    st.write('### Question 3: Are there any significant correlations between smoking and other attributes with the presence of lung cancer?')
    # Exclude non-numeric columns from the correlation calculation
    numeric_columns = filtered_df.select_dtypes(include=['number'])
    correlation_matrix = numeric_columns.corr()

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)
    st.write('Findings : According to the heat correlation map majority of the symptoms and conditions'
             ' have no correlation to one another. With the exception of Anxiety and Yellow fingers.')

    # Question 4: Common symptoms and conditions
    st.write('### Question 4: What are the most common symptoms and conditions associated with lung cancer?')
    symptoms_columns = filtered_df.columns[3:-1]  # Exclude the first three columns (GENDER, AGE, SMOKING)
    symptoms_counts = filtered_df[symptoms_columns].sum().sort_values(ascending=False)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=symptoms_counts.values, y=symptoms_counts.index, palette='viridis', ax=ax4)
    st.pyplot(fig4)
    st.write('Finding : The most common symptom and condition is Fatigue, Shortness Of Breath, and Coughing.'
             'The least common symptom and condition is Peer Pressure, Anxiety, and Swallowing Difficulty.')

    # Question 5: Does Gender have a correlation to lung cancer
    st.write('### Question 5: Does Gender have correlation to lung cancer')
    df_plot = filtered_df.copy()
    df_plot['GENDER'] = df_plot['GENDER'].replace({1: "Male", 2: "Female"})
    df_plot['LUNG_CANCER'] = df_plot['LUNG_CANCER'].replace({0: "No", 1: "Yes"})
    sns.set(style="whitegrid")
    sns.set_style("white")
    sns.despine()
    palette = ['#a3a3ec', '#ff9f9f']
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5 = sns.countplot(data=df_plot, x='GENDER', hue='LUNG_CANCER', palette=palette, ax=ax5)
    plt.xlabel("Gender", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Lung Cancer Count by Gender", fontsize=16, fontweight='bold')
    total_counts = len(filtered_df)
    for p in ax5.patches:
        count = int(p.get_height())
        percentage = f"{100 * count / total_counts:.2f}%"
        ax5.annotate(f'{count} ({percentage})', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                     va='baseline', fontsize=12, color='black')
    st.pyplot(fig5)
    st.write('Findings : Gender does not have a correlation to having or not having lung cancer '
             'It shows 46.93% of men have lung cancer and 40.45% of women have lung cancer, having a 6.48% difference.'
             'It shows 5.50% of men do not have lung cancer and 7.12% of women do not have lung cancer, having a 2.48% '
             'difference.')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write(df.to_markdown(), unsafe_allow_html=True)


# About page
def about():
    # Create heading
    st.markdown("<h1 style='text-align: center; color: white;'>About Me</h1>", unsafe_allow_html=True)
    st.write("")
    with st.container():
        col1, col2, col3 = st.columns([1.75, 3, 1.8])
        with col1:
            st.write("")
        with col2:
            # image = Image.open("IMG1.jpg")
            # st.image(image, width=600)
            st.write("I am a student at UC studying a Bachelors in Information Technology "
                     "u3241507@uni.canberra.edu.au ")
        with col3:
            st.write("")


# Pages
if selected == "Home":
    home()
if selected == "About":
    about()
if selected == "Data":
    pages()
