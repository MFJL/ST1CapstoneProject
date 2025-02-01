# **ST1 Capstone Project – Lung Cancer Prediction**

## **📌 Project Overview**
This project aims to predict the risk of lung cancer using machine learning techniques. We analyzed a lung cancer dataset, performed **Exploratory Data Analysis (EDA)**, built multiple classification models, and deployed the best-performing model as a **Streamlit web application**. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer/data).


## **🛠 Installation & Setup**
### **🔹 Prerequisites**
Ensure you have **Python 3.x** installed. Install required dependencies using:
```bash
pip install -r requirements.txt
```

### **🔹 Running the Project**
1. Clone this repository:
   ```bash
   git clone https://github.com/MFJL/ST1CapstoneProject.git
   cd ST1CapstoneProject
   ```
2. Run the Jupyter Notebooks for EDA and model training:
   ```bash
   jupyter notebook
   ```
3. Execute scripts for model training and Streamlit deployment:
   ```bash
   python src/Model.py
   python src/Streamlit.py
   ```

## **📊 Data Description**
- **Dataset Name:** Lung Cancer Prediction Dataset
- **Source:** Kaggle ([Link](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer/data))
- **Instances:** 284 patients
- **Features:** 16 attributes, including age, smoking history, gender, symptoms
- **Target Variable:** Presence of **Lung Cancer (Yes/No)**

## **📈 Machine Learning Models & Results**
We tested multiple classification models for lung cancer prediction:
- ✅ **Random Forest Classifier** – Best accuracy (88%)
- ✅ **Naïve Bayes** – Moderate performance
- ✅ **Decision Trees** – Overfitting observed
- ✅ **Best Model:** **Random Forest Classifier with Ordinal Encoding**

### **📌 Key Findings**
✔ **Smoking and age were the most significant factors for lung cancer prediction.**
✔ **Symptoms like wheezing, coughing, and fatigue showed strong correlations.**
✔ **The deployed model can provide early lung cancer risk assessment.**

## **🌐 Web Application (Streamlit Deployment)**
The **Streamlit web app** allows users to input symptoms and get lung cancer risk predictions.
- **To launch the web app locally:**
  ```bash
  streamlit run src/Streamlit.py
  ```

## **📢 Contributors**
👩‍💻 **Myeisha Foo** – [GitHub Profile](https://github.com/MFJL)  


