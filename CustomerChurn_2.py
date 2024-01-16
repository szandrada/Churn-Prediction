import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import time

# Custom colors
colors = ["#619b8a", "#a1c181", "#fcca46", "#fe7f2d"]

#Title and page icon
st.set_page_config(page_title="Churn Prediction App", page_icon="📈")
st.title("Customer Churn Prediction")

with st.spinner("Loading..."):
    # Task that takes time
    time.sleep(5)

# Load the dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if df is not None:
    st.write("Dataset Information")
    st.dataframe(df)

    # Number of Products Distribution
    st.subheader("Number of Products Distribution")
    num_products_counts = df['NumOfProducts'].value_counts()
    labels = num_products_counts.index
    sizes = num_products_counts.values

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'color': 'white'}, colors=colors)
    ax.axis('equal')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    tmpfile_pie = "temp_pie_chart.png"
    fig.savefig(tmpfile_pie, bbox_inches='tight', transparent=True)
    st.image(tmpfile_pie)

    # Age Distribution 
    st.subheader("Age Distribution")
    plt.figure(figsize=(8, 5))
    hist = plt.hist(df['Age'], bins=20, edgecolor='k', alpha=0.7, color=colors[1])
    plt.gca().set_facecolor('none')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.xlabel('Age', color='white')
    plt.ylabel('Count', color='white')
    tmpfile_histogram = "temp_histogram.png"
    plt.savefig(tmpfile_histogram, bbox_inches='tight', transparent=True)
    st.image(tmpfile_histogram)

    plt.close()

    # Gender Distribution
    st.subheader("Gender Distribution")
    gender_counts = df['Gender'].value_counts()
    st.bar_chart(gender_counts, color=colors[2])

    # Sidebar for user input
    st.sidebar.header(" Prediction")
    st.sidebar.subheader("Select Data for Prediction")
    age = st.sidebar.slider("Select Age", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].mean()))
    geography = st.sidebar.selectbox("Select Geography", df['Geography'].unique())
    gender = st.sidebar.selectbox("Select Gender", df['Gender'].unique())
    credit_score = st.sidebar.slider("Select Credit Score", min_value=int(df['CreditScore'].min()), max_value=int(df['CreditScore'].max()), value=int(df['CreditScore'].mean()))
    balance = st.sidebar.slider("Select Balance", min_value=float(df['Balance'].min()), max_value=float(df['Balance'].max()), value=float(df['Balance'].mean()))
    num_of_products = st.sidebar.slider("Select Number of Products", min_value=int(df['NumOfProducts'].min()), max_value=int(df['NumOfProducts'].max()), value=int(df['NumOfProducts'].mean()))
    has_cr_card = st.sidebar.selectbox("Has Credit Card?", [0, 1])
    is_active_member = st.sidebar.selectbox("Is Active Member?", [0, 1])
    estimated_salary = st.sidebar.slider("Select Estimated Salary", min_value=float(df['EstimatedSalary'].min()), max_value=float(df['EstimatedSalary'].max()), value=float(df['EstimatedSalary'].mean()))

    # Prepare the data for training
    le_geography = LabelEncoder()
    le_gender = LabelEncoder()

    df['Geography'] = le_geography.fit_transform(df['Geography'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    user_data = [[credit_score, le_geography.transform([geography])[0], le_gender.transform([gender])[0], age, 1, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]]
    prediction = model.predict(user_data)

    # Display the prediction
    st.sidebar.subheader("Churn Prediction:")
    st.sidebar.write("The predicted churn probability is:", prediction[0])

    # Evaluate the model
    st.sidebar.subheader("Model Evaluation:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.write("Accuracy:", accuracy)

