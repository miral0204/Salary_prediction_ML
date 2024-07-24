import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import io

# Title of the app
st.title("Texas Salary Prediction")


# Project Overview
st.header("Project Overview")
st.write(
    """
    The dataset provided includes employment details for state offices in Texas, including work weeks and salary information. The goal of this project is to develop a predictive model that provides insights and payroll information for the state of Texas.
    """
)

# Data Cleaning and Processing
st.header("Data Cleaning and Processing")
st.write(
    """
    The dataset needed to be cleaned and processed before training the machine learning models. The following steps were taken:
    - Processed null values
    - Handled categorical and continuous values
    - Performed Exploratory Data Analysis (EDA)
    """
)

# Exploratory Data Analysis (EDA)
st.header("Exploratory Data Analysis (EDA)")
st.write(
    """
    EDA provided insights into the data distribution and the relationships between different fields. Key findings include:
    - Many columns were not normally distributed and had outliers
    - Outliers were computed as they can affect model performance
    - Analysis of annual salary disparity across different departments and statuses
    """
)

# Model Training and Evaluation
st.header("Model Training and Evaluation")
st.write(
    """
    Various machine learning models were trained and evaluated. The models included:
    - XGBoost Regressor
    - Random Forest Regressor
    - Linear Regression
    - Decision Tree Regressor
    - K-Nearest Neighbors Regressor
    
    Out of these, XGBoost Regressor and Random Forest Regressor provided the best performance with an R2 score of 0.94.
    """
)

# Hyperparameter Tuning
st.header("Hyperparameter Tuning")
st.write(
    """
    To further improve model performance, hyperparameter tuning was performed with different parameters to find the optimal settings for XGBoost and Random Forest.
    """
)

# Upload dataset
st.header("Choose a file ")
uploaded_file = st.file_uploader("### Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display dataset
    st.subheader("Dataset")
    st.write(data.head())

    # EDA Section
    st.subheader("Exploratory Data Analysis (EDA)")

    # Dataset info
    st.write("Dataset Information:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Descriptive statistics
    st.write("Descriptive Statistics:")
    st.write(data.describe())

    # Missing values
    st.write("Missing Values:")
    st.write(data.isna().sum())

    # Drop irrelevant columns
    data.drop(['duplicated', 'multiple_full_time_jobs', 'combined_multiple_jobs', 'hide_from_search', 'summed_annual_salary', 'STATE NUMBER'], axis=1, inplace=True)

    # Display cleaned data
    st.write("Cleaned Data:")
    st.write(data.head())

    # Boxplots
    st.write("Boxplots for Numerical Columns:")
    fig, ax = plt.subplots(2, 2, figsize=(15, 18))
    sns.boxplot(data['HRLY RATE'], ax=ax[0, 0])
    sns.boxplot(data['HRS PER WK'], ax=ax[0, 1])
    sns.boxplot(data['MONTHLY'], ax=ax[1, 0])
    sns.boxplot(data['ANNUAL'], ax=ax[1, 1])
    st.pyplot(fig)
    st.write("We have quite a lot of outliers as seen in the boxplot, thus, processing the outliers by Inter Quartile Range method")

    # Handling outliers
    def find_boundaries(variable):
        q1 = data[variable].quantile(0.25)
        q3 = data[variable].quantile(0.75)
        iqr = q3 - q1
        lower_range = q1 - 1.5 * iqr
        upper_range = q3 + 1.5 * iqr
        return lower_range, upper_range

    lower_annual, upper_annual = find_boundaries('ANNUAL')
    data['ANNUAL'] = np.where(data['ANNUAL'] > upper_annual, upper_annual, data['ANNUAL'])
    data['ANNUAL'] = np.where(data['ANNUAL'] < lower_annual, lower_annual, data['ANNUAL'])

    lower_monthly, upper_monthly = find_boundaries('MONTHLY')
    data['MONTHLY'] = np.where(data['MONTHLY'] > upper_monthly, upper_monthly, data['MONTHLY'])
    data['MONTHLY'] = np.where(data['MONTHLY'] < lower_monthly, lower_monthly, data['MONTHLY'])

    # Boxplots after outlier handling
    st.write("Boxplots After Outlier Handling:")
    fig, ax = plt.subplots(2, 2, figsize=(15, 18))
    sns.boxplot(data['HRLY RATE'], ax=ax[0, 0])
    sns.boxplot(data['HRS PER WK'], ax=ax[0, 1])
    sns.boxplot(data['MONTHLY'], ax=ax[1, 0])
    sns.boxplot(data['ANNUAL'], ax=ax[1, 1])
    st.pyplot(fig)

    # Histograms
    st.write("Histograms:")
    fig, ax = plt.subplots(2, 2, figsize=(15, 18))
    sns.histplot(data['HRLY RATE'], ax=ax[0, 0])
    sns.histplot(data['HRS PER WK'], ax=ax[0, 1])
    sns.histplot(data['MONTHLY'], ax=ax[1, 0])
    sns.histplot(data['ANNUAL'], ax=ax[1, 1])
    st.pyplot(fig)

    # Convert the "EMPLOY DATE" to datetime and extract the year
    data['EMPLOY YEAR'] = pd.to_datetime(data['EMPLOY DATE'], errors='coerce').dt.year

# Calculate the mean annual salary by year, including handling NaNs automatically in the mean calculation
    annual_salary_by_year = data.groupby('EMPLOY YEAR')['ANNUAL'].mean().reset_index()

# Plot the trend over time
    st.write("Salaries and compensation changing over time")
    st.write("Trend of Average Annual Salary Over Time:")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(annual_salary_by_year['EMPLOY YEAR'], annual_salary_by_year['ANNUAL'], marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Annual Salary')
    ax.set_title('Average Annual Salary Over Time')
    ax.grid(True)
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("Correlation Heatmap:")
    numerical_data = data.select_dtypes(include=['int64', 'float64'])

    # Calculate correlation matrix
    correlation_matrix = numerical_data.corr()

    # Draw heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

    # Pivot table
    st.write("Number of all unique Agencies")
    st.write("Pivot Table:")
    pivot_table = pd.pivot_table(data, index=['AGENCY NAME'], values=['CLASS TITLE'], aggfunc='count')
    st.write(pivot_table)

    # Label encoding
    
    st.write("Label Encoding:")
    st.write("Converting Categorical data into numerical data for model Training. We need to use data as gender or ethnicity or status in model training but cannot use string/object in model training thats why we encode them using labelencoder which assigns a numerical value to each new unique field(data value)")
    from sklearn.preprocessing import LabelEncoder

    # Initialize label encoders
    ethnicity_le = LabelEncoder()
    gender_le = LabelEncoder()

    # Apply label encoding to 'ETHNICITY' and 'GENDER'
    data['ETHNICITY'] = ethnicity_le.fit_transform(data['ETHNICITY'].astype(str))
    data['GENDER'] = gender_le.fit_transform(data['GENDER'].astype(str))

    gender_mapping1 = dict(zip(gender_le.classes_, gender_le.transform(gender_le.classes_)))
    st.write("Gender Mapping (GENDER):", gender_mapping1)

    gender_mapping2 = dict(zip(ethnicity_le.classes_, ethnicity_le.transform(ethnicity_le.classes_)))
    st.write("Ethnicity Mapping (ETHNICITY):", gender_mapping2)

    # Gender and Ethnicity Distribution
    st.write("Gender and Ethnicity Distribution:")

    gender_distribution = data['GENDER'].value_counts()
    ethnicity_distribution = data['ETHNICITY'].value_counts()

    # Draw pie charts for GENDER and ETHNICITY
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    st.write("Gender distribution among employees, it shows how majority of employees are females (57.1%) & when it comes to ethnicity AM INDIAN and ASIAN contributes for majority of the employees")

    axes[0].pie(gender_distribution, labels=("Female", "Male"), autopct='%1.1f%%', startangle=140)
    axes[0].set_title('Gender Distribution')

    axes[1].pie(ethnicity_distribution, labels=("INDIAN", 'ASIAN', 'BLACK', 'HISPANIC', 'OTHER', "WHITE"), autopct='%1.1f%%', startangle=140)
    axes[1].set_title('Ethnicity Distribution')

    # Show the plots
    plt.tight_layout()
    st.pyplot(fig)

    # Gender Salary Comparison
    st.write("Average Summed Annual Salary by Gender:")

    gender_salary_comparison = data.groupby('GENDER')['ANNUAL'].mean()
    st.write(gender_salary_comparison)

    # Plotting the bar graph for Gender Salary Comparison
    st.write("istribution of Annual Salary between Male and Female are almost the same, There is a slight difference where Male's annual salary is almost 6k higher than female")
    fig, ax = plt.subplots(figsize=(8, 5))
    gender_salary_comparison.plot(kind='bar', color=['blue', 'orange'], ax=ax)
    ax.set_xlabel('Gender')
    ax.set_ylabel('Annual Salary')
    ax.set_title('Average Summed Annual Salary by Gender')
    st.pyplot(fig)

    numerical_data = data.select_dtypes(include=['int64', 'float64'])
    
    # Calculate correlation matrix
    corr = numerical_data.corr()
    
    # Plot heatmap
    st.write("Heatmap of Feature Correlations with Heart Disease:")
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr[['ANNUAL']].sort_values(by='ANNUAL', ascending=False), annot=True, cmap='coolwarm')
    st.pyplot(plt)
    
    
    
    # Encode categorical features
    st.write("Dataset is highly dependent on the class and status of employees, thus we cannot ignore them while model training. As they are not numeric values (string/categorical) we need to impute them as well like GENDER AND ETHNICITY")
    enc = LabelEncoder()
    if 'AGENCY NAME' in data.columns:
        data[['AGENCY NAME', 'CLASS CODE', 'CLASS TITLE', 'STATUS']] = data[['AGENCY NAME', 'CLASS CODE', 'CLASS TITLE', 'STATUS']].apply(enc.fit_transform)
    
    # Drop specified columns
    columns_to_drop = ['LAST NAME', 'FIRST NAME', 'MI', 'EMPLOY DATE']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1, inplace=True)


    # Model Training
    st.subheader("Model Training")

    # Select features and target variable
    X = data[['HRLY RATE', 'HRS PER WK', 'MONTHLY', 'EMPLOY YEAR']]
    y = data['ANNUAL']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    st.write("### Linear Regression")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)

    st.write(f"Training RMSE: {mean_squared_error(y_train, y_pred_train, squared=False)}")
    st.write(f"Test RMSE: {mean_squared_error(y_test, y_pred_test, squared=False)}")

    # Train Decision Tree model
    st.write("### Decision Tree")
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)

    y_pred_train = dt_model.predict(X_train)
    y_pred_test = dt_model.predict(X_test)

    st.write(f"Training RMSE: {mean_squared_error(y_train, y_pred_train, squared=False)}")
    st.write(f"Test RMSE: {mean_squared_error(y_test, y_pred_test, squared=False)}")

    # Train Random Forest model
    st.write("### Random Forest")
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)

    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    st.write(f"Training RMSE: {mean_squared_error(y_train, y_pred_train, squared=False)}")
    st.write(f"Test RMSE: {mean_squared_error(y_test, y_pred_test, squared=False)}")

    # Prediction Section
    st.subheader("Make Predictions")

    hrly_rate = st.number_input("Hourly Rate")
    hrs_per_wk = st.number_input("Hours Per Week")
    monthly = st.number_input("Monthly Salary")
    employ_year = st.number_input("Employment Year")
    employ_date = st.date_input("Employment Date")
    employ_year = employ_date.year

    model_choice = st.selectbox("Choose model for prediction", ("Linear Regression", "Decision Tree", "Random Forest"))

    if st.button("Predict"):
        if model_choice == "Linear Regression":
            prediction = lr_model.predict([[hrly_rate, hrs_per_wk, monthly, employ_year]])
        elif model_choice == "Decision Tree":
            prediction = dt_model.predict([[hrly_rate, hrs_per_wk, monthly, employ_year]])
        elif model_choice == "Random Forest":
            prediction = rf_model.predict([[hrly_rate, hrs_per_wk, monthly, employ_year]])
        
        st.write(f"Predicted Annual Salary: ${prediction[0]:.2f}")

# Instructions to run the app
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Upload the dataset.
2. Explore the data through the EDA section.
3. View the model training results.
4. Make predictions by entering the required features and choosing a model.
""")
