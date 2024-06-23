import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import parse
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# Function to load a saved model from a pickle file
def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model
# Helper function to handle user input for features
def preprocess_data(data):
    # Convert 'TRUE' and 'FALSE' strings to boolean values
    data = data.replace({'TRUE': True, 'FALSE': False})

    # Handle categorical variables (convert to category type)
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = data[col].astype('category')

    # Handle numerical variables (ensure numeric type)
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Handle any non-numeric values gracefully

    return data


# Helper function to handle user input for features
def get_user_input_features(processed_test_data):
    user_inputs = {}
    for feature in processed_test_data.columns[:-1]:  # Exclude the target column
        feature_type = processed_test_data[feature].dtype

        if feature_type == bool:
            user_input = st.checkbox(f"{feature} (True/False)")
        elif processed_test_data[feature].dtype.name == 'category':  # Check if categorical
            categories = processed_test_data[feature].cat.categories.tolist()
            user_input = st.selectbox(f"{feature}", categories)
        else:
            user_input = st.text_input(f"{feature}")

        user_inputs[feature] = user_input

    return user_inputs

# temp_dir = './temp'
# if not os.path.exists(temp_dir):
#     os.makedirs(temp_dir)
# Function to preprocess the uploaded data
def preprocess_data(data):
    # Assuming data preprocessing steps based on your model requirements
    # For example, handling missing values, encoding categorical variables, etc.
    return data


# Function to evaluate the model on test data
def evaluate_model(model, X_test, y_test):
    # Predict
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1


# Function to check if uploaded file is an Excel or CSV file
def check_file_type(uploaded_file):
    return uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith(
        '.csv')


# Function to load Excel or CSV file and handle date columns
def load_file(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)

    # Check for columns named 'Date' (case insensitive)
    date_columns = [col for col in data.columns if col.lower() == 'date']

    for col in date_columns:
        try:
            # Attempt to parse date with different formats
            data[col] = pd.to_datetime(data[col], dayfirst=True, errors='coerce')
            if data[col].notnull().any():
                break  # Stop if successfully parsed
        except ValueError:
            try:
                data[col] = pd.to_datetime(data[col], format='%m/%d/%Y', errors='coerce')
                if data[col].notnull().any():
                    break  # Stop if successfully parsed
            except ValueError:
                try:
                    data[col] = pd.to_datetime(data[col], format='%d-%m-%Y', errors='coerce')
                    if data[col].notnull().any():
                        break  # Stop if successfully parsed
                except ValueError:
                    st.error(f"Unable to parse '{col}' as date. Please check the format.")

    return data


# Function to plot line graph
def plot_line_graph(data, date_feature, categorical_feature):
    plt.figure(figsize=(10, 6))

    # Check if date_feature is a date column
    if pd.api.types.is_datetime64_any_dtype(data[date_feature]):
        # Date interval selection
        st.subheader("Date Interval Selection")
        min_date = data[date_feature].min()
        max_date = data[date_feature].max()
        start_date = st.date_input("Start Date", min_value=min_date.date(), max_value=max_date.date(),
                                   value=min_date.date())
        end_date = st.date_input("End Date", min_value=min_date.date(), max_value=max_date.date(),
                                 value=max_date.date())

        # Convert start_date and end_date to Pandas Timestamps
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter data by date interval
        filtered_data = data[(data[date_feature] >= start_date) & (data[date_feature] <= end_date)]

        # Check unique values in categorical_feature
        unique_values = filtered_data[categorical_feature].unique()

        if len(unique_values) <= 6:
            # Plot line graph for each unique value
            for value in unique_values:
                subset = filtered_data[filtered_data[categorical_feature] == value]
                sns.lineplot(data=subset, x=date_feature, y=categorical_feature, label=str(value))
            plt.legend()
            st.pyplot(plt)
        else:
            st.error(
                "You selected a feature with more than 6 unique values. Please choose a feature with fewer unique values for line graph.")
    else:
        st.error("The selected feature for line graph should be a date column.")


# Function to plot other graph types (scatter, bar, heatmap, pie)
def plot_graph(data, feature_x, feature_y, plot_type):
    plt.figure(figsize=(10, 6))

    if plot_type == 'Scatter Plot':
        sns.scatterplot(data=data, x=feature_x, y=feature_y)
        plt.title(f'Scatter Plot of {feature_x} vs {feature_y}')
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        st.pyplot(plt)

    elif plot_type == 'Bar Plot':
        sns.countplot(data=data, x=feature_x, hue=feature_y)
        plt.title(f'Bar Plot of {feature_x} by {feature_y}')
        plt.xlabel(feature_x)
        plt.ylabel('Count')
        plt.legend(title=feature_y)
        st.pyplot(plt)

    elif plot_type == 'Heatmap':
        # Check if both features are numeric or categorical
        if pd.api.types.is_numeric_dtype(data[feature_x]) and pd.api.types.is_numeric_dtype(data[feature_y]):
            sns.heatmap(data[[feature_x, feature_y]].corr(), annot=True, cmap='coolwarm')
            plt.title(f'Heatmap of {feature_x} vs {feature_y}')
            st.pyplot(plt)
        elif pd.api.types.CategoricalDtype(data[feature_x]) and pd.api.types.is_categorical_dtype(data[feature_y]):
            st.error("Heatmap requires both selected features to be numeric or both to be categorical.")
        else:
            st.error("Heatmap requires both selected features to be numeric or both to be categorical.")

    elif plot_type == 'Pie Chart':
        st.subheader(f"Pie Chart of {feature_x} by {feature_y}")
        unique_values = data[feature_x].unique()
        value_selection = st.selectbox(f"Select {feature_x} value", unique_values)
        if value_selection:
            subset = data[data[feature_x] == value_selection]
            plt.figure(figsize=(6, 6))
            plt.pie(subset[feature_y].value_counts(), labels=subset[feature_y].value_counts().index, autopct='%1.1f%%',
                    startangle=140)
            plt.title(f'{feature_x} {value_selection} distribution by {feature_y}')
            st.pyplot(plt)


# Streamlit app
st.title("Interactive Dashboard")

tabs = st.tabs(["Key Insights","Model Testing"])

with tabs[0]:
    st.header("Graph Key Insights")

    st.subheader("Graph 1")
    uploaded_file1 = st.file_uploader("Browse Excel/CSV File", key="file1")
    if uploaded_file1 and check_file_type(uploaded_file1):
        data1 = load_file(uploaded_file1)
        st.write("File loaded successfully!")

        feature1_x = st.selectbox("Select First Feature", data1.columns, key="feature1_x")
        feature1_y = st.selectbox("Select Second Feature", data1.columns, key="feature1_y")

        if feature1_x and feature1_y:
            plot_type1 = st.selectbox("Select Plot Type",
                                      ["Scatter Plot", "Line Plot", "Bar Plot", "Heatmap", "Pie Chart"],
                                      key="plot_type1")

            if plot_type1 == 'Line Plot':
                plot_line_graph(data1, feature1_x, feature1_y)
            else:
                if st.button("Plot Graph 1", key="plot1"):
                    plot_graph(data1, feature1_x, feature1_y, plot_type1)

    st.subheader("Graph 2")
    uploaded_file2 = st.file_uploader("Browse Excel/CSV File", key="file2")
    if uploaded_file2 and check_file_type(uploaded_file2):
        data2 = load_file(uploaded_file2)
        st.write("File loaded successfully!")

        feature2_x = st.selectbox("Select First Feature", data2.columns, key="feature2_x")
        feature2_y = st.selectbox("Select Second Feature", data2.columns, key="feature2_y")

        if feature2_x and feature2_y:
            plot_type2 = st.selectbox("Select Plot Type",
                                      ["Scatter Plot", "Line Plot", "Bar Plot", "Heatmap", "Pie Chart"],
                                      key="plot_type2")

            if plot_type2 == 'Line Plot':
                plot_line_graph(data2, feature2_x, feature2_y)
            else:
                if st.button("Plot Graph 2", key="plot2"):
                    plot_graph(data2, feature2_x, feature2_y, plot_type2)

with tabs[1]:
    st.header("Test Your Model")

    # Upload model pickle file
    st.subheader("Upload Model")
    model_file = st.file_uploader("Upload Model (.pickle or .pkl.txt)", type=["pickle", "pkl"])

    if model_file:
        # Save uploaded file to a temporary location
        temp_file_path = os.path.join("./temp", model_file.name)
        with open(temp_file_path, 'wb') as f:
            f.write(model_file.read())
        st.success(f"Saved model file to {temp_file_path}")

        # Load the model from the saved file
        loaded_model = load_model(temp_file_path)
        st.success("Model loaded successfully!")
        # Upload test data file
        st.subheader("Upload Test Data")
        test_file = st.file_uploader("Upload Test Data (CSV or Excel)", type=["csv", "xlsx"])

        if test_file:
            test_data = pd.read_csv(test_file) if test_file.name.endswith('.csv') else pd.read_excel(test_file)
            st.write("Test data loaded successfully!")
            st.subheader("Select Target Column")
            target_column = st.selectbox("Select Target Column", test_data.columns.tolist())
            # Preprocess test data (adjust according to your preprocessing steps)
            # For example, handling missing values, encoding categorical variables, etc.
            processed_test_data = preprocess_data(test_data)

            # Assume X_test and y_test are prepared for evaluation
            X_test = processed_test_data.drop(columns=[target_column])
            y_test = processed_test_data[target_column]

            # Evaluate model
            accuracy, precision, recall, f1 = evaluate_model(loaded_model, X_test, y_test)

            # Display evaluation metrics
            st.subheader("Evaluation Metrics")
            metrics = {
                'Accuracy': accuracy,
                'Precision (Weighted)': precision,
                'Recall (Weighted)': recall,
                'F1 Score (Weighted)': f1
            }
            st.write(pd.DataFrame.from_dict(metrics, orient='index', columns=['Score']))
st.header("Test Your Model")

# Upload model pickle file
st.subheader("Predict using  Model")
model_file = st.file_uploader("Upload Model (.pickle)", type=["pickle"])

if model_file:
    # Save uploaded file to a temporary location
    temp_file_path = os.path.join("./temp", model_file.name)
    with open(temp_file_path, 'wb') as f:
        f.write(model_file.read())
    st.success(f"Saved model file to {temp_file_path}")

    # Load the model from the saved file
    loaded_model = joblib.load(temp_file_path)
    st.success("Model loaded successfully!")

    # Upload test data file
    st.subheader("Upload Test Data")
    test_file = st.file_uploader("Upload Test Data (CSV or Excel)", type=["csv", "xlsx"])

    if test_file:
        test_data = pd.read_csv(test_file) if test_file.name.endswith('.csv') else pd.read_excel(test_file)
        st.write("Test data loaded successfully!")

        st.subheader("Select Target Column")
        target_column = st.selectbox("Select Target Column", test_data.columns.tolist())

        # Preprocess test data (adjust according to your preprocessing steps)
        processed_test_data = preprocess_data(test_data)

        # Display form for user input features
        st.subheader("Enter Feature Values")
        num_features = len(processed_test_data.columns) - 1  # Excluding the target column
        if num_features <= 3:
            user_inputs = get_user_input_features(processed_test_data)
        else:
            st.write(f"This model has {num_features} features. Please enter values for each feature.")
            user_inputs = get_user_input_features(processed_test_data)

        # Prepare the input data for prediction
        if len(user_inputs) == num_features:
            input_data = pd.DataFrame([user_inputs], columns=processed_test_data.columns[:-1])
            st.write("Input Features:")
            st.write(input_data)

            # Make predictions
            prediction = loaded_model.predict(input_data)
            st.subheader("Prediction")
            st.write(f"The predicted {target_column} is: {prediction[0]}")
