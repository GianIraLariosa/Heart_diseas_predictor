import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = load_model("random_forest_model (2).h5")

# Load the dataset
data = pd.read_csv("heart.csv")

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=['cp', 'restecg', 'thal'])

# Split features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to preprocess input data
def preprocess_input(df):
    arr = df.values
    scaled_data = arr.reshape(1, -1)
    arr = scaler.transform(scaled_data)
    return arr

# Main function for Streamlit app
def main():
    st.title("Heart Disease Prediction")

    # Create input fields for user input
    age = st.number_input("Age")
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)")
    chol = st.number_input("Serum Cholesterol (mg/dl)")
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved")
    exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest")
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Flouroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    # Map selected sex to binary value
    sex_binary = 0 if sex == "Male" else 1

    fbs_binary = 0 if fbs == "False" else 1

    exang_binary = 0 if exang == "No" else 1

    # Set cp binary indicator variables based on user input
    cp_0 = 1 if cp == 0 else 0
    cp_1 = 1 if cp == 1 else 0
    cp_2 = 1 if cp == 2 else 0
    cp_3 = 1 if cp == 3 else 0

    # Set restecg binary indicator variables based on user input
    restecg_0 = 1 if restecg == 0 else 0
    restecg_1 = 1 if restecg == 1 else 0
    restecg_2 = 1 if restecg == 2 else 0

    # Set thal binary indicator variables based on user input
    thal_0 = 1 if thal == 0 else 0
    thal_1 = 1 if thal == 1 else 0
    thal_2 = 1 if thal == 2 else 0
    thal_3 = 1 if thal == 3 else 0

    # Create input DataFrame
    if st.button("Predict"):
        input_data = pd.DataFrame({
            "age": [age],
            "sex": [sex_binary],  # Use sex_binary instead of sex
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs_binary],
            "thalach": [thalach],
            "exang": [exang_binary],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "cp_0": [cp_0],
            "cp_1": [cp_1],  # Set to True or False based on user input
            "cp_2": [cp_2],  # Set to True or False based on user input
            "cp_3": [cp_3],  # Set to True or False based on user input
            "restecq_0": [restecg_0],
            "restecg_1": [restecg_1],  # Set to True or False based on user input
            "restecg_2": [restecg_2],  # Set to True or False based on user input
            "thal_0": [thal_0],
            "thal_1": [thal_1],  # Set to True or False based on user input
            "thal_2": [thal_2],  # Set to True or False based on user input
            "thal_3": [thal_3],  # Set to True or False based on user input
        })

        processed_data = preprocess_input(input_data)

        # Make predictions
        prediction = model.predict(processed_data)
        result = "Heart Disease Present" if prediction > 0.5 else "No Heart Disease"

        # Display prediction result
        st.write("Prediction:", result)
        st.write()

if __name__ == "__main__":
    main()
