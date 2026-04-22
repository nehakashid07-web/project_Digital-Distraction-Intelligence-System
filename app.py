import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# Prediction Function
# -------------------------------
def predict_distraction(age, screen, social, notif, switch, sleep, work, scaler_path, model_path):
    try:
        # Load scaler
        with open(scaler_path, 'rb') as file1:
            scaler = pickle.load(file1)

        # Load model
        with open(model_path, 'rb') as file2:
            model = pickle.load(file2)

        # Input data
        dct = {
            'Age': [age],
            'Daily_Screen_Time': [screen],
            'Social_Media_Time': [social],
            'Notifications': [notif],
            'App_Switches': [switch],
            'Sleep_Hours': [sleep],
            'Work_Hours': [work]
        }

        x_new = pd.DataFrame(dct)

        # Scale
        xnew_pre = scaler.transform(x_new)

        # Prediction
        pred = model.predict(xnew_pre)
        probs = model.predict_proba(xnew_pre)
        max_prob = np.max(probs)

        return pred, max_prob

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📱 Digital Distraction Intelligence System")

# Add Image (you can replace URL or use local file)
st.image("https://tse3.mm.bing.net/th/id/OIP.3fDYnb9xBGkqg0AiQyxRIwHaEO?pid=Api&P=0&h=180", width=150)

st.subheader("Enter User Data")

# Inputs
age = st.number_input("Age", 16, 40, 22)
screen = st.slider("Daily Screen Time (hrs)", 1, 12, 6)
social = st.slider("Social Media Time (hrs)", 1, 8, 3)
notif = st.slider("Notifications per Day", 10, 200, 80)
switch = st.slider("App Switches", 5, 120, 40)
sleep = st.slider("Sleep Hours", 3, 10, 7)
work = st.slider("Work Hours", 1, 12, 6)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    scaler_path = "notebook/scaler.pkl"
    model_path = "notebook/model.pkl"

    pred, max_prob = predict_distraction(
        age, screen, social, notif, switch, sleep, work,
        scaler_path, model_path
    )

    if pred is not None:
        if pred[0] == 1:
            st.error("⚠️ Highly Distracted")
        else:
            st.success("✅ Not Distracted")

        st.subheader(f"Confidence: {max_prob:.2f}")
        st.progress(float(max_prob))


# -------------------------------
# Graph Section
# -------------------------------
st.subheader("📊 Data Visualization")

try:
    df = pd.read_csv("Digital_Distraction.csv")

    # Graph 1: Screen Time vs Distraction
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["Daily_Screen_Time"], df["Distraction_Level"])
    ax1.set_xlabel("Screen Time")
    ax1.set_ylabel("Distraction Level")
    ax1.set_title("Screen Time vs Distraction")
    st.pyplot(fig1)

    # Graph 2: Notifications Histogram
    fig2, ax2 = plt.subplots()
    ax2.hist(df["Notifications"])
    ax2.set_title("Notifications Distribution")
    st.pyplot(fig2)

except:
    st.warning("Dataset not found. Please keep Digital_Distraction.csv in same folder.")