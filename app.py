import streamlit as st
import pandas as pd
import pickle

# Load model
with open("churn_model.pkl", "rb") as f:
    model, model_columns = pickle.load(f)

# Page config
st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📉 Customer Churn Prediction")
st.caption("Predict whether a customer is likely to churn")

st.divider()

st.subheader("🔢 Customer Information")

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    tenure = st.number_input("Tenure (months)", 0, 60, 12)
    usage = st.number_input("Usage Frequency", 0, 10, 3)

with col2:
    total_spend = st.number_input("Total Spend", 0, 100000, 5000)
    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

st.subheader("📞 Service Details")

col3, col4 = st.columns(2)

with col3:
    contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])
    payment_delay = st.selectbox("Payment Delay", ["On time", "Slight Delay", "Long Delay"])

with col4:
    support_calls = st.selectbox("Support Calls", ["Low", "Medium", "High"])
st.divider()

if st.button("🔮 Predict Churn"):
    input_data = {
        'Age': age,
        'Tenure': tenure,
        'Usage Frequency': usage,
        'Total Spend': total_spend,
        'Gender': gender,
        'Subscription Type': subscription,
        'Contract Length': contract,
        'PaymentDelay_Bin': payment_delay,
        'Support Calls': support_calls
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"❌ Customer likely to churn\n\n**Probability:** {probability:.2%}")
    else:
        st.success(f"✅ Customer likely to stay\n\n**Probability:** {probability:.2%}")
