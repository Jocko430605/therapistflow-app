import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

st.set_page_config(page_title="TherapistFlow", page_icon="🧠", layout="wide")
st.title("🧠 TherapistFlow – Burnout Risk Tracker")

# Simulated daily burnout scores
today = datetime.date.today()
days = [today - datetime.timedelta(days=i) for i in range(30)][::-1]
scores = [round(0.4 + 0.5 * abs((15 - i) / 15) + 0.1 * (-1) ** i, 2) for i in range(30)]

df = pd.DataFrame({
    "Date": days,
    "Burnout Risk": scores
})

st.subheader("📆 Your 30-Day Burnout Curve")
fig = px.line(df, x="Date", y="Burnout Risk", markers=True, title="Daily Burnout Risk Score")
fig.update_yaxes(range=[0, 1], title="Risk (0 = Low, 1 = High)")
st.plotly_chart(fig)
