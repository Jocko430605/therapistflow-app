
import streamlit as st
from schedule import session_panel
from calendar_ui import calendar_view
from recovery_guard import burnout_chart

st.set_page_config(page_title="TherapistFlow", layout="wide")
st.title("🧠 TherapistFlow Assistant")

tabs = st.tabs(["📋 Schedule", "📅 Calendar", "🔥 Recovery AI"])

with tabs[0]:
    session_panel()

with tabs[1]:
    calendar_view()

with tabs[2]:
    burnout_chart()
