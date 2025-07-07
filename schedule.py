
import streamlit as st
import pandas as pd
import datetime

def session_panel():
    st.subheader("📋 Therapist Session Scheduler")

    if "schedule" not in st.session_state:
        st.session_state.schedule = []

    with st.form("add_session"):
        col1, col2 = st.columns(2)
        with col1:
            therapist = st.text_input("Therapist Name")
            session_time = st.time_input("Session Time")
        with col2:
            client = st.text_input("Client Name")
            duration = st.slider("Duration (hrs)", 0.5, 3.0, 1.0)
        submitted = st.form_submit_button("Add Session")

    if submitted and therapist and client:
        st.session_state.schedule.append({
            "Therapist": therapist,
            "Client": client,
            "Time": session_time.strftime("%H:%M"),
            "Duration": duration,
            "Date": datetime.date.today().strftime("%Y-%m-%d")
        })
        st.success(f"✅ Session added for {client} with {therapist} at {session_time.strftime('%H:%M')}")

    if st.session_state.schedule:
        st.write("📅 Scheduled Sessions")
        st.dataframe(pd.DataFrame(st.session_state.schedule))
