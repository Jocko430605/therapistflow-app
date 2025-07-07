
import streamlit as st
import pandas as pd
import plotly.express as px

def calendar_view():
    st.subheader("📅 Calendar View")

    if "schedule" not in st.session_state or not st.session_state.schedule:
        st.info("No sessions scheduled yet. Add some from the 'Schedule' tab.")
        return

    df = pd.DataFrame(st.session_state.schedule)
    df["Start"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["End"] = df["Start"] + pd.to_timedelta(df["Duration"], unit="h")

    fig = px.timeline(
        df, x_start="Start", x_end="End", y="Therapist",
        color="Client", title="Therapist Schedule Timeline",
        hover_data=["Client", "Duration"]
    )

    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
