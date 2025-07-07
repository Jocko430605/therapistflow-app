
import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

project = os.getenv("PROJECT_ID")
location = os.getenv("REGION")
model = os.getenv("VERTEX_MODEL_ID")

aiplatform.init(project=project, location=location)

def summarize_schedule():
    if "schedule" not in st.session_state or not st.session_state.schedule:
        return "No sessions scheduled yet."

    df = pd.DataFrame(st.session_state.schedule)
    df["Duration"] = pd.to_numeric(df["Duration"])
    summary = df.groupby("Therapist")["Duration"].sum().reset_index()
    return summary.to_dict(orient="records")

def query_vertex(prompt_text):
    endpoint = aiplatform.TextEndpoint(endpoint_name=model)
    response = endpoint.predict([prompt_text])
    return response.predictions[0]

def burnout_chart():
    st.subheader("🔥 Burnout Risk + Recovery Suggestions")

    workload = summarize_schedule()
    if workload == "No sessions scheduled yet.":
        st.info(workload)
        return

    prompt_template = PromptTemplate(
        input_variables=["summary"],
        template="""
A therapist assistant app monitors weekly workloads. Here is a summary:
{summary}

Based on this data, recommend individualized recovery strategies for overloaded therapists. Include rest planning, session pacing, and check-in frequency.
"""
    )

    prompt = prompt_template.format(summary=workload)
    response = query_vertex(prompt)

    st.markdown("### 💡 AI Recovery Suggestions")
    st.write(response)
