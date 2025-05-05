import streamlit as st
import sqlite3
from query_engine import generate_sql_or_followup
from typing import List

# Page settings
st.set_page_config(page_title="🧠 SQL + RAG Assistant", layout="wide")
st.title("🧠 Ask Questions in Plain English (SQL + Follow-ups + General)")

# Sidebar: Table selection
conn = sqlite3.connect('Northwind.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
available_tables = [t[0] for t in cursor.fetchall() if t[0] != "sqlite_sequence"]
conn.close()

st.sidebar.header("Select Table(s)")
selected_tables = st.sidebar.multiselect("Choose table(s) to query:", available_tables)
st.sidebar.write("✅ Selected Tables:", selected_tables)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not selected_tables and user_input.lower().startswith(("give", "show", "fetch", "select")):
        assistant_reply = "❌ Please select at least one table from the sidebar."
    else:
        history = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]

        try:
            intent, response = generate_sql_or_followup(user_input, selected_tables, history)

            if intent == "sql":
                assistant_reply = f"```sql\n{response.sql_query}\n```\n**Explanation:** {response.explanation}"
            elif intent == "followup":
                assistant_reply = response.explanation
            elif intent == "general":
                assistant_reply = response.explanation
            else:
                assistant_reply = f"⚠️ Unrecognized intent or error: {response}"

        except Exception as e:
            assistant_reply = f"❌ Exception occurred: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)