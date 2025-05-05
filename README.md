# GenAI IMDB SQL POC

## About
A project where you can ask plain English questions about movies, and the system generates SQL queries dynamically!

## Setup
```bash
pip install -r requirements.txt
python3 embedding_utils.py  # Build FAISS Index
streamlit run app.py