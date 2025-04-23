import streamlit as st
from qa_utils import ingest_urls, answer_question

st.set_page_config(page_title="Web Content Q&A Tool", layout="centered")
st.title(" Web Content Q&A Tool")
st.caption("Ask questions using only the content from the webpages you provide.")

# Input section
st.markdown("### 🔗 Step 1: Enter URLs")
url_input = st.text_area("Enter one or more URLs (comma-separated):", height=100)
urls = [url.strip() for url in url_input.split(",") if url.strip()]

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# Ingest
if st.button("📥 Ingest Content"):
    if not urls:
        st.warning("Please enter at least one valid URL.")
    else:
        with st.spinner("Scraping and indexing content..."):
            try:
                chunks, embeddings = ingest_urls(urls)
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.success("Content ingested successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

# Ask question
if st.session_state.chunks and st.session_state.embeddings is not None:
    st.markdown("### Step 2: Ask a Question")
    question = st.text_input("Ask a question based on the above content:")
    
    if question:
        with st.spinner("Searching relevant content..."):
            try:
                answer = answer_question(question, st.session_state.chunks, st.session_state.embeddings)
                st.markdown("### Answer")
                st.success(answer)
            except Exception as e:
                st.error(f" Error: {e}")
else:
    st.info("📝 Ingest content before asking a question.")
