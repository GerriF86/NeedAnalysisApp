import json
#import faiss
import numpy as np
import re
import os
import pandas as pd
from tqdm import tqdm, trange
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import streamlit as st
import logging
from langchain_groq import ChatGroq
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
NUMBER_OF_MESSAGES_TO_DISPLAY = 20
API_DOCS_URL = "https://docs.streamlit.io/library/api-reference"

# Streamlit Page Configuration
st.set_page_config(
    page_title="Power2Rec - Job Description Generator and Streamlit Assistant",
    page_icon="imgs/avatar_streamly.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/GerriF86/nlu",
        "Report a bug": "https://github.com/GerriF86/nlu",
        "About": """
            ## Power2Rec Job Description Generator and Streamlit Assistant
            ### Powered using llamayxz

            **GitHub**: https://github.com/GerriF86/

            The AI Assistant, Power2Rec, aims to assist in generating job descriptions, 
            and provide Streamlit app development guidance.
        """
    }
)

# Streamlit Title
st.title("Power2Rec - Job Description Generator & Streamlit Assistant")

# ## Step 0: Load the JSON Files
salaries = pd.read_json('data/json/salaries')
resumes = pd.read_json('data/json/Entity Recognition in Resumes.json')
it_jobs = pd.read_json('data/json/IT Job Desc Annotated Detailed.json')

# # Job Description Generator

# ## Step 1: Initialization

class JobDescriptionGenerator:
    def __init__(self):
        # Initialize data with default values
        self.data = {
            "Position": "N/A",
            "Specialization": "N/A",
            "Work Model": "N/A",
            "Remote Location": "N/A",
            "Remote Timezone": "N/A",
            "Technical Equipment": "N/A",
            "Remote Percentage": "N/A",
            "BI Tools": "N/A",
            "Required Tools": "N/A",
            "Visualization Tools": "N/A",
            "Statistical Methods": "N/A",
            "Big Data Tools": "N/A",
            "Experience Level": "N/A",
            "Leadership Skills": "None",
            "Educational Requirements": "None",
            "Project Leadership": "No",
            "Compensation": "N/A",
            "Home Office Allowance": "None",
            "Remote Benefits": "None",
            "Additional Benefits": "None"
        }
        # Load pre-trained model for embedding generation
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize FAISS index for similarity search
        self.index = None
        self.documents = []

    # ## Step 2: Data Cleaning and Preprocessing
    def clean_and_preprocess_dataset(self, dataset):
        filtered_data = dataset[dataset['job_description'].notna()]
        job_descriptions = filtered_data['job_description'].str.strip().str.lower().tolist()
        unique_job_descriptions = list(set(job_descriptions))
        processed_descriptions = [re.sub(r'[^a-zA-Z0-9\s]', '', desc) for desc in unique_job_descriptions]
        return processed_descriptions

    # ## Step 3: Loading Dataset and Building FAISS Index
    def load_dataset_and_build_index(self, dataset):
        st.write("Cleaning and preprocessing dataset...")
        job_descriptions = self.clean_and_preprocess_dataset(dataset)
        self.documents = job_descriptions

        st.write("Generating embeddings...")
        embeddings = []
        for desc in trange(len(job_descriptions), desc="Embedding job descriptions"):
            embedding = self.model.encode(job_descriptions[desc])
            embeddings.append(embedding)
        embeddings = np.array(embeddings)

        st.write("Building FAISS index...")
        if embeddings.size > 0:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        else:
            st.write("Error: No embeddings found to build the FAISS index.")

    # ## Step 4: Finding Similar Job Descriptions
    def find_similar_jobs(self, query, k=3):
        if self.index is None:
            st.write("Error: FAISS index is not initialized. Please load the dataset and build the index first.")
            return []

        st.write("Generating embedding for the query...")
        query_embedding = self.model.encode([query])

        st.write("Searching for similar job descriptions...")
        _, indices = self.index.search(np.array(query_embedding), k)

        return [self.documents[idx] for idx in indices[0]]

# Create an instance of the JobDescriptionGenerator
generator = JobDescriptionGenerator()

# Load dataset and build FAISS index
if st.button("Build Job Description Index"):
    with st.spinner("Building FAISS index..."):
        generator.load_dataset_and_build_index(it_jobs)
    st.success("FAISS index built successfully!")

# Collect job information and generate description
if st.button("Generate Job Description"):
    with st.spinner("Collecting job information and generating description..."):
        generator.run()
    st.success("Job description generated!")

# Find similar job descriptions
query = st.text_input("Enter a job role or description to find similar roles:")
if st.button("Find Similar Jobs"):
    if query:
        with st.spinner("Searching for similar jobs..."):
            similar_jobs = generator.find_similar_jobs(query)
        st.write("\n--- Similar Job Descriptions ---")
        for job in similar_jobs:
            st.write(job)
    else:
        st.warning("Please enter a job role or description to find similar roles.")

# Additional sections of the Streamlit assistant to display updates
def display_streamlit_updates():
    with st.expander("Streamlit 1.36 Announcement", expanded=False):
        st.markdown("For more details on this version, check out the [Streamlit Forum post](https://docs.streamlit.io/library/changelog#version).")

st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Mode:", options=["Job Description Generator", "Streamlit Updates"], index=0)

if mode == "Streamlit Updates":
    display_streamlit_updates()