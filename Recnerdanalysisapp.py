import streamlit as st
import subprocess
import requests  # Importing requests for API calls
from PIL import Image  # Uncomment if using an image
import json

# Page Configuration
st.set_page_config(page_title="AI-Powered Job Ad Generator", page_icon="📄", layout="wide")

# Sidebar for Navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Home", "About Us", "Impressum"])

if selected_page == "About Us":
    st.title("About Us")
    st.write("We are a team of HR professionals and data scientists dedicated to helping you craft the perfect job description. Our goal is to make the hiring process more efficient and effective for both recruiters and candidates.")
elif selected_page == "Impressum":
    st.title("Impressum")
    st.write("This is the legal information about our company. Here you can put the details required by law.")
elif selected_page == "Home":
    # App Title
    st.title("Recruitment Need Analysis App")
    st.subheader("Discover important aspects of the role you want to fill and identify the skills and qualifications you need.")

    # Introduction
    st.markdown("""
        This app assists HR professionals in creating comprehensive job descriptions 
        by guiding them through interactive questions. Answer a series of prompts to 
        generate a formatted job description tailored to your needs.
    """)

    # Include an Image on the Front Page
    # Uncomment the following lines if you want to display the image
    # try:
    #     image = Image.open('images/iceberg-model.jpg')
    #     st.image(image, caption="Iceberg Model of System Thinking", use_column_width=True)
    # except FileNotFoundError:
    #     st.warning("Image not found. Ensure the file path is correct.")

    # Step-by-Step Job Analysis
    st.header("Job Title")
    job_title = st.text_input("Enter the Job Title:")

    if st.button("Next") and job_title:
        st.session_state['job_title'] = job_title
        st.experimental_rerun()

    if 'job_title' in st.session_state:
        # Display the selected job title and proceed with questions
        st.sidebar.header("Summary of Inputs")
        st.sidebar.write(f"**Job Title:** {st.session_state['job_title']}")
        
        # Step 2: Important Job Questions (Using LLM to determine the next question)
        st.header("Key Details")
        prompt = f"What are the most important details to ask for when creating a job description for a {st.session_state['job_title']}?"

        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": "koesn/dolphin-llama3-8b",
                    "prompt": prompt,
                    "num_ctx": 8192
                },
                stream=True  # Enable streaming response
            )   
            response.raise_for_status()

            # Process the streaming response for suggested questions
            questions = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    data = json.loads(chunk)  # Convert the string to a dictionary
                    questions += data.get("response", "")
                    if data.get("done", False):
                        break

            if questions.strip():
                st.write("Please answer the following questions to describe the job role:")
                questions_list = questions.split('\n')
                for idx, question in enumerate(questions_list, start=1):
                    answer = st.text_input(f"{idx}. {question}", key=f'question_{idx}')
                    if answer:
                        st.session_state[f'question_{idx}'] = answer
                    st.sidebar.write(f"**{question}:** {st.session_state.get(f'question_{idx}', 'Not answered yet')}")
            else:
                st.error("The model did not return any questions. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

        # Generate Job Ad Button
        if st.button("Generate Job Advertisement"):
            if all(st.session_state.get(f'question_{idx}', '').strip() for idx in range(1, len(questions_list) + 1)):
                # Prepare the prompt for the model
                prompt = f"""
                Create a job advertisement for the following position:

                **Job Title:** {st.session_state['job_title']}

                """
                for idx, question in enumerate(questions_list, start=1):
                    prompt += f"**{question}:** {st.session_state.get(f'question_{idx}', '')}\n"

                # Call the Ollama API with streaming response handling
                try:
                    response = requests.post(
                        "http://127.0.0.1:11434/api/generate",
                        json={
                            "model": "koesn/dolphin-llama3-8b",
                            "prompt": prompt,
                            "num_ctx": 8192
                        },
                        stream=True  # Enable streaming response
                    )   
                    response.raise_for_status()

                    # Process the streaming response
                    job_ad = ""
                    for line in response.iter_lines():
                        if line:
                            chunk = line.decode("utf-8")
                            data = json.loads(chunk)  # Convert the string to a dictionary
                            job_ad += data.get("response", "")
                            if data.get("done", False):
                                break

                    if job_ad.strip():
                        st.subheader("Generated Job Advertisement:")
                        st.markdown(job_ad)
                        st.download_button(
                            label="Download Job Advertisement",
                            data=job_ad,
                            file_name=f"{st.session_state['job_title']}_Job_Ad.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("The model did not return any content. Please try again.")
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please fill in all fields before generating the job advertisement.")

# Corrected: Removed undefined main() function call.
# The script is already structured properly for Streamlit.
# The __name__ == "__main__" block is not needed for Streamlit apps.
