import streamlit as st
import subprocess
import requests  # Importing requests for API calls
from PIL import Image  # Uncomment if using an image
import json

# Page Configuration
st.set_page_config(page_title="AI-Powered Job Ad Generator", page_icon="📄", layout="wide")

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

# Step 1: Role Selection
st.header("Position-Specific Details")
selected_role = st.selectbox(
    "Select the Position:",
    ["Data Scientist", "Data Analyst", "Software Engineer", "Other"]
)

if selected_role == "Data Scientist":
    specialization = st.multiselect(
        "Choose Specializations:",
        ["Machine Learning", "Statistics", "Deep Learning", "Data Visualization"]
    )
    tools = st.multiselect(
        "Choose Tools/Technologies:",
        ["Python", "R", "TensorFlow", "PyTorch", "SQL", "Pandas", "NumPy", "Jupyter Notebooks"]
    )
elif selected_role == "Data Analyst":
    focus_area = st.radio(
        "Primary Focus:",
        ["Data Cleaning", "Reporting", "Business Analysis"]
    )
    software = st.multiselect(
        "Choose Key Software Tools:",
        ["Excel", "Power BI", "Tableau", "SQL", "Python", "R", "Google Analytics"]
    )
elif selected_role == "Software Engineer":
    tech_stack = st.multiselect(
        "Select Preferred Tech Stack:",
        ["Java", "C++", "Python", "JavaScript", "Ruby", "AWS", "Docker", "Kubernetes"]
    )
else:
    custom_details = st.text_area("Describe the Role in Detail:")

# Step 2: Work Model and Compensation
st.header("Work Environment and Compensation")
work_model = st.selectbox(
    "Select Work Model:",
    ["Onsite", "Remote", "Hybrid"]
)
salary_range = st.slider("Expected Salary Range (in USD):", 30000, 200000, (50000, 100000))

# Step 3: Job Ad Details
st.header("Additional Job Details")
job_title = st.text_input("Job Title:")
company = st.text_input("Company Name:")
location = st.text_input("Location:")
responsibilities = st.text_area("Key Responsibilities:")
requirements = st.text_area("Key Requirements:")
benefits = st.text_area("Benefits Offered:")
target_audience = st.text_area("Describe the Target Audience:")

# Generate Job Ad Button
if st.button("Generate Job Advertisement"):
    if all([job_title, company, location, responsibilities, requirements, benefits, target_audience]):
        # Prepare the prompt for the model
        prompt = f"""
        Create a job advertisement for the following position:

        **Job Title:** {job_title}
        **Company:** {company}
        **Location:** {location}

        **Responsibilities:**
        {responsibilities}

        **Requirements:**
        {requirements}

        **Benefits:**
        {benefits}

        **Target Audience:**
        {target_audience}
        """

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
                    file_name=f"{job_title}_Job_Ad.txt",
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
