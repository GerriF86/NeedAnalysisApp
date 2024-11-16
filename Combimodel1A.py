import streamlit as st
import subprocess
import requests  # Importing requests for API calls
from PIL import Image  # Uncomment if using an image
import json
import folium
from streamlit_folium import st_folium

# Page Configuration
st.set_page_config(page_title="AI-Powered Job Ad Generator", page_icon="📄", layout="wide")

# Sidebar for Navigation
st.sidebar.title("")
selected_page = st.sidebar.radio("Go to", ["Home", "About Us", "Impressum"])

if selected_page == "About Us":
    st.title("About Us")
    st.write("We are Olivia and Gerrit, a team of HR professionals and data scientists dedicated to helping you craft the perfect job description. Our goal is to make the hiring process more efficient and effective for both recruiters and candidates.")
elif selected_page == "Impressum":
    st.title("Impressum")
    st.write("This is the legal information about our company. Here we will put the details required by law.")
elif selected_page == "Home":
    # Add an Image above the Header
    try:
        image = Image.open(r'F:\Capstone\Github Repo\WebApp\NeedAnalysisApp\imgs\iceberg-model-of-system-thinking-is-an-illustration-of-a-blue-mountain-and-presentation-this-theory-is-to-analyze-the-root-causes-of-events-hidden-underwater-for-developing-marketing-and-trend-vector.jpg')
        st.image(image, caption="AI-Powered Job Ad Generator", use_column_width=True)
    except FileNotFoundError:
        st.warning("Image not found. Please ensure the file path is correct.")

    # App Title
    st.title("Recruitment Need Analysis App")
    st.subheader("Discover important aspects of the role you want to fill and identify the skills and qualifications you need.")

    # Introduction
    st.markdown("""
        This app assists HR professionals in creating comprehensive job descriptions 
        by guiding them through interactive questions. Answer a series of prompts to 
        generate a formatted job description tailored to your needs.
    """)

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

        # Step 2: Company Details
        st.header("Company Details")
        company = st.text_input("Enter the Company Name:")
        email = st.text_input("Enter the Contact Email to be Published on Job Ad:")
        place_of_work = st.text_input("Enter the Place of Work:")
        start_date = st.date_input("Select the Start Date:")
        employment_type = st.selectbox("Select Employment Type:", ["Full-Time", "Part-Time", "Contract", "Temporary"])

        # Display Place of Work on OpenStreetMap
        if place_of_work:
            location_map = folium.Map(location=[50.1109, 8.6821], zoom_start=12)  # Example coordinates, can be modified
            folium.Marker([50.1109, 8.6821], popup=place_of_work).add_to(location_map)  # Add marker for place of work
            st_folium(location_map, width=700, height=500)

        if st.button("Next - Responsibilities") and company and email and place_of_work:
            st.session_state['company'] = company
            st.session_state['email'] = email
            st.session_state['place_of_work'] = place_of_work
            st.session_state['start_date'] = start_date
            st.session_state['employment_type'] = employment_type
            st.experimental_rerun()

    if 'company' in st.session_state:
        # Step 3: Responsibilities
        st.header("Role-Specific Responsibilities")
        suggested_responsibilities = [
            "Develop machine learning models",
            "Analyze large datasets",
            "Collaborate with cross-functional teams",
            "Create detailed reports"
        ]
        selected_responsibilities = st.multiselect("Select Suitable Responsibilities:", suggested_responsibilities)
        additional_responsibilities = st.text_area("Enter Additional Responsibilities (if any):")

        if st.button("Next - Requirements"):
            st.session_state['responsibilities'] = selected_responsibilities + [additional_responsibilities]
            st.experimental_rerun()

    if 'responsibilities' in st.session_state:
        # Step 4: Requirements
        st.header("Role-Specific Requirements")
        suggested_requirements = [
            "Bachelor's degree in Computer Science or related field",
            "3+ years of experience in data analysis",
            "Proficiency in Python and SQL",
            "Strong problem-solving skills"
        ]
        selected_requirements = st.multiselect("Select Suitable Requirements:", suggested_requirements)
        additional_requirements = st.text_area("Enter Additional Requirements (if any):")

        if st.button("Next - Benefits"):
            st.session_state['requirements'] = selected_requirements + [additional_requirements]
            st.experimental_rerun()

    if 'requirements' in st.session_state:
        # Step 5: Benefits
        st.header("Benefits")
        suggested_benefits = [
            "Health insurance",
            "Flexible working hours",
            "Remote work options",
            "Professional development opportunities"
        ]
        selected_benefits = st.multiselect("Select Suitable Benefits:", suggested_benefits)
        additional_benefits = st.text_area("Enter Additional Benefits (if any):")

        if st.button("Generate Job Advertisement"):
            st.session_state['benefits'] = selected_benefits + [additional_benefits]
            st.experimental_rerun()

    if 'benefits' in st.session_state:
        # Generate Job Ad
        st.header("Generated Job Advertisement")
        prompt = f"""
        Create a job advertisement for the following position:

        **Job Title:** {st.session_state['job_title']}
        **Company:** {st.session_state['company']}
        **Location:** {st.session_state['place_of_work']}
        **Start Date:** {st.session_state['start_date']}
        **Employment Type:** {st.session_state['employment_type']}

        **Responsibilities:**
        {', '.join(st.session_state['responsibilities'])}

        **Requirements:**
        {', '.join(st.session_state['requirements'])}

        **Benefits:**
        {', '.join(st.session_state['benefits'])}
        """

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
                # Summary of Inputs
                st.sidebar.header("Summary of Inputs")
                st.sidebar.write(f"**Job Title:** {st.session_state['job_title']}")
                st.sidebar.write(f"**Company:** {st.session_state['company']}")
                st.sidebar.write(f"**Location:** {st.session_state['place_of_work']}")
                st.sidebar.write(f"**Start Date:** {st.session_state['start_date']}")
                st.sidebar.write(f"**Employment Type:** {st.session_state['employment_type']}")
                st.sidebar.write(f"**Responsibilities:** {', '.join(st.session_state['responsibilities'])}")
                st.sidebar.write(f"**Requirements:** {', '.join(st.session_state['requirements'])}")
                st.sidebar.write(f"**Benefits:** {', '.join(st.session_state['benefits'])}")
            else:
                st.error("The model did not return any content. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
            