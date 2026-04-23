import streamlit as st
import os
import subprocess
import sys
import time
import webbrowser
import random

# Function to check if a file exists
def file_exists(file_name):
    return os.path.isfile(file_name)

# Function to check dependencies for a milestone
def check_dependencies(milestone_file):
    if milestone_file == "adata_ingestion&parsing.py":
        required_files = ["skills_list.txt"]
        for file in required_files:
            if not file_exists(file):
                return False, f"Missing required file: {file}"
        return True, None
    elif milestone_file == "app.py":
        return True, None  # No specific file dependencies for Milestone 2
    return False, "Invalid milestone file"

# Function to run a milestone script as a separate Streamlit process and open it
def run_milestone_script(file_name):
    try:
        # Check if the file exists
        if not file_exists(file_name):
            return False, f"❌ {file_name} not found in the current directory."
        
        # Check dependencies
        deps_ok, deps_error = check_dependencies(file_name)
        if not deps_ok:
            return False, deps_error
        
        # Assign a random available port between 8502 and 8600 to avoid conflicts
        port = random.randint(8502, 8600)
        cmd = [sys.executable, "-m", "streamlit", "run", file_name, "--server.port", str(port)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Give the subprocess time to start
        
        # Open the new tab with the dynamically assigned port
        webbrowser.open_new_tab(f"http://localhost:{port}")
        return True, f"✅ Launched {file_name} on http://localhost:{port}."
    except Exception as e:
        return False, f"❌ Error launching {file_name}: {str(e)}"

# Configure page
st.set_page_config(
    page_title="AI Skill Extractor Launcher",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="auto"
)

# Inject CSS
st.markdown("""
<style>
    :root {
        --primary: #ffc107;
        --secondary: #ffa000;
        --text-primary: #2c3e50;
    }
    .main .block-container {
        padding: 2rem;
    }
    .hero-section {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .hero-section h1 {
        font-size: 2.5rem;
        font-weight: 800;
    }
    .stButton>button {
        background: var(--primary);
        color: white;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: var(--secondary);
    }
    .stSelectbox div[data-baseweb="select"] {
        background: white;
        border: 1px solid var(--primary);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-section">
    <h1>🚀 AI Skill Extractor Launcher</h1>
    <p>Select a milestone to run the AI Skill Extraction application</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state with all required keys
if 'launched' not in st.session_state:
    st.session_state.launched = False
    st.session_state.launch_message = ""
    st.session_state.last_port = None

# Initial milestone selection prompt
milestone = st.radio(
    "Select Milestone to Run",
    ["Milestone 1: Basic Skill Extraction", "Milestone 2: Advanced Skill Extraction"],
    index=None,
    help="Choose a milestone to proceed."
)

if milestone:
    # Map selection to module details
    milestone_configs = {
        "Milestone 1: Basic Skill Extraction": {
            "file_name": "adata_ingestion&parsing.py",
            "description": "Basic skill extraction with file ingestion, text cleaning, and rule-based skill detection."
        },
        "Milestone 2: Advanced Skill Extraction": {
            "file_name": "app.py",
            "description": "Advanced NLP-powered skill extraction with custom NER, semantic analysis, and visualizations."
        }
    }

    selected_config = milestone_configs[milestone]

    # Display description
    st.markdown(f"**Description**: {selected_config['description']}")

    # Run button
    if st.button("🚀 Launch Application", type="primary"):
        if not st.session_state.launched:
            with st.spinner(f"Launching {milestone}..."):
                success, message = run_milestone_script(selected_config['file_name'])
                st.session_state.launched = success
                st.session_state.launch_message = message
                if success:
                    # Extract port from message and update last_port
                    port_start = message.find("http://localhost:") + len("http://localhost:")
                    port_end = message.find(".", port_start)
                    st.session_state.last_port = message[port_start:port_end] if port_start != -1 and port_end != -1 else None
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.info(f"Application already launched: {st.session_state.launch_message}")

    # Display launch status with link to open if not automatic
    if st.session_state.launched and st.session_state.last_port:
        st.markdown(f"**Status**: {st.session_state.launch_message}")
        st.markdown(f"[Click here to open](http://localhost:{st.session_state.last_port}) if the tab didn’t open automatically.")

    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 Instructions
    1. Ensure `adata_ingestion&parsing.py` and `app.py` are in the same directory as this script.
    2. For Milestone 1, ensure `skills_list.txt` is present in the directory.
    3. Install required dependencies:
       - For Milestone 1: `pip install streamlit pandas python-docx pdfplumber`
       - For Milestone 2: `pip install streamlit spacy sentence-transformers pandas numpy plotly sklearn python-docx pdfplumber`
       - Optional: `python -m spacy download en_core_web_sm` for Milestone 2
    4. Select a milestone and click 'Launch Application' to run.
    5. The selected app should open in a new browser tab. If it fails, check the error message or terminal output.
    6. If the app redirects back to this page, try running `streamlit run "adata_ingestion&parsing.py"` or `streamlit run app.py` directly to isolate the issue.
    """)
else:
    st.warning("Please select a milestone to proceed.")

if __name__ == "__main__":
    pass  # Streamlit handles execution