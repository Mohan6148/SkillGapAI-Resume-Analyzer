import spacy
from spacy.matcher import PhraseMatcher

# 1. Load a spaCy pipeline
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback for low-disk or offline environments.
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

# 2. Function to load the skill dictionary from the text file
def load_skills(file_path):
    """Reads skills from a file, one per line, and converts them to spaCy Doc objects."""
    skills = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read and clean each skill (remove extra spaces/newlines)
            raw_skills = [line.strip() for line in f if line.strip()]
        
        # Convert each skill string into a spaCy Doc object (a 'pattern')
        # This is required by PhraseMatcher
        skills = [nlp.make_doc(skill) for skill in raw_skills]
    except FileNotFoundError:
        print(f"Error: Skill list file not found at {file_path}")
    return skills

# 3. Main function to extract skills from text
def extract_skills(clean_text, skills_file_path):
    """Extracts skills from text using spaCy's PhraseMatcher."""
    
    # Load the skills and create the matcher
    skill_patterns = load_skills(skills_file_path)
    if not skill_patterns:
        return []

    # Initialize the PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab)
    
    # Add the skill patterns to the matcher. 'SKILL' is the label for the matches.
    matcher.add("SKILL", skill_patterns)
    
    # Process the clean resume text
    doc = nlp(clean_text)
    
    # Find matches (skill tokens) in the text
    matches = matcher(doc)
    
    # Extract the text of the matched skills
    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]  # The span of the match
        found_skills.add(span.text)
        
    # Return the unique list of skills found
    return sorted(list(found_skills))

# --- Example of how to use this function (will be imported by pipeline.py later) ---
if __name__ == '__main__':
    # Since you run this script from the parent folder (skillgapAI), 
    # the files are in the current working directory, not the parent (../).
    SKILLS_FILE = "skills_list.txt"
    CLEANED_FILE = "cleaned_sri_resume.txt" 
    # ...
    
    try:
        with open(CLEANED_FILE, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Cleaned resume file not found at {CLEANED_FILE}. Please run pipeline.py first.")
        exit(1)
        
    print(f"Attempting to extract skills from {CLEANED_FILE}...")
    skills = extract_skills(text, SKILLS_FILE)
    
    print("\n--- EXTRACTED SKILLS (Rule-Based) ---")
    print(skills)
    print(f"\nTotal unique skills found: {len(skills)}")