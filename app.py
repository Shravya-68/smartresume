import openai
import os
from fpdf import FPDF
from dotenv import load_dotenv
import spacy
from nltk.corpus import stopwords

# Load environment variables (API key, etc.)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load NLTK stopwords for filtering irrelevant words
stop_words = set(stopwords.words("english"))

# Extract keywords from job description using NLP
def extract_keywords(job_description):
    doc = nlp(job_description)
    keywords = set()
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in stop_words:
            keywords.add(token.text.lower())
    return list(keywords)

# Generate a custom resume using OpenAI GPT-3
def generate_custom_resume(user_data, job_description):
    job_keywords = extract_keywords(job_description)
    prompt = f"""
    Create a professional resume for the following individual, customized for the job description below:
    
    Job Description Keywords: {', '.join(job_keywords)}
    
    Name: {user_data['name']}
    Contact Information: {user_data['contact_info']}
    Summary: {user_data['summary']}
    Skills: {', '.join(user_data['skills'])}
    Experience:
    {user_data['experience']}
    Education:
    {user_data['education']}
    Certifications:
    {user_data['certifications']}
    
    Tailor the resume based on the job description keywords and focus on relevant skills, experience, and qualifications.
    """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.5,
    )
    
    return response.choices[0].text.strip()

# Create a PDF version of the generated resume
def create_pdf_resume(resume_content, output_filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Customized Resume", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, resume_content)
    pdf.output(output_filename)

# Main function to execute the process
def main():
    user_data = {
        'name': 'John Doe',
        'contact_info': 'john.doe@example.com | +1234567890',
        'summary': 'A highly motivated software developer with 5 years of experience in web development.',
        'skills': ['Python', 'JavaScript', 'SQL', 'React', 'Django'],
        'experience': """
        Software Engineer at XYZ Corp (2019 - Present):
        - Developed web applications using Python and Django.
        - Led a team of 5 engineers to build scalable services.
        
        Junior Developer at ABC Inc (2017 - 2019):
        - Contributed to front-end development using JavaScript and React.
        - Optimized website performance, increasing speed by 20%.
        """,
        'education': """
        Bachelor of Science in Computer Science, University of Tech (2017)
        """,
        'certifications': 'Certified Python Developer, AWS Certified Solutions Architect'
    }

    # Sample job description (could be fetched from an API or entered by the user)
    job_description = """
    We are looking for a highly skilled software engineer with experience in Python and Django. Knowledge of React is essential. 
    The candidate should have experience working in an Agile environment and be able to lead teams.
    """

    customized_resume_content = generate_custom_resume(user_data, job_description)
    
    output_filename = "customized_resume_for_job.pdf"
    
    # Create the PDF for the customized resume
    create_pdf_resume(customized_resume_content, output_filename)
    
    print(f"Customized resume generated and saved as {output_filename}.")

if __name__ == "__main__":
    main()

