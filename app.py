import streamlit as st  # Importing necessary libraries AND MAKING THE WEB APP
import pandas as pd #loading, analyzing and preprocessing data
import pickle #joblib ka backup
import numpy as np #maths and matrix operation in IR ML
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re #(Regular Expressions) #for text processing (dlt faltu chize space wegra)
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from reportlab.lib.pagesizes import letter #generates pdf
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io #file handling
import plotly.express as px # for interactive visualizations
import plotly.graph_objects as go
import joblib  #for saving the model

def generate_tailored_resume(resume_text, job, matched_skills):
    """Generate a tailored resume draft for a selected job."""
    job_title = job.get('Job Title', 'N/A')
    company = job.get('company', 'N/A')
    job_desc = job.get('description', job.get('job_description', ''))
    keywords = ', '.join(matched_skills)
    tailored_intro = (
        f"Resume tailored for: {job_title} at {company}\n\n"
        f"Summary:\n"
        f"Experienced professional with proven skills in {keywords}.\n"
        f"Seeking to contribute to {company} as a {job_title}.\n\n"
        f"Relevant Experience & Skills:\n"
        f"{keywords}\n\n"
        f"---\n"
    )
    # Optionally, highlight keywords in the resume
    highlighted_resume = resume_text
    for skill in matched_skills:
        highlighted_resume = re.sub(f"(?i)({re.escape(skill)})", r"**\1**", highlighted_resume)
    return tailored_intro + highlighted_resume

def compute_features(resume_text, jobs_df, vectorizer):
    # Vectorize resume and jobs
    resume_vec = vectorizer.transform([resume_text])
    job_vecs = vectorizer.transform(jobs_df['combined'])

    # tfidf_sim: Cosine similarity between resume and each job
    tfidf_sim = cosine_similarity(resume_vec, job_vecs).flatten()

    # skill_overlap: Overlap between resume and job skills
    resume_tokens = set(resume_text.lower().split())
    skill_overlap = []
    title_match = []
    for _, row in jobs_df.iterrows():
        job_skills = set(row['skills'].lower().split())
        job_title = set(row['Job Title'].lower().split())
        skill_overlap.append(len(resume_tokens & job_skills))
        title_match.append(len(resume_tokens & job_title))

    features = pd.DataFrame({
        'tfidf_sim': tfidf_sim,
        'skill_overlap': skill_overlap,
        'title_match': title_match
    })
    return features
# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

@st.cache_resource
def load_models_and_data():
    """Load the trained model, vectorizer, and job dataset"""
    try:
        model = None
        vectorizer = None
        
        # Try loading with joblib first (more compatible)
        try:
            model = joblib.load('best_model.pkl')
            st.success("✅ Model loaded with joblib")
        except:
            pass
        
        try:
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            st.success("✅ Vectorizer loaded with joblib")
        except:
            pass
        
        # If joblib fails, try pickle with different protocols
        if model is None:
            for protocol in [None, 4, 3, 2]:
                try:
                    with open('best_model.pkl', 'rb') as f:
                        if protocol is None:
                            model = pickle.load(f)
                        else:
                            model = pickle.load(f, encoding='latin1')
                    st.success("✅ Model loaded with pickle")
                    break
                except Exception as e:
                    continue
        
        if vectorizer is None:
            for protocol in [None, 4, 3, 2]:
                try:
                    with open('tfidf_vectorizer.pkl', 'rb') as f:
                        if protocol is None:
                            vectorizer = pickle.load(f)
                        else:
                            vectorizer = pickle.load(f, encoding='latin1')
                    st.success("✅ Vectorizer loaded with pickle")
                    break
                except Exception as e:
                    continue
        
        # Load the job dataset
        jobs_df = pd.read_csv('cleaned_jobs_deduped.csv')
        st.success(f"✅ Loaded {len(jobs_df)} job listings")
        jobs_df['combined'] = jobs_df['combined'].fillna('')
        if model is None or vectorizer is None:
            st.error("❌ Failed to load required model files.")
            show_troubleshooting_guide()
            st.stop()
        
        return model, vectorizer, jobs_df
        
    except FileNotFoundError as e:
        st.error(f"❌ Required file not found: {e}")
        st.info("📋 Make sure the following files are in the same directory as this script:")
        st.code("""
• best_model.pkl
• tfidf_vectorizer.pkl
• cleaned_jobs_deduped.csv
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        show_troubleshooting_guide()
        st.stop()

def show_troubleshooting_guide():
    """Display troubleshooting information"""
    with st.expander("🔧 Troubleshooting Guide", expanded=True):
        st.markdown("""
        *Common Solutions for Pickle Loading Issues:*
        
        1. *Version Compatibility*: The pickle files were created with a different Python/scikit-learn version.
           python
           # Recreate your pickle files with current versions:
           import pickle
           import joblib
           from sklearn.linear_model import LogisticRegression
           from sklearn.feature_extraction.text import TfidfVectorizer
           
           # Save with joblib (recommended):
           joblib.dump(your_model, 'best_model.pkl')
           joblib.dump(your_vectorizer, 'tfidf_vectorizer.pkl')
           
        
        2. *Alternative*: Create a simple script to convert your models:
           python
           import pickle
           import joblib
           
           # Load old pickle
           with open('best_model.pkl', 'rb') as f:
               model = pickle.load(f, encoding='latin1')
           
           # Save with joblib
           joblib.dump(model, 'best_model_new.pkl')
           
        
        3. *Manual Recreation*: If you have the training code, retrain the models with current versions.
        
        4. *File Permissions*: Check if the files are accessible and not corrupted.
        """)
        
        st.markdown("*Current Environment Info:*")
        import sklearn
        st.code(f"""
Python Version: {st.runtime.legacy_caching.get_python_version()}
Scikit-learn Version: {sklearn._version_}
Pandas Version: {pd._version_}
        """)

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, lemmatize, remove stopwords"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_words = [
        lemmatizer.lemmatize(word) 
        for word in words 
        if word not in stop_words and len(word) > 2
    ]
    
    return ' '.join(processed_words)
# ...existing code...

def get_learning_resources(skill):
    """Return a list of learning resources for a given skill."""
    resources = {
        "power bi": [
            {"title": "Power BI Full Course (YouTube)", "url": "https://www.youtube.com/watch?v=AGrl-H87pRU", "type": "Free"},
            {"title": "Microsoft Power BI Learning Path", "url": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi", "type": "Free"},
            {"title": "Power BI for Beginners (Coursera)", "url": "https://www.coursera.org/learn/power-bi", "type": "Paid"}
        ],
        "sql": [
            {"title": "SQL for Beginners (YouTube)", "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY", "type": "Free"},
            {"title": "SQL Tutorial (W3Schools)", "url": "https://www.w3schools.com/sql/", "type": "Free"},
            {"title": "Databases and SQL for Data Science (Coursera)", "url": "https://www.coursera.org/learn/sql-data-science", "type": "Paid"}
        ],
        # Add more skills as needed
    }
    return resources.get(skill.lower(), [
        {"title": f"Search {skill} on YouTube", "url": f"https://www.youtube.com/results?search_query={skill}+tutorial", "type": "Free"},
        {"title": f"Search {skill} on Coursera", "url": f"https://www.coursera.org/search?query={skill}", "type": "Paid"}
    ])

def extract_skills(text):
    """Extract potential skills from text"""
    # Common technical skills pattern
    skill_patterns = [
        r'\b(?:python|java|javascript|c\+\+|sql|html|css|react|angular|vue|node\.js|django|flask|tensorflow|pytorch|pandas|numpy|scikit-learn|docker|kubernetes|aws|azure|gcp|git|github|linux|windows|mysql|postgresql|mongodb|redis|elasticsearch|spark|hadoop|tableau|powerbi|excel|r|scala|go|rust|swift|kotlin|php|ruby|perl|bash|shell|ci/cd|devops|agile|scrum|jira|confluence|slack|trello|photoshop|illustrator|figma|sketch|autocad|solidworks|matlab|sas|spss|salesforce|sap|oracle|microsoft office|word|powerpoint|outlook|teams|zoom|slack)\b',
        r'\b(?:machine learning|deep learning|artificial intelligence|data science|data analysis|web development|mobile development|frontend|backend|full stack|software engineering|quality assurance|project management|product management|business analysis|digital marketing|social media|content creation|graphic design|ui/ux|user experience|database administration|network administration|cybersecurity|information security|cloud computing|big data|data mining|statistical analysis|financial modeling|risk management|supply chain|logistics|human resources|customer service|sales|marketing|accounting|finance)\b'
    ]
    
    skills = set()
    text_lower = text.lower()
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        skills.update(matches)
    
    return list(skills)

def get_top_jobs(resume_vector, job_vectors, jobs_df, top_n=20):
    """Get top N jobs based on cosine similarity"""
    # Calculate cosine similarities
    similarities = cosine_similarity(resume_vector, job_vectors).flatten()
    
    # Get top N indices
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Return top jobs with their similarity scores
    top_jobs = jobs_df.iloc[top_indices].copy()
    top_jobs['similarity_score'] = similarities[top_indices]
    
    return top_jobs

def analyze_resume(resume_text, model, vectorizer, jobs_df, top_n=10):
    """Main analysis function"""
    # Preprocess resume
    processed_resume = preprocess_text(resume_text)
    jobs_df['combined'] = jobs_df['combined'].fillna('')
    # Vectorize resume
    resume_vector = vectorizer.transform([processed_resume])

    # Find job description column
    job_text_column = None
    for col in ['description', 'job_description', 'text', 'content', 'job_text']:
        if col in jobs_df.columns:
            job_text_column = col
            break

    if job_text_column is None:
        st.error("Could not find job description column in dataset. Expected columns: 'description', 'job_description', 'text', 'content', or 'job_text'")
        return None

    # Vectorize job descriptions
    job_vectors = vectorizer.transform(jobs_df[job_text_column].fillna(''))

    # Prepare features for model prediction
    features = compute_features(processed_resume, jobs_df, vectorizer)
    compatibility_scores = model.predict_proba(features)[:, 1]  # Use positive class probability

    # Assign compatibility scores to the full jobs_df
    jobs_df = jobs_df.copy()  # Avoid SettingWithCopyWarning
    jobs_df['compatibility_score'] = compatibility_scores

    # Calculate similarity scores for all jobs
    similarities = cosine_similarity(resume_vector, job_vectors).flatten()
    jobs_df['similarity_score'] = similarities

    # Now, sort by compatibility_score and take top_n
    top_jobs = jobs_df.sort_values('compatibility_score', ascending=False).head(top_n)

    # Extract skills
    resume_skills = extract_skills(resume_text)

    # Extract skills from top jobs
    job_skills = []
    for _, job in top_jobs.iterrows():
        skills = extract_skills(job[job_text_column])
        job_skills.append(skills)

    # Find common skills across top jobs
    all_job_skills = set()
    for skills in job_skills:
        all_job_skills.update(skills)

    # Identify matched and missing skills
    resume_skills_set = set([skill.lower() for skill in resume_skills])
    job_skills_set = set([skill.lower() for skill in all_job_skills])

    matched_skills = list(resume_skills_set.intersection(job_skills_set))
    missing_skills = list(job_skills_set - resume_skills_set)

    return {
        'top_jobs': top_jobs,
        'resume_skills': resume_skills,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'processed_resume': processed_resume
    }
def create_match_report(results, resume_text):
    """Create a downloadable match report"""
    report_content = f"""
JOB MATCHING ANALYSIS REPORT
============================

RESUME ANALYSIS
--------------
Total Resume Skills Identified: {len(results['resume_skills'])}
Resume Skills: {', '.join(results['resume_skills']) if results['resume_skills'] else 'None identified'}

SKILL ANALYSIS
--------------
Matched Skills: {len(results['matched_skills'])}
{', '.join(results['matched_skills']) if results['matched_skills'] else 'None'}

Missing Skills (Skill Gaps): {len(results['missing_skills'])}
{', '.join(results['missing_skills'][:20]) if results['missing_skills'] else 'None'}
{'...' if len(results['missing_skills']) > 20 else ''}

TOP JOB MATCHES
--------------
"""
    for idx, (_, job) in enumerate(results['top_jobs'].head(10).iterrows(), 1):
        title = job.get('Job Title', 'N/A')
        company = job.get('company', 'N/A')
        compatibility = job['compatibility_score']
        similarity = job['similarity_score']
        report_content += f"""
{idx}. {title} at {company}
   Compatibility Score: {compatibility:.3f}
   Similarity Score: {similarity:.3f}
"""

    report_content += f"""

RECOMMENDATIONS
--------------
1. Focus on developing the missing skills identified above
2. Highlight your matched skills prominently in your resume
3. Consider the top-ranked positions for your job search
4. Tailor your resume to include more keywords from your target jobs

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return report_content
def create_demo_results(resume_text):
    """Create demo results when models are not available"""
    # Extract skills from resume
    resume_skills = extract_skills(resume_text)
    
    # Create fake job data
    demo_jobs = pd.DataFrame({
        'title': [
            'Senior Data Scientist',
            'Machine Learning Engineer', 
            'Python Developer',
            'Software Engineer',
            'Data Analyst',
            'Backend Developer',
            'AI Research Scientist',
            'Full Stack Developer',
            'DevOps Engineer',
            'Product Manager'
        ],
        'company': [
            'TechCorp Inc.',
            'AI Solutions Ltd.',
            'DataFlow Systems',
            'InnovateTech',
            'Analytics Pro',
            'CodeBase Solutions',
            'Research Labs',
            'WebDev Corp',
            'CloudOps Inc.',
            'Product Innovations'
        ],
        'compatibility_score': np.random.uniform(0.4, 0.95, 10),
        'similarity_score': np.random.uniform(0.3, 0.9, 10),
        'description': [
            'Looking for a senior data scientist with Python and machine learning experience...',
            'Machine learning engineer role requiring TensorFlow and deep learning skills...',
            'Python developer position for backend development and API creation...',
            'Software engineer role with focus on scalable system design...',
            'Data analyst position requiring SQL and statistical analysis skills...',
            'Backend developer role using Python, Django, and PostgreSQL...',
            'AI research scientist position for cutting-edge machine learning research...',
            'Full stack developer role with React, Node.js, and MongoDB...',
            'DevOps engineer position requiring Docker, Kubernetes, and AWS...',
            'Product manager role for technical products and data-driven decisions...'
        ] * 1
    })
    
    # Sort by compatibility score
    demo_jobs = demo_jobs.sort_values('compatibility_score', ascending=False)
    
    # Create some common tech skills for demo
    common_skills = [
        'python', 'javascript', 'sql', 'machine learning', 'data analysis',
        'react', 'django', 'postgresql', 'aws', 'docker', 'git', 'linux',
        'tensorflow', 'pandas', 'numpy', 'scikit-learn', 'html', 'css'
    ]
    
    # Simulate matched and missing skills
    resume_skills_lower = [skill.lower() for skill in resume_skills]
    matched_skills = [skill for skill in common_skills if any(rs in skill or skill in rs for rs in resume_skills_lower)]
    missing_skills = [skill for skill in common_skills if skill not in matched_skills]
    
    return {
        'top_jobs': demo_jobs,
        'resume_skills': resume_skills,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'processed_resume': preprocess_text(resume_text)
    }
    """Create a downloadable match report"""
    report_content = f"""
JOB MATCHING ANALYSIS REPORT
============================

RESUME ANALYSIS
--------------
Total Resume Skills Identified: {len(results['resume_skills'])}
Resume Skills: {', '.join(results['resume_skills']) if results['resume_skills'] else 'None identified'}

SKILL ANALYSIS
--------------
Matched Skills: {len(results['matched_skills'])}
{', '.join(results['matched_skills']) if results['matched_skills'] else 'None'}

Missing Skills (Skill Gaps): {len(results['missing_skills'])}
{', '.join(results['missing_skills'][:20]) if results['missing_skills'] else 'None'}
{'...' if len(results['missing_skills']) > 20 else ''}

TOP JOB MATCHES
--------------
"""
    
    for idx, (_, job) in enumerate(results['top_jobs'].head(10).iterrows(), 1):
        title = job.get('title', job.get('job_title', 'N/A'))
        title = job.get('Job Title', 'N/A')
        similarity = job['similarity_score']
        
        report_content += f"""
{idx}. {title} at {company}
   Compatibility Score: {compatibility:.3f}
   Similarity Score: {similarity:.3f}
"""
    
    report_content += f"""

RECOMMENDATIONS
--------------
1. Focus on developing the missing skills identified above
2. Highlight your matched skills prominently in your resume
3. Consider the top-ranked positions for your job search
4. Tailor your resume to include more keywords from your target jobs

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report_content

def create_pdf_report(results, resume_text):
    """Create a PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Job Matching Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Resume Skills
    story.append(Paragraph("Resume Skills Identified", styles['Heading2']))
    skills_text = ', '.join(results['resume_skills']) if results['resume_skills'] else 'None identified'
    story.append(Paragraph(f"<b>Skills:</b> {skills_text}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Matched Skills
    story.append(Paragraph("Matched Skills", styles['Heading2']))
    matched_text = ', '.join(results['matched_skills']) if results['matched_skills'] else 'None'
    story.append(Paragraph(f"<b>Matched:</b> {matched_text}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Missing Skills
    story.append(Paragraph("Skill Gaps", styles['Heading2']))
    missing_text = ', '.join(results['missing_skills'][:20]) if results['missing_skills'] else 'None'
    story.append(Paragraph(f"<b>Missing Skills:</b> {missing_text}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Top Jobs
    story.append(Paragraph("Top Job Matches", styles['Heading2']))
    for idx, (_, job) in enumerate(results['top_jobs'].head(5).iterrows(), 1):
        title = job.get('Job Title', 'N/A')
        company = job.get('company', 'N/A')
        compatibility = job['compatibility_score']
        job_text = f"{idx}. {title} at {company} (Compatibility: {compatibility:.3f})"
        story.append(Paragraph(job_text, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Streamlit App
def main():
    st.set_page_config(
        page_title="Resume Skill Gap Analyzer", 
        page_icon="📊", 
        layout="wide"
    )
    
    st.title("📊 Resume Skill Gap Analyzer")
    st.markdown("Upload your resume to find the best matching jobs and identify skill gaps!")
    
    # Load models and data
    try:
        with st.spinner("Loading models and data..."):
            model, vectorizer, jobs_df = load_models_and_data()
    except Exception as e:
        st.error("Could not load required files. Using demo mode.")
        st.info("In demo mode, you can test the interface but won't get real predictions.")
        
        # Ask user if they want to continue in demo mode
        if st.button("Continue in Demo Mode"):
            st.session_state.demo_mode = True
        else:
            st.stop()
    
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    
    if not st.session_state.demo_mode:
        st.success(f"✅ Ready to analyze resumes!")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    top_n = st.sidebar.slider("Number of jobs to analyze", 5, 50, 10)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Resume Input")
        
        # Option to upload file or paste text
        input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])
        
        resume_text = ""
        
        if input_method == "Paste Text":
            resume_text = st.text_area(
                "Paste your resume text here:", 
                height=300,
                placeholder="Paste your resume content here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload your resume", 
                type=['txt', 'pdf', 'docx'],
                help="Currently supports .txt files. PDF and DOCX support coming soon!"
            )
            
            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    resume_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("PDF and DOCX support coming soon! Please use text format.")
    
    with col2:
        st.header("📈 Quick Stats")
        if resume_text:
            word_count = len(resume_text.split())
            char_count = len(resume_text)
            st.metric("Word Count", word_count)
            st.metric("Character Count", char_count)
            
            # Quick skill preview
            skills_preview = extract_skills(resume_text)
            st.metric("Skills Detected", len(skills_preview))
            
            if skills_preview:
                st.write("*Top Skills Found:*")
                for skill in skills_preview[:5]:
                    st.write(f"• {skill}")
    
    # Analysis button
    if st.button("🔍 Analyze Resume", type="primary", disabled=not resume_text.strip()):
        if resume_text.strip():
            if st.session_state.demo_mode:
                st.warning("⚠ Running in demo mode - results are simulated")
                # Create demo results
                demo_results = create_demo_results(resume_text)
                st.session_state.analysis_results = demo_results
            else:
                with st.spinner("Analyzing your resume..."):
                    results = analyze_resume(resume_text, model, vectorizer, jobs_df, top_n)
                    st.session_state.analysis_results = results
    
    # Display results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.header("📊 Analysis Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Resume Skills", len(results['resume_skills']))
        with col2:
            st.metric("Matched Skills", len(results['matched_skills']))
        with col3:
            st.metric("Skill Gaps", len(results['missing_skills']))
        with col4:
            avg_compatibility = results['top_jobs']['compatibility_score'].mean()
            st.metric("Avg. Compatibility", f"{avg_compatibility:.3f}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Top Jobs", "🔧 Skills Analysis", "📈 Visualizations", "📄 Download Report"])
        
        with tab1:
            st.subheader("Best Job Matches")
            
            for idx, (_, job) in enumerate(results['top_jobs'].iterrows(), 1):
                with st.container():
                    # Job header
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        title = job.get('Job Title', 'Job Title Not Available')
                        company = job.get('company', 'N/A')
                        st.write(f"{idx}. {title}")
                        st.write(f"Company: {company}")
                    
                    with col2:
                        compatibility = job['compatibility_score']
                        similarity = job['similarity_score']
                        st.metric("Compatibility", f"{compatibility:.3f}")
                        st.caption(f"Similarity: {similarity:.3f}")
                    
                    # Progress bars
                    st.progress(compatibility, text=f"Compatibility Score: {compatibility:.1%}")
                    
                    # Job description preview (if available)
                    job_desc_col = None
                    for col in ['description', 'job_description', 'text', 'content', 'job_text']:
                        if col in job.index and pd.notna(job[col]):
                            job_desc_col = col
                            break
                    
                    if job_desc_col:
                        with st.expander("View Job Description"):
                            st.write(job[job_desc_col][:500] + "..." if len(job[job_desc_col]) > 500 else job[job_desc_col])
                    
                    st.divider()
        
        with tab2:
            st.subheader("Skills Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("✅ Matched Skills**")
                if results['matched_skills']:
                    for skill in results['matched_skills']:
                        st.write(f"• {skill}")
                else:
                    st.write("No matched skills found")
            
            with col2:
                st.write("❌ Missing Skills (Skill Gaps)")
                if results['missing_skills']:
                    for skill in results['missing_skills'][:20]:
                        st.write(f"• {skill}")
                        with st.expander(f"Learning Path for '{skill}'"):
                            resources = get_learning_resources(skill)
                            for res in resources:
                                st.markdown(f"- [{res['title']}]({res['url']}) ({res['type']})")
                    if len(results['missing_skills']) > 20:
                        st.write(f"... and {len(results['missing_skills']) - 20} more")
                else:
                    st.write("No skill gaps identified")
        
        with tab3:
            st.subheader("Visualizations")
            
            # Compatibility scores chart
            fig_scores = px.bar(
                x=results['top_jobs']['compatibility_score'],
                y=[f"{job.get('Job Title', 'N/A')[:30]}..." for _, job in results['top_jobs'].iterrows()],
                orientation='h',
                title="Job Compatibility Scores",
                labels={'x': 'Compatibility Score', 'y': 'Job Title'}
            )
            fig_scores.update_layout(height=600)
            st.plotly_chart(fig_scores, use_container_width=True)
            
            # Skills comparison
            col1, col2 = st.columns(2)
            
            with col1:
                skills_data = {
                    'Category': ['Matched Skills', 'Missing Skills', 'Resume Skills'],
                    'Count': [len(results['matched_skills']), len(results['missing_skills']), len(results['resume_skills'])]
                }
                fig_skills = px.pie(
                    values=skills_data['Count'],
                    names=skills_data['Category'],
                    title="Skills Breakdown"
                )
                st.plotly_chart(fig_skills, use_container_width=True)
            
            with col2:
                # Top job scores
                top_5_jobs = results['top_jobs'].head(5)
                job_names = [f"{job.get('Job Title', 'N/A')[:20]}..." for _, job in top_5_jobs.iterrows()]
                fig_top = go.Figure(data=[
                    go.Bar(name='Compatibility', x=job_names, y=top_5_jobs['compatibility_score']),
                    go.Bar(name='Similarity', x=job_names, y=top_5_jobs['similarity_score'])
                ])
                fig_top.update_layout(barmode='group', title='Top 5 Jobs: Compatibility vs Similarity')
                st.plotly_chart(fig_top, use_container_width=True)
        
        with tab4:
            st.subheader("Download Analysis Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Text report
                report_text = create_match_report(results, resume_text)
                st.download_button(
                    label="📄 Download Text Report",
                    data=report_text,
                    file_name="job_match_report.txt",
                    mime="text/plain"
                )
            
            with col2:
                # PDF report
                pdf_buffer = create_pdf_report(results, resume_text)
                st.download_button(
                    label="📑 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name="job_match_report.pdf",
                    mime="application/pdf"
                )
            
            # Report preview
            with st.expander("Preview Text Report"):
                st.text(report_text)
        
        with tab1:  # Top Jobs tab
            st.subheader("🎯 One-Click Resume Tailoring")
            job_options = [f"{idx+1}. {job.get('Job Title', 'N/A')} at {job.get('company', 'N/A')}" 
               for idx, (_, job) in enumerate(results['top_jobs'].iterrows())]
            selected_job_idx = st.selectbox("Select a job to tailor your resume:", options=list(range(len(job_options))), format_func=lambda i: job_options[i])
            if st.button("Generate Tailored Resume"):
                selected_job = results['top_jobs'].iloc[selected_job_idx]
                tailored_resume = generate_tailored_resume(resume_text, selected_job, results['matched_skills'])
                st.text_area("Tailored Resume Draft", tailored_resume, height=400)
                st.download_button("Download Tailored Resume", tailored_resume, file_name="tailored_resume.txt")

if __name__ == "__main__":
    main()

