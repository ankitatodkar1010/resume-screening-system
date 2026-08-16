# AI Resume Screening & Career Assistant

An AI-powered resume screening and career assistance web application built with **Python, Flask, NLP, TF-IDF, SQLite, and Groq AI**.

The system analyzes resumes against a target job description, calculates an ATS-style match score, identifies matching and missing skills, and provides AI-powered career insights.

## 🌐 Live Demo

🔗 https://resume-screening-system-3fu2.onrender.com

## 🚀 Features

### 👤 Candidate Features

* User registration and login
* Secure password hashing
* Resume PDF upload
* Job description analysis
* ATS-style resume scoring
* Skill matching
* Missing skill detection
* Project relevance analysis
* Experience relevance analysis
* Education relevance analysis
* NLP / TF-IDF job matching
* AI resume recommendations
* AI-generated interview questions
* AI-generated cover letter
* Career roadmap
* Resume analysis history
* Professional ATS report PDF

### 👨‍💼 Recruiter Features

* Recruiter screening dashboard
* Upload multiple candidate resumes
* Add a job description
* Automatically analyze candidates
* Rank candidates by ATS score
* View matched skills
* View missing skills
* Search candidates
* Filter candidates
* Shortlist candidates
* Mark candidates for review
* Reject candidates
* Store recruiter decisions

## 🧠 ATS Scoring

The application calculates the overall score using multiple factors:

| Category             | Weight |
| -------------------- | -----: |
| Skills Match         |    40% |
| Projects Relevance   |    20% |
| Experience Relevance |    15% |
| Education Relevance  |    10% |
| NLP / Keyword Match  |    15% |

The final score is calculated from these components to provide a more meaningful assessment than simple keyword matching.

## 🤖 AI Analysis

The application uses Groq AI to generate:

* Resume improvement recommendations
* Resume strength analysis
* Technical interview questions
* Personalized cover letters
* Recruiter decision suggestions
* Career roadmaps

The AI analysis is based on the uploaded resume and target job description.

## 🛠️ Tech Stack

### Backend

* Python
* Flask
* SQLite

### AI / NLP

* Groq API
* TF-IDF
* Cosine Similarity
* Scikit-learn

### Resume Processing

* PyPDF2

### PDF Reports

* ReportLab

### Frontend

* HTML
* CSS
* Bootstrap
* JavaScript

### Security

* Werkzeug password hashing
* Flask sessions
* Environment variables

## 📁 Project Structure

```text
AI_Resume_Project/
│
├── static/
│
├── templates/
│
├── uploads/
│
├── resumes/
│
├── .env
├── .gitignore
├── app.py
├── database.db
├── job_description.txt
└── README.md
```

> Sensitive files such as `.env`, `database.db`, uploaded resumes, and Python cache files are excluded from Git using `.gitignore`.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI_Resume_Project
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the environment

macOS / Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install flask werkzeug PyPDF2 reportlab scikit-learn groq python-dotenv
```

### 5. Create `.env`

Create a `.env` file in the project root:

```text
FLASK_SECRET_KEY=your_secret_key
GROQ_API_KEY=your_groq_api_key
```

**Never upload `.env` to GitHub.**

### 6. Run the application

```bash
python3 app.py
```

Then open the local address shown in your terminal.

## 🔄 Application Flow

```text
User
  ↓
Register / Login
  ↓
Upload Resume
  ↓
Add Job Description
  ↓
PDF Text Extraction
  ↓
Skill Matching
  ↓
TF-IDF Similarity
  ↓
ATS Score Calculation
  ↓
AI Analysis
  ↓
Recommendations
  ↓
Interview Questions
  ↓
Cover Letter
  ↓
Career Roadmap
  ↓
ATS PDF Report
```

### Recruiter Flow

```text
Recruiter
   ↓
Add Job Description
   ↓
Upload Multiple Resumes
   ↓
Analyze Candidates
   ↓
Calculate ATS Scores
   ↓
Rank Candidates
   ↓
View Candidate Details
   ↓
Shortlist / Review / Reject
```

## 📊 Example

For an SDE-I job description, the system can evaluate:

* Python
* Flask
* SQL
* Git
* REST APIs
* OOP
* Data Structures
* Docker
* AWS
* HTML
* CSS
* JavaScript
* Machine Learning
* Data Science

The system then identifies which skills appear in the candidate's resume and which required skills are missing.

## 🔐 Security Notes

The project uses:

* Password hashing instead of storing plain-text passwords
* Environment variables for API credentials
* `.gitignore` to protect secrets and local data
* Secure filename handling for uploaded resumes

For production deployment, additional security measures such as CSRF protection, stronger session configuration, input validation, and a production database should be added.

## 🔮 Future Improvements

* PostgreSQL production database
* Cloud deployment
* Role-based authentication
* Advanced semantic resume matching
* Resume version comparison
* Recruiter analytics dashboard
* Candidate comparison charts 
* Job recommendation system
* Email notifications
* Better AI evaluation with structured scoring
* Automated resume improvement suggestions

## 🎯 Project Goal

The goal of this project is to demonstrate practical software engineering skills by combining:

**Web Development + Backend Engineering + Database Management + NLP + AI + Recruiter Automation**

## 👩‍💻 Author

**Ankita Todkar**

Built as a software engineering project focused on AI-powered recruitment automation.
