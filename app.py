from flask import Flask, render_template, request, redirect, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether
)
from reportlab.lib.units import mm

import io
import os
import sqlite3
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq
from dotenv import load_dotenv


# =========================================================
# APP CONFIGURATION
# =========================================================

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "resume_secret_key"
)

groq_api_key = os.getenv("GROQ_API_KEY")

client = None

if groq_api_key:
    client = Groq(api_key=groq_api_key)


UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# DATABASE
# =========================================================

def create_database():

    conn = sqlite3.connect("database.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            resume_name TEXT,
            score INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recruiter_decisions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            resume_name TEXT,
            score INTEGER,
            decision TEXT,
            UNIQUE(user_email, resume_name)
        )
    """)

    conn.commit()
    conn.close()


create_database()


# =========================================================
# SKILLS DATABASE
# =========================================================

skills = [
    "Python",
    "Flask",
    "SQL",
    "Git",
    "HTML",
    "CSS",
    "JavaScript",
    "Java",
    "Machine Learning",
    "Data Science",
    "Docker",
    "AWS",
    "REST API",
    "OOP",
    "Data Structures"
]


# =========================================================
# SCORE WEIGHTS
# =========================================================

SCORE_WEIGHTS = {
    "skills": 0.40,
    "projects": 0.20,
    "experience": 0.15,
    "education": 0.10,
    "nlp": 0.15
}


# =========================================================
# SKILL RESOURCE INFORMATION
# =========================================================

skill_resources = {

    "Docker": {
        "reason": "Used for containerizing applications."
    },

    "AWS": {
        "reason": "Important for cloud deployment."
    },

    "REST API": {
        "reason": "Important for backend services."
    },

    "Git": {
        "reason": "Important for version control."
    },

    "SQL": {
        "reason": "Important for database development."
    },

    "Flask": {
        "reason": "Important for Python web development."
    },

    "Data Structures": {
        "reason": "Important for software engineering interviews."
    },

    "OOP": {
        "reason": "Important for maintainable software design."
    },

    "Python": {
        "reason": "Core programming language for this role."
    },

    "Java": {
        "reason": "Common programming language for software development."
    },

    "JavaScript": {
        "reason": "Important for frontend and web development."
    },

    "Machine Learning": {
        "reason": "Useful for AI and intelligent application development."
    },

    "Data Science": {
        "reason": "Useful for data-driven applications and analytics."
    },

    "HTML": {
        "reason": "Important for building web page structure."
    },

    "CSS": {
        "reason": "Important for professional web interfaces."
    }
}


# =========================================================
# HELPER: EXTRACT PDF TEXT
# =========================================================

def extract_resume_text(filepath):

    reader = PdfReader(filepath)

    resume_text = ""

    for page in reader.pages:

        try:
            text = page.extract_text()

            if text:
                resume_text += text + "\n"

        except Exception:
            continue

    return resume_text.strip()


# =========================================================
# HELPER: NORMALIZE TEXT
# =========================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# HELPER: CHECK SKILL
# =========================================================

def skill_found(skill, text):

    normalized_text = normalize_text(text)

    normalized_skill = normalize_text(skill)

    if normalized_skill in normalized_text:
        return True

    return False


# =========================================================
# HELPER: FIND REQUIRED SKILLS
# =========================================================

def get_required_skills(job_description):

    required_skills = []

    for skill in skills:

        if skill_found(
            skill,
            job_description
        ):
            required_skills.append(skill)

    return required_skills


# =========================================================
# HELPER: MATCH SKILLS
# =========================================================

def get_matched_skills(
    resume_text,
    required_skills
):

    matched_skills = []

    for skill in required_skills:

        if skill_found(
            skill,
            resume_text
        ):
            matched_skills.append(skill)

    return matched_skills


# =========================================================
# HELPER: NLP SCORE
# =========================================================

def calculate_nlp_score(
    resume_text,
    job_description
):

    if not resume_text.strip():
        return 0

    if not job_description.strip():
        return 0

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        documents = [
            resume_text,
            job_description
        ]

        tfidf_matrix = vectorizer.fit_transform(
            documents
        )

        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )

        score = round(
            similarity[0][0] * 100
        )

        return max(
            0,
            min(score, 100)
        )

    except Exception:

        return 0


# =========================================================
# HELPER: KEYWORD SCORE
# =========================================================

def calculate_keyword_overlap(
    resume_text,
    job_description
):

    resume_words = set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9+#.-]{2,}\b",
            normalize_text(resume_text)
        )
    )

    jd_words = set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9+#.-]{2,}\b",
            normalize_text(job_description)
        )
    )

    if not jd_words:
        return 0

    common_words = resume_words.intersection(
        jd_words
    )

    score = round(
        (
            len(common_words)
            /
            len(jd_words)
        ) * 100
    )

    return max(
        0,
        min(score, 100)
    )


# =========================================================
# HELPER: PROJECT SCORE
# =========================================================

def calculate_projects_score(
    resume_text,
    job_description
):

    resume_lower = normalize_text(
        resume_text
    )

    jd_lower = normalize_text(
        job_description
    )

    project_section_keywords = [
        "project",
        "projects",
        "developed",
        "built",
        "created",
        "implemented",
        "application",
        "system"
    ]

    technical_project_keywords = [
        "python",
        "flask",
        "sql",
        "javascript",
        "java",
        "machine learning",
        "data science",
        "docker",
        "aws",
        "rest api",
        "api",
        "git",
        "html",
        "css"
    ]

    project_presence = sum(
        1
        for keyword in project_section_keywords
        if keyword in resume_lower
    )

    technical_matches = sum(
        1
        for keyword in technical_project_keywords
        if keyword in resume_lower
        and keyword in jd_lower
    )

    general_technical_presence = sum(
        1
        for keyword in technical_project_keywords
        if keyword in resume_lower
    )

    presence_score = min(
        50,
        project_presence * 7
    )

    jd_project_score = min(
        35,
        technical_matches * 7
    )

    technical_score = min(
        15,
        general_technical_presence * 2
    )

    final_score = round(
        presence_score
        + jd_project_score
        + technical_score
    )

    return min(
        final_score,
        100
    )


# =========================================================
# HELPER: EXPERIENCE SCORE
# =========================================================

def calculate_experience_score(
    resume_text,
    job_description
):

    resume_lower = normalize_text(
        resume_text
    )

    jd_lower = normalize_text(
        job_description
    )

    experience_keywords = [
        "experience",
        "internship",
        "intern",
        "work experience",
        "employment",
        "developer",
        "software engineer",
        "software developer",
        "trainee",
        "worked"
    ]

    jd_role_keywords = [
        "developer",
        "software engineer",
        "software developer",
        "backend",
        "frontend",
        "full stack",
        "python",
        "java",
        "flask",
        "sql",
        "api",
        "machine learning"
    ]

    experience_presence = sum(
        1
        for keyword in experience_keywords
        if keyword in resume_lower
    )

    role_matches = sum(
        1
        for keyword in jd_role_keywords
        if keyword in resume_lower
        and keyword in jd_lower
    )

    presence_score = min(
        60,
        experience_presence * 8
    )

    role_score = min(
        40,
        role_matches * 8
    )

    return min(
        round(
            presence_score + role_score
        ),
        100
    )


# =========================================================
# HELPER: EDUCATION SCORE
# =========================================================

def calculate_education_score(
    resume_text,
    job_description
):

    resume_lower = normalize_text(
        resume_text
    )

    jd_lower = normalize_text(
        job_description
    )

    education_keywords = [
        "education",
        "bachelor",
        "b.tech",
        "btech",
        "b.e",
        "be ",
        "degree",
        "university",
        "college",
        "engineering",
        "computer science",
        "information technology",
        "electronics"
    ]

    education_presence = sum(
        1
        for keyword in education_keywords
        if keyword in resume_lower
    )

    education_jd_matches = sum(
        1
        for keyword in education_keywords
        if keyword in resume_lower
        and keyword in jd_lower
    )

    presence_score = min(
        75,
        education_presence * 9
    )

    jd_score = min(
        25,
        education_jd_matches * 8
    )

    return min(
        round(
            presence_score + jd_score
        ),
        100
    )


# =========================================================
# HELPER: WEIGHTED ATS SCORE
# =========================================================

def calculate_weighted_score(
    skills_score,
    projects_score,
    experience_score,
    education_score,
    nlp_score
):

    weighted_score = (

        skills_score
        * SCORE_WEIGHTS["skills"]

        +

        projects_score
        * SCORE_WEIGHTS["projects"]

        +

        experience_score
        * SCORE_WEIGHTS["experience"]

        +

        education_score
        * SCORE_WEIGHTS["education"]

        +

        nlp_score
        * SCORE_WEIGHTS["nlp"]

    )

    return max(
        0,
        min(
            round(weighted_score),
            100
        )
    )


# =========================================================
# HELPER: RECOMMENDATION
# =========================================================

def get_basic_recommendation(score):

    if score >= 80:

        return (
            "Excellent match. "
            "The resume strongly aligns with "
            "the target job requirements."
        )

    elif score >= 65:

        return (
            "Strong match. "
            "Improve the remaining missing skills "
            "and strengthen measurable project impact."
        )

    elif score >= 50:

        return (
            "Good match. "
            "Improve the missing skills and "
            "strengthen project and experience evidence."
        )

    else:

        return (
            "Low match. "
            "Add relevant skills, technologies, "
            "projects and role-specific keywords."
        )


# =========================================================
# HELPER: FULL RESUME ANALYSIS
# =========================================================

def analyze_resume(
    resume_text,
    job_description
):

    required_skills = get_required_skills(
        job_description
    )

    matched_skills = get_matched_skills(
        resume_text,
        required_skills
    )

    missing_skills = []

    for skill in required_skills:

        if skill not in matched_skills:

            missing_skills.append({
                "name": skill,
                "reason": skill_resources.get(
                    skill,
                    {}
                ).get(
                    "reason",
                    "Recommended for this role."
                )
            })

    # -----------------------------------------------------
    # SKILLS SCORE
    # -----------------------------------------------------

    if required_skills:

        skills_score = round(
            (
                len(matched_skills)
                /
                len(required_skills)
            ) * 100
        )

    else:

        skills_score = 0

    # -----------------------------------------------------
    # NLP SCORE
    # -----------------------------------------------------

    nlp_score = calculate_nlp_score(
        resume_text,
        job_description
    )

    # -----------------------------------------------------
    # PROJECTS
    # -----------------------------------------------------

    projects_score = calculate_projects_score(
        resume_text,
        job_description
    )

    # -----------------------------------------------------
    # EXPERIENCE
    # -----------------------------------------------------

    experience_score = calculate_experience_score(
        resume_text,
        job_description
    )

    # -----------------------------------------------------
    # EDUCATION
    # -----------------------------------------------------

    education_score = calculate_education_score(
        resume_text,
        job_description
    )

    # -----------------------------------------------------
    # KEYWORD / NLP
    # -----------------------------------------------------

    keywords_score = nlp_score

    # -----------------------------------------------------
    # FINAL WEIGHTED SCORE
    # -----------------------------------------------------

    score = calculate_weighted_score(
        skills_score,
        projects_score,
        experience_score,
        education_score,
        keywords_score
    )

    recommendation = get_basic_recommendation(
        score
    )

    return {
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skills_score": skills_score,
        "projects_score": projects_score,
        "experience_score": experience_score,
        "education_score": education_score,
        "keywords_score": keywords_score,
        "nlp_score": nlp_score,
        "score": score,
        "recommendation": recommendation
    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    if "user_id" in session:
        return redirect("/dashboard")
    return redirect("/login")


# =========================================================
# RECRUITER DASHBOARD
# =========================================================

@app.route(
    "/recruiter",
    methods=["GET", "POST"]
)
def recruiter():

    if "user_email" not in session:

        return redirect("/login")

    if request.method == "POST":

        job_description = request.form.get(
            "job_description",
            ""
        ).strip()

        resumes = request.files.getlist(
            "resumes"
        )

        candidates = []

        required_skills = get_required_skills(
            job_description
        )

        for resume in resumes:

            if not resume.filename:
                continue

            if not resume.filename.lower().endswith(".pdf"):
                continue

            original_name = secure_filename(
                resume.filename
            )

            if not original_name:
                continue

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                original_name
            )

            resume.save(filepath)

            try:

                resume_text = extract_resume_text(
                    filepath
                )

                analysis = analyze_resume(
                    resume_text,
                    job_description
                )

                candidate = {
                    "name": original_name,
                    "score": analysis["score"],
                    "skills_score": analysis["skills_score"],
                    "projects_score": analysis["projects_score"],
                    "experience_score": analysis["experience_score"],
                    "education_score": analysis["education_score"],
                    "keywords_score": analysis["keywords_score"],
                    "nlp_score": analysis["nlp_score"],
                    "matched_skills": analysis["matched_skills"],
                    "missing_skills": analysis["missing_skills"],
                    "required_skills": analysis["required_skills"],
                    "decision": "Not Decided"
                }

                candidates.append(
                    candidate
                )

            except Exception as e:

                print(
                    "RECRUITER RESUME ERROR:",
                    repr(e)
                )

        # -------------------------------------------------
        # RANK CANDIDATES
        # -------------------------------------------------

        candidates.sort(
            key=lambda candidate: candidate["score"],
            reverse=True
        )

        # -------------------------------------------------
        # ADD RANK
        # -------------------------------------------------

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            candidate["rank"] = index

            if candidate["score"] >= 80:

                candidate["status"] = "Strong Match"

            elif candidate["score"] >= 65:

                candidate["status"] = "Good Match"

            elif candidate["score"] >= 50:

                candidate["status"] = "Moderate Match"

            else:

                candidate["status"] = "Low Match"

        # -------------------------------------------------
        # LOAD RECRUITER DECISIONS
        # -------------------------------------------------

        conn = sqlite3.connect(
            "database.db"
        )

        cursor = conn.cursor()

        for candidate in candidates:

            cursor.execute(
                """
                SELECT decision
                FROM recruiter_decisions
                WHERE user_email=?
                AND resume_name=?
                """,
                (
                    session["user_email"],
                    candidate["name"]
                )
            )

            result = cursor.fetchone()

            if result:

                candidate["decision"] = result[0]

        conn.close()

        session["recruiter_job_description"] = (
            job_description
        )

        session["recruiter_candidates"] = (
            candidates
        )

        session["recruiter_required_skills"] = (
            required_skills
        )

        return render_template(
            "recruiter.html",
            job_description=job_description,
            candidates=candidates,
            required_skills=required_skills
        )

    return render_template(
        "recruiter.html",
        job_description=session.get(
            "recruiter_job_description"
        ),
        candidates=session.get(
            "recruiter_candidates",
            []
        ),
        required_skills=session.get(
            "recruiter_required_skills",
            []
        )
    )


# =========================================================
# RECRUITER DECISION
# =========================================================

@app.route(
    "/recruiter/decision",
    methods=["POST"]
)
def recruiter_decision():

    if "user_email" not in session:

        return redirect("/login")

    resume_name = request.form.get(
        "resume_name",
        ""
    )

    try:

        score = int(
            request.form.get(
                "score",
                0
            )
        )

    except ValueError:

        score = 0

    decision = request.form.get(
        "decision",
        "Not Decided"
    )

    user_email = session[
        "user_email"
    ]

    conn = sqlite3.connect(
        "database.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO recruiter_decisions
        (
            user_email,
            resume_name,
            score,
            decision
        )
        VALUES (?, ?, ?, ?)

        ON CONFLICT(
            user_email,
            resume_name
        )

        DO UPDATE SET
            score = excluded.score,
            decision = excluded.decision
        """,
        (
            user_email,
            resume_name,
            score,
            decision
        )
    )

    conn.commit()
    conn.close()

    candidates = session.get(
        "recruiter_candidates",
        []
    )

    for candidate in candidates:

        if candidate["name"] == resume_name:

            candidate["decision"] = decision

    session["recruiter_candidates"] = (
        candidates
    )

    return redirect(
        "/recruiter"
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if "user_email" not in session:

        return redirect("/login")

    conn = sqlite3.connect(
        "database.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT name
        FROM users
        WHERE email=?
        """,
        (
            session["user_email"],
        )
    )

    user = cursor.fetchone()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM resume_history
        WHERE user_email=?
        """,
        (
            session["user_email"],
        )
    )

    total_resumes = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT MAX(score)
        FROM resume_history
        WHERE user_email=?
        """,
        (
            session["user_email"],
        )
    )

    highest_score = cursor.fetchone()[0]

    if highest_score is None:

        highest_score = 0

    cursor.execute(
        """
        SELECT resume_name, score
        FROM resume_history
        WHERE user_email=?
        ORDER BY id DESC
        LIMIT 5
        """,
        (
            session["user_email"],
        )
    )

    recent_resumes = cursor.fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        name=user[0],
        total_resumes=total_resumes,
        highest_score=highest_score,
        recent_resumes=recent_resumes
    )


# =========================================================
# REGISTER
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = generate_password_hash(
            request.form.get(
                "password",
                ""
            )
        )

        conn = sqlite3.connect(
            "database.db"
        )

        cursor = conn.cursor()

        try:

            cursor.execute(
                """
                INSERT INTO users(
                    name,
                    email,
                    password
                )
                VALUES(?,?,?)
                """,
                (
                    name,
                    email,
                    password
                )
            )

            conn.commit()
            conn.close()

            return redirect(
                "/login"
            )

        except sqlite3.IntegrityError:

            conn.close()

            return "Email already exists!"

    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        conn = sqlite3.connect(
            "database.db"
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM users
            WHERE email=?
            """,
            (
                email,
            )
        )

        user = cursor.fetchone()

        if user and check_password_hash(
            user[3],
            password
        ):

            session["user_email"] = user[2]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM resume_history
                WHERE user_email=?
                """,
                (
                    user[2],
                )
            )

            total_resumes = (
                cursor.fetchone()[0]
            )

            cursor.execute(
                """
                SELECT resume_name, score
                FROM resume_history
                WHERE user_email=?
                ORDER BY id DESC
                LIMIT 5
                """,
                (
                    user[2],
                )
            )

            recent_resumes = (
                cursor.fetchall()
            )

            cursor.execute(
                """
                SELECT MAX(score)
                FROM resume_history
                WHERE user_email=?
                """,
                (
                    user[2],
                )
            )

            highest_score = (
                cursor.fetchone()[0]
            )

            if highest_score is None:

                highest_score = 0

            conn.close()

            return render_template(
                "dashboard.html",
                name=user[1],
                total_resumes=total_resumes,
                highest_score=highest_score,
                recent_resumes=recent_resumes
            )

        else:

            conn.close()

            return "Invalid Email or Password!"

    return render_template(
        "login.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        "/login"
    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history():

    user_email = session.get(
        "user_email"
    )

    if not user_email:

        return redirect(
            "/login"
        )

    conn = sqlite3.connect(
        "database.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT resume_name, score
        FROM resume_history
        WHERE user_email=?
        ORDER BY id DESC
        """,
        (
            user_email,
        )
    )

    history_data = (
        cursor.fetchall()
    )

    conn.close()

    return render_template(
        "history.html",
        history=history_data
    )


# =========================================================
# ANALYZE RESUME
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    if "user_email" not in session:

        return redirect(
            "/login"
        )

    resume = request.files.get(
        "resume"
    )

    job_description = request.form.get(
        "job_description",
        ""
    ).strip()

    if not resume or not resume.filename:

        return "Please upload a resume."

    if not resume.filename.lower().endswith(
        ".pdf"
    ):

        return "Only PDF resumes are supported."

    resume_name = secure_filename(
        resume.filename
    )

    if not resume_name:

        return "Invalid resume filename."

    session["resume_name"] = resume_name

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        resume_name
    )

    resume.save(filepath)

    # -----------------------------------------------------
    # EXTRACT RESUME TEXT
    # -----------------------------------------------------

    try:

        resume_text = extract_resume_text(
            filepath
        )

    except Exception as e:

        print(
            "PDF EXTRACTION ERROR:",
            repr(e)
        )

        return "Could not read the uploaded PDF."

    if not resume_text.strip():

        return (
            "Could not extract text from this PDF. "
            "Please upload a text-based PDF resume."
        )

    # -----------------------------------------------------
    # COMPLETE ANALYSIS
    # -----------------------------------------------------

    analysis = analyze_resume(
        resume_text,
        job_description
    )

    required_skills = analysis[
        "required_skills"
    ]

    matched_skills = analysis[
        "matched_skills"
    ]

    missing_skills = analysis[
        "missing_skills"
    ]

    score = analysis[
        "score"
    ]

    nlp_score = analysis[
        "nlp_score"
    ]

    skills_score = analysis[
        "skills_score"
    ]

    projects_score = analysis[
        "projects_score"
    ]

    experience_score = analysis[
        "experience_score"
    ]

    education_score = analysis[
        "education_score"
    ]

    keywords_score = analysis[
        "keywords_score"
    ]

    recommendation = analysis[
        "recommendation"
    ]

    # -----------------------------------------------------
    # AI VARIABLES
    # -----------------------------------------------------

    resume_analysis = {}

    interview_questions = []

    cover_letter = ""

    recruiter_decision = {}

    career_roadmap = {}

    # -----------------------------------------------------
    # GROQ AI
    # -----------------------------------------------------

    if client:

        try:

            chat_completion = client.chat.completions.create(

                model="openai/gpt-oss-120b",

                messages=[

                    {
                        "role": "system",

                        "content": """
You are an expert technical recruiter,
resume analyst and career coach.

Analyze the candidate resume against
the job description.

Return ONLY valid JSON.

Do not use markdown.
Do not use code fences.

Use exactly this structure:

{
    "recommendations": [
        "suggestion 1",
        "suggestion 2",
        "suggestion 3",
        "suggestion 4",
        "suggestion 5"
    ],

    "resume_analysis": {
        "skills": {
            "strength": "",
            "improvement": ""
        },
        "projects": {
            "strength": "",
            "improvement": ""
        },
        "education": {
            "strength": "",
            "improvement": ""
        },
        "experience": {
            "strength": "",
            "improvement": ""
        }
    },

    "interview_questions": [
        {
            "question": "",
            "reason": "",
            "difficulty": "",
            "topic": ""
        },
        {
            "question": "",
            "reason": "",
            "difficulty": "",
            "topic": ""
        },
        {
            "question": "",
            "reason": "",
            "difficulty": "",
            "topic": ""
        },
        {
            "question": "",
            "reason": "",
            "difficulty": "",
            "topic": ""
        },
        {
            "question": "",
            "reason": "",
            "difficulty": "",
            "topic": ""
        }
    ],

    "cover_letter": "",

    "recruiter_decision": {
        "status": "",
        "confidence": 0,
        "reasons": [
            "",
            "",
            ""
        ],
        "next_round": ""
    },

    "career_roadmap": {
        "target_role": "",
        "priority_skills": [],
        "weeks": []
    }
}

Rules:

- Base everything on the actual resume
  and job description.
- Do not invent experience.
- Make interview questions specific.
- Recommendations must be actionable.
- Cover letter must be professional.
- Recruiter decision must consider the
  resume and job description.
"""
                    },

                    {
                        "role": "user",

                        "content": f"""
CANDIDATE RESUME:

{resume_text}

TARGET JOB DESCRIPTION:

{job_description}
"""
                    }
                ]
            )

            ai_text = (
                chat_completion
                .choices[0]
                .message
                .content
            )

            # Remove accidental markdown fences
            ai_text = ai_text.strip()

            if ai_text.startswith(
                "```"
            ):

                ai_text = re.sub(
                    r"^```(?:json)?",
                    "",
                    ai_text
                )

                ai_text = re.sub(
                    r"```$",
                    "",
                    ai_text
                )

                ai_text = ai_text.strip()

            ai_data = json.loads(
                ai_text
            )

            ai_recommendations = (
                ai_data.get(
                    "recommendations",
                    []
                )
            )

            if ai_recommendations:

                recommendation = "\n".join(
                    "• " + str(item)
                    for item in ai_recommendations
                )

            resume_analysis = (
                ai_data.get(
                    "resume_analysis",
                    {}
                )
            )

            interview_questions = (
                ai_data.get(
                    "interview_questions",
                    []
                )
            )

            cover_letter = (
                ai_data.get(
                    "cover_letter",
                    ""
                )
            )

            recruiter_decision = (
                ai_data.get(
                    "recruiter_decision",
                    {}
                )
            )

            career_roadmap = (
                ai_data.get(
                    "career_roadmap",
                    {}
                )
            )

        except Exception as e:

            print(
                "AI ERROR:",
                repr(e)
            )

    else:

        print(
            "GROQ_API_KEY not found. "
            "Continuing without AI analysis."
        )

    # -----------------------------------------------------
    # SAVE HISTORY
    # -----------------------------------------------------

    user_email = session.get(
        "user_email"
    )

    conn = sqlite3.connect(
        "database.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO resume_history
        (
            user_email,
            resume_name,
            score
        )
        VALUES (?, ?, ?)
        """,
        (
            user_email,
            resume_name,
            score
        )
    )

    conn.commit()
    conn.close()

    # -----------------------------------------------------
    # SAVE SESSION
    # -----------------------------------------------------

    session["resume_name"] = resume_name

    session["score"] = score

    session["nlp_score"] = nlp_score

    session["matched_skills"] = matched_skills

    session["missing_skills"] = missing_skills

    session["recommendation"] = recommendation

    session["skills_score"] = skills_score

    session["projects_score"] = projects_score

    session["experience_score"] = experience_score

    session["education_score"] = education_score

    session["keywords_score"] = keywords_score

    session["required_skills"] = required_skills

    # -----------------------------------------------------
    # RESULT PAGE
    # -----------------------------------------------------

    return render_template(

        "result.html",

        resume_name=resume_name,

        score=score,

        nlp_score=nlp_score,

        skills_score=skills_score,

        projects_score=projects_score,

        experience_score=experience_score,

        education_score=education_score,

        keywords_score=keywords_score,

        resume_analysis=resume_analysis,

        career_roadmap=career_roadmap,

        matched_skills=matched_skills,

        missing_skills=missing_skills,

        recommendation=recommendation,

        interview_questions=interview_questions,

        cover_letter=cover_letter,

        recruiter_decision=recruiter_decision,

        required_skills=required_skills
    )


# =========================================================
# PROFESSIONAL ONE-PAGE ATS PDF
# =========================================================

@app.route(
    "/download-report"
)
def download_report():

    if "user_email" not in session:

        return redirect(
            "/login"
        )

    buffer = io.BytesIO()

    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=13 * mm,

        leftMargin=13 * mm,

        topMargin=10 * mm,

        bottomMargin=10 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=21,
        alignment=TA_CENTER,
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=7
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=9.5,
        leading=11,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=4,
        spaceAfter=3
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=7,
        leading=8.5
    )

    score_style = ParagraphStyle(
        "Score",
        parent=styles["Normal"],
        fontSize=19,
        leading=21,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#16a34a")
    )

    story = []

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

    resume_name = session.get(
        "resume_name",
        "Resume"
    )

    score = int(
        session.get(
            "score",
            0
        )
    )

    nlp_score = int(
        session.get(
            "nlp_score",
            0
        )
    )

    skills_score = int(
        session.get(
            "skills_score",
            0
        )
    )

    projects_score = int(
        session.get(
            "projects_score",
            0
        )
    )

    experience_score = int(
        session.get(
            "experience_score",
            0
        )
    )

    education_score = int(
        session.get(
            "education_score",
            0
        )
    )

    keywords_score = int(
        session.get(
            "keywords_score",
            0
        )
    )

    matched_skills = session.get(
        "matched_skills",
        []
    )

    missing_skills = session.get(
        "missing_skills",
        []
    )

    recommendation = session.get(
        "recommendation",
        "No recommendation available."
    )

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "ResumeAI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "ATS Resume Screening Report",
            subtitle_style
        )
    )

    # -----------------------------------------------------
    # RESUME INFORMATION
    # -----------------------------------------------------

    file_table = Table(

        [[
            Paragraph(
                "<b>Resume</b>",
                normal_style
            ),

            Paragraph(
                resume_name,
                normal_style
            )
        ]],

        colWidths=[
            28 * mm,
            155 * mm
        ]
    )

    file_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, 0),
                colors.HexColor("#eff6ff")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#dbeafe")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4
            )
        ])
    )

    story.append(
        file_table
    )

    story.append(
        Spacer(1, 4)
    )

    # -----------------------------------------------------
    # OVERALL SCORE
    # -----------------------------------------------------

    if score >= 80:

        score_status = "Excellent Match"

    elif score >= 65:

        score_status = "Strong Match"

    elif score >= 50:

        score_status = "Good Match"

    else:

        score_status = "Needs Improvement"

    score_table = Table(

        [[
            Paragraph(
                "OVERALL ATS SCORE",
                normal_style
            ),

            Paragraph(
                f"{score}%",
                score_style
            ),

            Paragraph(
                score_status,
                normal_style
            )
        ]],

        colWidths=[
            55 * mm,
            55 * mm,
            73 * mm
        ]
    )

    score_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#f8fafc")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cbd5e1")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "ALIGN",
                (1, 0),
                (1, 0),
                "CENTER"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            )
        ])
    )

    story.append(
        score_table
    )

    # -----------------------------------------------------
    # SCORE BREAKDOWN
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Score Breakdown",
            section_style
        )
    )

    breakdown_data = [

        [
            Paragraph(
                "<b>Category</b>",
                small_style
            ),

            Paragraph(
                "<b>Score</b>",
                small_style
            ),

            Paragraph(
                "<b>Weight</b>",
                small_style
            )
        ],

        [
            Paragraph(
                "Skills Match",
                small_style
            ),

            Paragraph(
                f"{skills_score}%",
                small_style
            ),

            Paragraph(
                "40%",
                small_style
            )
        ],

        [
            Paragraph(
                "Projects Relevance",
                small_style
            ),

            Paragraph(
                f"{projects_score}%",
                small_style
            ),

            Paragraph(
                "20%",
                small_style
            )
        ],

        [
            Paragraph(
                "Experience Relevance",
                small_style
            ),

            Paragraph(
                f"{experience_score}%",
                small_style
            ),

            Paragraph(
                "15%",
                small_style
            )
        ],

        [
            Paragraph(
                "Education Relevance",
                small_style
            ),

            Paragraph(
                f"{education_score}%",
                small_style
            ),

            Paragraph(
                "10%",
                small_style
            )
        ],

        [
            Paragraph(
                "Keyword / NLP Match",
                small_style
            ),

            Paragraph(
                f"{keywords_score}%",
                small_style
            ),

            Paragraph(
                "15%",
                small_style
            )
        ]
    ]

    breakdown_table = Table(
        breakdown_data,
        colWidths=[
            115 * mm,
            35 * mm,
            33 * mm
        ]
    )

    breakdown_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#eff6ff")
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                colors.HexColor("#dbe2ea")
            ),

            (
                "ALIGN",
                (1, 0),
                (-1, -1),
                "CENTER"
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                3
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                3
            )
        ])
    )

    story.append(
        breakdown_table
    )

    # -----------------------------------------------------
    # SKILL ANALYSIS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Skill Analysis",
            section_style
        )
    )

    matched_text = ", ".join(
        matched_skills
    )

    if not matched_text:

        matched_text = (
            "No matched skills detected."
        )

    missing_names = []

    for item in missing_skills:

        if isinstance(
            item,
            dict
        ):

            missing_names.append(
                item.get(
                    "name",
                    ""
                )
            )

        else:

            missing_names.append(
                str(item)
            )

    missing_text = ", ".join(
        missing_names
    )

    if not missing_text:

        missing_text = (
            "No major missing skills detected."
        )

    skill_data = [

        [
            Paragraph(
                "<b>Matched Skills</b>",
                small_style
            ),

            Paragraph(
                matched_text,
                small_style
            )
        ],

        [
            Paragraph(
                "<b>Missing Skills</b>",
                small_style
            ),

            Paragraph(
                missing_text,
                small_style
            )
        ]
    ]

    skill_table = Table(

        skill_data,

        colWidths=[
            35 * mm,
            148 * mm
        ]
    )

    skill_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, 0),
                colors.HexColor("#f0fdf4")
            ),

            (
                "BACKGROUND",
                (0, 1),
                (0, 1),
                colors.HexColor("#fff7ed")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#dbe2ea")
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor("#e2e8f0")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4
            )
        ])
    )

    story.append(
        skill_table
    )

    # -----------------------------------------------------
    # RECOMMENDATION
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Recruiter Recommendation",
            section_style
        )
    )

    recommendation_text = (
        str(recommendation)
        .replace(
            "&",
            "&amp;"
        )
        .replace(
            "<",
            "&lt;"
        )
        .replace(
            ">",
            "&gt;"
        )
        .replace(
            "\n",
            "<br/>"
        )
    )

    recommendation_table = Table(

        [[
            Paragraph(
                recommendation_text,
                small_style
            )
        ]],

        colWidths=[
            183 * mm
        ]
    )

    recommendation_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#f8fafc")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cbd5e1")
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            )
        ])
    )

    story.append(
        KeepTogether(
            [recommendation_table]
        )
    )

    story.append(
        Spacer(1, 4)
    )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Generated by ResumeAI • ATS Resume Screening & Career Assistant",
            subtitle_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return send_file(

        buffer,

        as_attachment=True,

        download_name="ATS_Professional_Report.pdf",

        mimetype="application/pdf"
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5062
    )