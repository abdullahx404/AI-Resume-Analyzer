# =============================================================================
# data/skills_db.py
# Master dictionary of recognized technical skills.
# Organised by category so it is easy to extend.
# skill_extractor.py imports SKILLS_SET (a flat set) for O(1) lookups.
# =============================================================================

# ---------------------------------------------------------------------------
# Categorised skill dictionary
# ---------------------------------------------------------------------------
SKILLS_DICT: dict[str, list[str]] = {
    "Programming Languages": [
        "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript",
        "Go", "Rust", "Swift", "Kotlin", "R", "MATLAB", "Scala", "Ruby",
        "PHP", "Bash", "Shell Scripting", "Perl",
    ],
    "Web & Frontend": [
        "React", "Angular", "Vue.js", "Next.js", "HTML", "CSS", "Sass",
        "Bootstrap", "Tailwind CSS", "Redux", "GraphQL", "REST API",
        "WebSockets", "jQuery", "Svelte",
    ],
    "Backend & Frameworks": [
        "Node.js", "Express.js", "Django", "Flask", "FastAPI", "Spring Boot",
        "Laravel", "ASP.NET", "Ruby on Rails", "Nest.js",
    ],
    "Databases": [
        "SQL", "MySQL", "PostgreSQL", "SQLite", "MongoDB", "Redis",
        "Cassandra", "DynamoDB", "Firebase", "Elasticsearch", "Oracle DB",
    ],
    "Machine Learning & AI": [
        "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
        "Computer Vision", "Reinforcement Learning", "Transfer Learning",
        "Feature Engineering", "Model Deployment", "MLOps",
        "LLMs", "Large Language Models", "Generative AI", "Prompt Engineering",
        "RAG", "Retrieval Augmented Generation",
    ],
    "ML Libraries & Frameworks": [
        "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "XGBoost",
        "LightGBM", "CatBoost", "Hugging Face", "spaCy", "NLTK",
        "OpenCV", "Pandas", "NumPy", "Matplotlib", "Seaborn",
        "Plotly", "SciPy", "Statsmodels",
    ],
    "Cloud & DevOps": [
        "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes",
        "Terraform", "Ansible", "Jenkins", "GitHub Actions", "CI/CD",
        "Linux", "Nginx", "Apache", "Serverless",
    ],
    "Data Engineering": [
        "Data Analysis", "Data Science", "Data Engineering", "ETL",
        "Apache Spark", "Hadoop", "Kafka", "Airflow", "dbt",
        "Power BI", "Tableau", "Looker", "Data Warehousing",
    ],
    "Software Engineering": [
        "Git", "GitHub", "GitLab", "Agile", "Scrum", "Jira",
        "Unit Testing", "Test Driven Development", "TDD", "OOP",
        "Design Patterns", "Microservices", "System Design",
        "API Development", "Software Architecture",
    ],
    "Security": [
        "Cybersecurity", "Penetration Testing", "Network Security",
        "Cryptography", "OWASP", "Ethical Hacking", "Firewalls",
    ],
}

# ---------------------------------------------------------------------------
# Flat set — used by skill_extractor.py for fast membership testing.
# All values are lowercased for case-insensitive matching.
# ---------------------------------------------------------------------------
SKILLS_SET: set[str] = {
    skill.lower()
    for skills in SKILLS_DICT.values()
    for skill in skills
}

# Preserve original-case mapping so we can display proper skill names.
# key: lowercase skill → value: display-case skill
SKILL_DISPLAY: dict[str, str] = {
    skill.lower(): skill
    for skills in SKILLS_DICT.values()
    for skill in skills
}
