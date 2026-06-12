# AI Resume Analyzer

A tool that reads resumes and job descriptions, then scores and ranks resumes based on how well they match the job.

## What You Need

- Python 3.8 or higher
- pip (comes with Python)

## Setup Steps

1. **Download or clone this project** to your computer

2. **Open a command prompt or terminal** in the project folder

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (needed for text processing):
   ```
   python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
   ```

## How to Run

1. **Open command prompt or terminal** in the project folder

2. **Run the app:**
   ```
   streamlit run app.py
   ```

3. **A web page will open automatically** in your browser. If not, open:
   ```
   http://localhost:8501
   ```

4. **Use the web page to:**
   - Upload resumes (PDF or text files)
   - Paste a job description
   - Click "Analyze" to see matching scores
   - Download results as CSV

## How It Works

The app does this in order:
1. Reads the resume files
2. Cleans and extracts text
3. Finds skills and information
4. Compares with the job description
5. Scores each resume
6. Ranks them from best to worst match

## Troubleshooting

**Error: "Module not found"**
- Make sure you ran `pip install -r requirements.txt`

**Error: "port 8501 is in use"**
- The app will try a different port automatically

**Slow on first run**
- First time takes longer because it downloads AI models. This is normal.
