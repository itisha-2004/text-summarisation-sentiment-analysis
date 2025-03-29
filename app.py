from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
import sqlite3
import re
import random
from youtube_transcript_api import YouTubeTranscriptApi
from fpdf import FPDF
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
import pdfplumber
load_dotenv()
gemini_api_key = os.getenv('GEMINI_KEY')

if gemini_api_key is None:
    print("Error: GEMINI_KEY not set in environment variables.")
else:
    genai.configure(api_key=gemini_api_key)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


def create_user_table():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()


create_user_table()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)

def create_contact_table():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS contact_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        subject TEXT NOT NULL,
                        message TEXT NOT NULL)''')
    conn.commit()
    conn.close()

create_contact_table()
@app.post("/signup/")
async def signup(username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

    if user:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = get_password_hash(password)
    conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
    conn.commit()
    conn.close()

    return RedirectResponse(url="/signin/", status_code=303)


@app.post("/signin/")
async def signin(username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

    if not user or not verify_password(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    conn.close()
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout/")
async def logout():
    return RedirectResponse(url="/", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
@app.get("/home", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/signin", response_class=HTMLResponse)
async def signin_page(request: Request):
    return templates.TemplateResponse("signin.html", {"request": request})


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/contactus", response_class=HTMLResponse)
async def signin_page(request: Request):
    return templates.TemplateResponse("contact us.html", {"request": request})
@app.post("/contactus", response_class=HTMLResponse)
async def submit_contact_us(request: Request, name: str = Form(...), email: str = Form(...), subject: str = Form(...), message: str = Form(...)):
    # Save the contact message to the database
    conn = get_db_connection()
    conn.execute('INSERT INTO contact_messages (name, email, subject, message) VALUES (?, ?, ?, ?)', 
        (name, email, subject, message))
    conn.commit()
    conn.close()

    # After saving, you can display a success message
    return templates.TemplateResponse("contact us.html", {
        "request": request,
        "success_message": "Your message has been sent successfully!"
    })

@app.get("/about", response_class=HTMLResponse)
async def signin_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})
@app.get("/index.js", response_class=HTMLResponse)
async def signin_page(request: Request):
    return templates.TemplateResponse("index.js", {"request": request})
@app.get("/text_analyzer", response_class=HTMLResponse)
async def text_analyzer(request: Request):
    return templates.TemplateResponse("text_analyzer.html", {"request": request})

@app.get("/upload_pdf", response_class=HTMLResponse)
async def upload_pdf_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/summarize_youtube", response_class=HTMLResponse)
async def youtube_link_page(request: Request):
    return templates.TemplateResponse("youtube.html", {"request": request})

# YouTube Transcript Retrieval
def get_youtube_transcript(video_url):
    try:
        video_id = video_url.split('v=')[1].split('&')[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL or no transcript available. Please provide a valid YouTube video URL with a transcript.")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "".join(page.extract_text() for page in pdf.pages)

def preprocess_text(text):
    return re.sub(r'[^A-Za-z0-9\s.,?!:;()\'\"\+\-\*/=<>^]', '', text)


def summarize_text(text, max_words=400):
    summarization_model = genai.GenerativeModel("gemini-1.5-flash")
    response = summarization_model.generate_content([f"Summarize the following text into approximately {max_words} words:\n{text}"])
    return response.text.strip()


def content_type_analysis(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([f"Classify the following content into categories such as news, movie, technology, etc.:\n{text}"])
    return response.text.strip()


def sentiment_analysis(text):
    sentiment_model = genai.GenerativeModel("gemini-1.5-flash")
    response = sentiment_model.generate_content([f"Analyze the sentiment of this text and provide the sentiment score between 0 and 1:\n{text}"])
    sentiment_score = float(re.search(r'(\d*\.\d+|\d+)', response.text).group(0))

    if sentiment_score >= 0.8:
        sentiment = 'Positive'
        subcategory = random.choice(['Happy', 'Excited', 'Optimistic', 'Grateful', 'Cheerful'])
    elif sentiment_score >= 0.6:
        sentiment = 'Positive'
        subcategory = random.choice(['Enjoy', 'Good', 'Nice', 'Hopeful', 'Content'])
    elif sentiment_score >= 0.4:
        sentiment = 'Neutral'
        subcategory = random.choice(['Okay', 'Fine', 'Indifferent', 'Neutral', 'Average'])
    elif sentiment_score >= 0.2:
        sentiment = 'Negative'
        subcategory = random.choice(['Sad', 'Disappointed', 'Frustrated', 'Upset', 'Angry'])
    else:
        sentiment = 'Negative'
        subcategory = random.choice(['Miserable', 'Irritated', 'Depressed', 'Annoyed', 'Hopeless'])

    return sentiment, subcategory, sentiment_score

@app.get("/download/{filename}", response_class=FileResponse)
async def download_file(filename: str):
    file_path = os.path.join("static", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type='application/pdf', filename=filename)

def save_summary_as_pdf(summary, filename="summarized_text.pdf"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, summary, align='L')
        static_path = os.path.join("static", filename)
        pdf.output(static_path)

        if not os.path.exists(static_path):
            raise HTTPException(status_code=500, detail="Failed to generate PDF. Please try again.")

        return static_path
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating PDF: " + str(e))

@app.post("/summarize_text/")
async def summarize_text_input(text: str = Form(...)):
    if not text.strip():
        return {
            "message": "Error: Empty text provided. Please enter valid text."
        }

    try:
        cleaned_text = preprocess_text(text)
        summarized_text = summarize_text(cleaned_text, max_words=400)

        content_type = content_type_analysis(summarized_text)
        sentiment, subcategory, score = sentiment_analysis(summarized_text)

        return {
            "message": "Text successfully summarized.",
            "content_type": content_type,
            "sentiment": sentiment,
            "subcategory": subcategory,
            "sentiment_score": round(score, 2),
            "summary": summarized_text
        }
    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            "message": "Error processing text. Please check the input text and try again."
        }


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {
            "message": "Error: Invalid file type. Please upload a valid PDF file."
        }

    try:
        with open(file.filename, "wb") as f:
            f.write(await file.read())

        extracted_text = extract_text_from_pdf(file.filename)
        if not extracted_text.strip():
            return {
                "message": "Error: No text found in the PDF file."
            }

        cleaned_text = preprocess_text(extracted_text)
        summarized_text = summarize_text(cleaned_text, max_words=100)

        content_type = content_type_analysis(summarized_text)
        sentiment, subcategory, score = sentiment_analysis(summarized_text)

        summarized_pdf = save_summary_as_pdf(summarized_text)

        return {
            "message": "PDF successfully processed.",
            "content_type": content_type,
            "sentiment": sentiment,
            "subcategory": subcategory,
            "sentiment_score": round(score, 2),
            "file": summarized_pdf,
            "summary": summarized_text
        }
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return {
            "message": "Error processing PDF. Please ensure the file is not corrupted and try again."
        }


@app.post("/summarize_youtube/")
async def summarize_youtube(video_url: str = Form(...)):
    if not re.match(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.*', video_url):
        return {
            "message": "Error: Invalid YouTube URL. Please provide a valid URL."
        }

    try:
        extracted_text = get_youtube_transcript(video_url)
        if not extracted_text.strip():
            return {
                "message": "Error: No transcript found for this YouTube video."
            }

        cleaned_text = preprocess_text(extracted_text)
        summarized_text = summarize_text(cleaned_text, max_words=400)

        content_type = content_type_analysis(summarized_text)
        sentiment, subcategory, score = sentiment_analysis(summarized_text)

        summarized_pdf = save_summary_as_pdf(summarized_text)

        return {
            "message": "YouTube video successfully summarized.",
            "content_type": content_type,
            "sentiment": sentiment,
            "subcategory": subcategory,
            "sentiment_score": round(score, 2),
            "file": summarized_pdf,
            "summary": summarized_text
        }
    except Exception as e:
        print(f"Error processing YouTube video: {e}")
        return {
            "message": "Error processing YouTube video. Please ensure you provide a valid YouTube URL."
        }
