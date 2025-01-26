import os
import re
import string
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from PyPDF2 import PdfReader
from docx import Document
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Global variables
stopwords_list = set(stopwords.words("english"))
lemmer = WordNetLemmatizer()
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
corpus = []
document_paths = []
vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)

def preprocess_text(text):
    """Preprocess the input text by removing unwanted characters and lemmatizing words."""
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"[0-9]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    words = [lemmer.lemmatize(word) for word in text.split() if word not in stopwords_list]
    return " ".join(words)

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages])
        return text
    except Exception:
        return ""

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception:
        return ""

def fit_vectorizer():
    """Fit the TF-IDF vectorizer with the current corpus."""
    if corpus:
        vectorizer.fit(corpus)
    else:
        print("Corpus is empty. TF-IDF vectorizer not fitted.")

def upload_file(request):
    """Handle file uploads and process the uploaded documents."""
    global corpus, document_paths

    if request.method == 'POST' and 'document' in request.FILES:
        files = request.FILES.getlist("document")

        for file in files:
            if not file:
                return JsonResponse({"error": "No file uploaded"}, status=400)

            filename = file.name
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif filename.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            else:
                return JsonResponse({"error": f"Unsupported file type: {filename}"}, status=400)

            if text.strip():
                preprocessed_text = preprocess_text(text)
                corpus.append(preprocessed_text)
                document_paths.append(file_path)
            else:
                print(f"Failed to extract text from file: {filename}")

        if corpus:
            fit_vectorizer()
            message = "Files uploaded and processed successfully!"
        else:
            message = "No text extracted from files."
    else:
        message = None

    uploaded_files = os.listdir(UPLOAD_FOLDER)
    return render(request, 'search/upload_and_result.html', {'message': message, 'uploaded_files': uploaded_files})

def search(request):
    """Handle search queries and return similarity results."""
    if request.method == 'POST':
        query = request.POST.get("query", "").strip()
        top_n = int(request.POST.get("top_n", 5))

        if not query:
            return render(request, 'search/upload_and_result.html', {'error': 'Query cannot be empty'})

        processed_query = preprocess_text(query)
        query_words = set(processed_query.split())

        results = []
        word_window = 5

        for doc in corpus:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', doc)
            sentences_clean = [preprocess_text(sentence) for sentence in sentences]

            for original_sentence, clean_sentence in zip(sentences, sentences_clean):
                clean_words = clean_sentence.split()
                original_words = original_sentence.split()

                for i, word in enumerate(clean_words):
                    if word in query_words:
                        start_idx = max(0, i - word_window)
                        end_idx = min(len(original_words), i + word_window + 1)

                        text_portion = " ".join(original_words[start_idx:end_idx])

                        similarity = cosine_similarity(
                            vectorizer.transform([processed_query]).toarray(),
                            vectorizer.transform([" ".join(clean_words[start_idx:end_idx])]).toarray()
                        )[0][0]

                        results.append({
                            "sentence": text_portion.strip(),
                            "similarity": similarity
                        })

        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        limited_results = results[:top_n]

        if not limited_results:
            return render(request, 'search/upload_and_result.html', {'error': 'No matching portions found.'})

        return render(request, 'search/upload_and_result.html', {'results': limited_results})

    return render(request, 'search/upload_and_result.html')