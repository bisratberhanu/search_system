# Search System

This repository is a search system that can accept multiple PDFs or Word documents, then search for a specific word in them and return the sentences that are similar to the word with their similarity score.

It is built using the Python Django framework.

## How to Run the Project

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd search_system
    ```

2. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Django development server:**
    ```sh
    python manage.py runserver
    ```

## Project Structure

- **Templates:**
  - Contains a simple HTML file (`upload_and_result.html`) in the `templates/search` folder.

- **Main Code:**
  - The main code is found in the `views.py` file of the `search` app

## Features

- Upload multiple PDF or DOCX files.
- Search for a specific word or phrase in the uploaded documents.
- Return sentences that are similar to the search query along with their similarity scores.

Demo photo:
![Demo](demo.png)
