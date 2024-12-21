# Chat With Documents Using LLM

This project enables a conversational AI chatbot capable of processing and answering questions from multiple document formats, including **CSV**, **JSON**, **PDF**, and **DOCX**. It uses **LangChain** and **Hugging Face's pre-trained models** to extract information from these documents and provide relevant responses.

## Features

- **File Upload**: Upload files in CSV, JSON, PDF, or DOCX format for processing.
- **Text Extraction**: Extracts text from these document formats and processes them into chunks for efficient retrieval.
- **Conversational Retrieval**: Uses LangChain's conversational retriever to interact with the uploaded documents.
- **Chat Interface**: A simple and interactive chatbot powered by Hugging Face's language model.
- **Real-time Interaction**: Process documents once uploaded, and chat with the AI in real-time.

## Technologies Used

- **LangChain**: A framework for building language model-powered applications.
- **Streamlit**: A web app framework for machine learning and data science.
- **FAISS**: Facebook AI Similarity Search library for efficient similarity search of embeddings.
- **Hugging Face**: Provides state-of-the-art models for NLP tasks.
- **Python Libraries**:
  - `pandas`: For processing CSV files.
  - `PyPDF2`: For processing PDF files.
  - `python-docx`: For processing DOCX files.
  - `dotenv`: For handling environment variables.
  
## Setup Instructions

Follow the steps below to set up and run the project locally:

### 1. Clone the Repository

Clone the repository to your local machine using Git:

```bash
git clone https://github.com/VRAJ-07/Chat-With-Documents-Using-LLM.git
cd Chat-With-Documents-Using-LLM
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

It's a good practice to create a virtual environment to manage dependencies.

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install the Dependencies

Install the required Python dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Set Up Hugging Face API Token

This project uses Hugging Face's pre-trained models for the conversational AI. To authenticate, you'll need to set your Hugging Face API token.

1. Create an account on Hugging Face (https://huggingface.co).
2. Obtain an API token from [here](https://huggingface.co/settings/tokens).
3. Save the token in a `.env` file in your project root directory:

```plaintext
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 5. Run the Application

Once everything is set up, you can start the Streamlit app by running:

```bash
streamlit run app.py
```

This will launch a local web application, and you can interact with the chatbot by uploading documents and typing questions.

## How It Works

1. **File Upload**: You can upload documents in CSV, JSON, PDF, or DOCX format.
2. **Text Extraction**: The content of the uploaded document is extracted and processed into text chunks.
3. **Embeddings and FAISS**: The extracted text is converted into embeddings using Hugging Face's model, and stored in a FAISS vector database.
4. **Conversational Retrieval**: The chatbot allows you to ask questions about the contents of the document. It retrieves relevant information using the vector database and returns an AI-generated response.

### Example Use Case

1. Upload a **CSV** file containing customer data.
2. Ask the chatbot questions like:
   - "What are the names of all the customers?"
   - "How many customers live in New York?"

The bot will search the content of the uploaded CSV, extract relevant information, and provide an answer.

## Acknowledgments

- **LangChain**: A powerful framework that facilitates the integration of language models and tools.
- **Hugging Face**: Provides the pre-trained models used in this project.
- **Streamlit**: Makes it easy to build interactive web applications with Python.
- **FAISS**: Used for efficient vector similarity search.
- **Python Libraries**: `pandas`, `PyPDF2`, `python-docx`, and `dotenv` for handling file types and environment variables.

### Additional Notes:

- The `requirements.txt` file should contain the following dependencies:
  ```
  streamlit
  langchain
  langchain_huggingface
  pandas
  PyPDF2
  python-docx
  huggingface_hub
  faiss-cpu
  dotenv
  ```
