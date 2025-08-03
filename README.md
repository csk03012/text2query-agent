# Text2Query Agent

This project is a text-to-query agent that uses a large language model (LLM) to answer natural language questions about data in an Excel sheet. The agent is built using Langchain/Langgraph and uses an open-source LLM served via Ollama.

## Features

- **Natural Language Queries:** Ask questions about your data in plain English.
- **Pandas Integration:** The agent generates and executes pandas code to answer your questions.
- **Iterative Error Correction:** The agent can correct its own code if it encounters errors.
- **Streamlit UI:** A simple web interface for uploading your Excel sheet and asking questions.

## Getting Started

### Prerequisites

- Python 3.7+
- Pip
- Ollama

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/text2query-agent.git
   cd text2query-agent
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Pull the LLM from Ollama:**

   ```bash
   ollama pull <your-chosen-llm>
   ```

### Usage

1. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

2. **Open your web browser to `http://localhost:8501`**

3. **Upload your Excel sheet.**

4. **Enter your natural language query.**

5. **Click "Run Query".**
