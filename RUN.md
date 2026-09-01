# How to Run the Ingestion Pipeline

Follow these quick instructions to configure your environment variables and execute the ingestion pipeline script.

---

## 🔐 Step 1: Configure Environment Variables

The ingestion script relies on environment variables to know where your data is located and how to authenticate with your embedding models.

1. **Create your `.env` file**:
   We have provided a template file called `.env.example`. You need to create a copy of this file and name it `.env` in the root of the project.
   - **Mac/Linux/Windows (Git Bash)**:
     ```bash
     cp .env.example .env
     ```
     *(On standard Windows Command Prompt or PowerShell, simply copy and paste the file and rename it).*

2. **Fill in the required values**:
   Open the newly created `.env` file in your code editor and fill in the missing information:
   
   * **`DATA_DIRECTORY`**: Provide the absolute path to the folder where your raw `.md` or `.txt` files live. (e.g., `/Users/yourname/Documents/.../data/test/`)
   * **`EMBEDDING_MODEL_SOURCE`**: Set this to `openai`, `ollama`, or `docker` depending on what you want to use.
   * **`OPENAI_API_KEY`**: If using OpenAI, paste your secret API key here.

---

## 🚀 Step 2: Run the Ingestion Script (`store.py`)

We have provided convenient shell scripts in the `bin/` folder to automatically set up your virtual environment, install dependencies, and execute the data ingestion pipeline (`src/pipeline/store.py`).

**If you are on Mac or Linux:**
```bash
bash bin/run_ingestion.sh
```

**If you are on Windows (PowerShell):**
```powershell
.\bin\run_ingestion.ps1
```

*These scripts will automatically configure your local Python environment and begin loading, chunking, and embedding the files located in the `DATA_DIRECTORY` specified in your `.env`.*
