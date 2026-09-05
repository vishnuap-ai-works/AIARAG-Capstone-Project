# Pipeline Architecture

This document outlines the end-to-end architecture of the Document Ingestion and Inference Pipelines. It details the high-level user workflow and the underlying Python orchestration that processes raw files into AI-ready vectors and subsequently uses them to generate accurate answers.

## 🔄 1. The High-Level User Workflow

The system is designed to be easily configurable without altering the underlying code. The ingestion process follows this simple user flow:

1. **Files**: You collect your raw knowledge-base documents (e.g., `.md`, `.txt`).
2. **Specific Folder**: You place all these files into a specific directory on your machine (e.g., `data/test/`).
3. **Update `.env`**: You update the `DATA_DIRECTORY` variable in your `.env` file to point to that specific folder.
4. **Execution**: You run the ingestion script using either `bin/run_ingestion.sh` (Mac/Linux) or `bin/run_ingestion.ps1` (Windows).

```mermaid
flowchart LR
    A[Raw Files] --> B[Data Folder]
    B --> C[Update .env]
    C --> D[Run .sh / .ps1]
    D --> E((Ingestion Starts))
```

---

## 🧠 2. The Orchestration Layer (`store.py`)

When you execute the shell scripts, they trigger the `src/pipeline/store.py` module. 

`store.py` contains the `DocumentIngestionPipeline`. This class acts as the central **Orchestrator** (or conductor) for the entire ingestion process. Rather than doing the heavy lifting itself, it delegates tasks to specialized modules sequentially.

It coordinates the flow of data through three primary RAG modules:

1. **Document Loader** (`document_loader.py`): Extracts raw text from files.
2. **Chunking** (`chunking.py`): Splits the raw text into manageable pieces.
3. **Embedding** (`embeddings.py`): Converts those pieces into mathematical vectors.

---

## 🏗️ 3. Detailed Architecture Diagram

The flowchart below demonstrates exactly how `store.py` coordinates data between the specialized modules during the ingestion of a single file:

```mermaid
flowchart TD
    subgraph Orchestration
        Store[store.py\nDocumentIngestionPipeline]
    end

    subgraph Core RAG Modules
        Loader(document_loader.py\nExtracts Text)
        Chunker(chunking.py\nSplits Text)
        Embedder(embeddings.py\nGenerates Vectors)
    end

    %% Flow of data
    Store -->|1. Passes File Path| Loader
    Loader -->|2. Returns Raw Text| Store
    
    Store -->|3. Passes Raw Text| Chunker
    Chunker -->|4. Returns Text Chunks| Store
    
    Store -->|5. Passes Text Chunks| Embedder
    Embedder -->|6. Returns Vectors| Store
    
    Store -->|7. Stores Final Vectors & Metadata| DB[(Vector Database)]
    
    %% Styling
    classDef orchestrator fill:#f9f,stroke:#333,stroke-width:2px;
    class Store orchestrator;
```

### Deep Dive into the Modules:
* **`document_loader.py`**: Utilizes a Factory Pattern to determine if a file is a `.txt` or `.md` file, and uses asynchronous threads to read the file from the disk without blocking the system.
* **`chunking.py`**: Implements a Sliding Window technique. It takes the massive string of raw text returned by the loader and cuts it into chunks of exactly 1200 characters, leaving a 200-character overlap between chunks so that context isn't lost at the boundaries.
* **`embeddings.py`**: Takes the array of text chunks and passes them to a configured Machine Learning model (like OpenAI or Ollama). The model translates the semantic meaning of the text into dense arrays of floating-point numbers (vectors), which are ultimately what the vector database uses to perform similarity searches.

---

## 🔍 4. The Inference Pipeline (`inference.py`)

Once data has been ingested, the Inference Pipeline takes over to answer user questions using the stored knowledge. Similar to ingestion, it can be triggered via the CLI scripts `bin/run_inference.sh` (or `.ps1`).

```mermaid
flowchart LR
    A[User Query] --> B[Run .sh / .ps1]
    B --> C((Inference Starts))
    C --> D[Retrieve Context]
    D --> E[Generate Answer]
    E --> F[Output to User]
```

`inference.py` contains the `InferencePipeline` which coordinates the querying process by delegating to specialized modules.

```mermaid
flowchart TD
    subgraph Orchestration
        Inference[inference.py\nInferencePipeline]
    end

    subgraph Core RAG Modules
        Retriever(retriever.py\nRetrieves & Reranks)
        Generator(generator.py\nGenerates Answer)
    end

    %% Flow of data
    Inference -->|1. Passes Query| Retriever
    Retriever <-->|2. Fetch similar chunks| DB[(Vector Database)]
    Retriever -->|3. Returns Top K Context| Inference
    
    Inference -->|4. Passes Query + Context| Generator
    Generator -->|5. Returns Synthesized Answer| Inference
    
    %% Styling
    classDef orchestrator fill:#f9f,stroke:#333,stroke-width:2px;
    class Inference orchestrator;
```

### Deep Dive into the Inference Modules:
* **`retriever.py`**: Embeds the user query and performs a similarity search against the vector database to find the most relevant document chunks. It optionally utilizes `reranker.py` to refine the results.
* **`reranker.py`**: (Optional) Acts as a secondary retrieval stage. It takes the initial results from the retriever and scores them using a Cross-Encoder model to ensure only the highest-quality context is returned.
* **`generator.py`**: Takes the highly relevant context chunks and the user's original query, injects them into an LLM prompt template, and calls the LLM to synthesize a conversational and accurate answer.
