# RAG Document Assistant

A local document-grounded question answering system built with Python, Streamlit, Ollama, ChromaDB, and LangChain. The app lets users upload documents, index them locally, retrieve the most relevant chunks, and ask grounded questions with source-aware answers.

## Project Overview

This project is designed to work entirely on a local machine without relying on cloud APIs. It focuses on:

- uploading and processing documents in multiple formats
- splitting long documents into meaningful chunks
- embedding each chunk using a local embedding model
- storing the vectors in a persistent local database
- retrieving the best matching context for a user question
- generating answers grounded in retrieved content
- showing evidence, document names, pages, and chunk metadata
- allowing users to manage uploaded documents within the app

## Key Features

- Local AI workflow using Ollama models
- Document ingestion from PDF, TXT, Markdown, CSV, DOCX, XLSX, PPTX, HTML, and more
- Persistent vector storage with ChromaDB
- Semantic retrieval using local embeddings
- Re-ranking and cleanup of retrieved passages for higher accuracy
- Source-aware responses with evidence-based answers
- Sidebar-based document management and deletion
- Retrieval quality benchmark panel and performance tracking
- Responsive Streamlit UI with a polished dashboard layout

## Core Concepts

### 1. Retrieval-Augmented Generation (RAG)
RAG combines retrieval and generation. Instead of asking the model to answer from memory alone, the system first finds relevant chunks from the uploaded documents and then asks the model to answer using only that evidence.

This improves:

- factual accuracy
- traceability to source documents
- relevance of answers
- user trust in the output

### 2. Document Ingestion
Documents are uploaded through the Streamlit interface. The system checks file type, size, and supported formats before processing them. Once accepted, the file is parsed and converted into a clean, chunkable representation.

### 3. Chunking
Large documents are split into smaller text segments using a recursive character text splitter. This is important because:

- large files exceed context limits
- small chunks are easier to embed and search
- relevant local passages can be retrieved more precisely
- answer generation remains grounded and focused

### 4. Embeddings
Each chunk is transformed into an embedding vector using the local embedding model. These embeddings represent the semantic meaning of text in a numeric form so that similar queries can be matched to similar document segments.

### 5. Vector Storage
The generated embeddings are saved in ChromaDB, a lightweight vector database. Chroma stores metadata such as:

- source document name
- chunk content
- page number
- line or chunk location
- file path metadata if available

This allows faster retrieval and traceability during answer generation.

### 6. Retrieval and Reranking
When a user asks a question, the app searches the vector store for the closest matching chunks. The retrieval layer may then perform ranking, deduplication, and pruning to keep only the most relevant and trustworthy content before passing it to the LLM.

This process helps avoid:

- noisy retrieval results
- repeated or duplicated chunks
- context overload
- weak matches that do not contribute to the answer

### 7. Grounded Answer Generation
The language model receives the question and the retrieved chunks, and then produces an answer based on the available evidence. The system is designed to encourage answers that are:

- grounded in source content
- concise but complete
- evidence-aware
- suitable for summarization or detailed explanations

### 8. Source Transparency
The app displays source locations and document references so the user can see where the answer came from. This is a crucial part of a trustworthy local RAG system and helps users validate the output against the original files.

### 9. Document Lifecycle Management
Users can:

- upload new documents
- replace existing documents with the same name
- remove a selected document permanently
- clear all indexed documents

This ensures the document collection stays relevant and manageable.

### 10. Benchmarking and Evaluation
The app includes benchmark-style checks to evaluate retrieval quality. This gives a quick indication of whether the document index is returning useful passages for common questions.

## System Workflow

1. User uploads a document.
2. The app checks whether it is a supported and allowable file.
3. The file is processed and converted into text content.
4. The text is chunked into smaller sections.
5. Each chunk is embedded and sent to ChromaDB.
6. The user asks a question in the chat interface.
7. The app retrieves the top matching chunks.
8. Relevant chunks are ranked and filtered.
9. The model generates an answer grounded in those chunks.
10. The UI displays the answer and source evidence.

## Technology Stack

### Python
Python is the backbone of the application. It handles:

- file processing
- splitting and enrichment logic
- ChromaDB operations
- model interaction with Ollama
- Streamlit app logic

### Streamlit
Streamlit provides the user interface. It handles:

- file upload
- chat input
- sidebar management
- document registry display
- metrics and response UI

### Ollama
Ollama is used for local model hosting. The app uses:

- a local embedding model for semantic search
- a local chat model for answer generation

This allows the system to run without remote service dependencies.

### ChromaDB
ChromaDB stores vectorized document chunks and their metadata. It is ideal for a local RAG workflow because it is lightweight, persistent, and efficient for semantic search.

### LangChain
LangChain provides document loaders and text splitters used in the app. It helps standardize the pipeline from raw file content to chunked data ready for retrieval.

## Project Structure

- app.py: main application logic and UI
- README.md: project documentation and overview
- chroma_db/: persistent vector store generated by the app

## Main Application Behavior

The application is designed for practical document Q&A, including the following behaviors:

- summarize long content when the user asks for summary
- answer grounded questions using the retrieved document context
- include relevant evidence with citations or source references
- support multiple document uploads and searches
- show a clear view of what documents are attached and indexed
- delete documents effectively from the local index

## Typical Use Cases

- analyze uploaded PDFs and reports
- ask questions about policy documents
- summarize textbook chapters or notes
- query internal data saved in text or spreadsheet files
- use local AI without sending private documents to cloud services

## Advantages of This Project

- privacy-friendly because it runs locally
- suitable for sensitive or internal documents
- customizable and easy to extend
- transparent answer generation through retrieval evidence
- suitable for small to medium document collections

## Limitations

- performance depends on local hardware
- large document sets may require careful chunk tuning
- OCR and scanned image PDFs may need additional preprocessing
- answer quality depends on retrieval quality and model choice

## Suggested Improvements

- support OCR for scanned PDFs
- add more advanced reranking strategies
- support multi-document comparison
- show exact page numbers and source highlights more cleanly
- add exportable summaries and answer history
- add a stronger evaluation dashboard and accuracy metrics

## Summary

This project is a local, document-aware AI assistant for retrieving and answering questions from user-uploaded files. It combines the key elements of modern RAG systems: ingestion, chunking, embedding, vector storage, retrieval, ranking, and grounded generation. The result is a more transparent and practical document assistant that users can inspect, trust, and manage locally.
#   R A G  
 