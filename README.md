# RAG-ChatBot

## Project Overview

This project implements a chatbot capable of ingesting PDF documents, converting their contents into vector representations, and answering user queries based on the embedded information. The chatbot leverages vector search techniques to efficiently retrieve relevant document sections for natural language question answering.

Document content is extracted from PDFs, transformed into vectors stored in a "vector db" folder, and used in similarity searches to find the best answers to user questions. This approach combines document understanding with conversational AI, enabling interactive and contextual responses derived directly from source documents.

## Files & Folder

- `ragbot.ipynb`: A Jupyter notebook demonstrating data ingestion, vectorization, and interaction with the chatbot.
- `ragbot.py`: Python script file containing all the above jupyter notebook codes (it is created to execute the code in a single step).
- `file.env`: Environment configuration file storing API keys and project variables.
- `Short-Story.pdf`: Sample PDF document used for testing and demonstration.
- `vector db/`: Directory that contains vectorized representations of the ingested documents for efficient retrieval.
