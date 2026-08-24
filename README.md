# Northstar QA Critic Agent

This project implements a QA Critic Agent that evaluates customer support triage messages for clarity, accuracy, and professionalism. I built a retry loop that automatically rewrites failed triages and re-checks them until they meet quality standards.

## What This Agent Does 
- Reviews a triage message
- Scores it PASS or FAIL using a custom rubric
- Automatically improves failed triages
- Logs final results for QA tracking

## Tech Used
- Python
- LangChain Core
- DeepSeek v4 Flash (ChatOpenAI API)
- `.env` for secure API key storage

## Project Structure
- `critic_agent.py` — main agent logic  
- `.gitignore` — hides `.env` and `venv`  
- `requirements.txt` — dependencies  

## How to Run
Install dependencies:

pip install -r requirements.txt

Run agent
python critic_agent.py

## Author:
**Alexis Avina**
