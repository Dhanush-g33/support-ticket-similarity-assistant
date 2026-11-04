# Solari Analytics — Support Ticket Similarity Assistant

A Streamlit application that analyzes a newly submitted support ticket, predicts a category and priority, retrieves similar historical tickets, and presents a resolution suggestion based on the closest historical ticket.

## Features

- Submit a ticket using a subject and description.
- Predict a ticket category using a Logistic Regression classifier trained on the supplied historical ticket data.
- Estimate priority using keyword-based rules.
- Retrieve similar historical tickets using a hybrid similarity approach:
  - TF-IDF cosine similarity
  - Sentence Transformer semantic similarity (`all-MiniLM-L6-v2`)
- Show historical ticket details and resolution notes.
- Display a dashboard with ticket totals, category distribution, and status distribution.
- Use a fallback category from the strongest historical match when the classifier is uncertain.
- Flag weak matches for manual review.

## Technology Stack

- Python
- Streamlit
- scikit-learn
- Sentence Transformers
- pandas
- Plotly

## Project Structure

```text
support-ticket-similarity-assistant/
├── app.py
├── data/
│   └── tickets.json
├── requirements.txt
├── README.md
└── SOLUTION_DESIGN.md
```

## Run Locally

1. Install Python 3.11 or a compatible Python version.
2. Open a terminal in the `starter_kit` directory (the directory containing `app.py`).
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local Streamlit URL shown in the terminal.

The Sentence Transformer model may take longer to download/load on its first run. Later runs should benefit from Streamlit resource caching.

## How It Works

1. The app loads the historical tickets from `data/tickets.json`.
2. It combines each ticket's subject, description, and tags into text and preprocesses it.
3. TF-IDF vectors are created for historical ticket text.
4. A Logistic Regression model is trained on the historical ticket categories.
5. For a new ticket, the app calculates TF-IDF similarity and semantic similarity against the historical tickets and combines the scores.
6. The highest-ranked historical tickets are displayed.
7. The resolution notes from the closest sufficiently relevant historical ticket are presented as a suggestion.

## Limitations

- Category and priority outputs are predictions and should be reviewed by a support agent.
- Priority is estimated using keyword rules, not a separately trained priority model.
- Resolution suggestions are based on historical ticket notes; they are not independently verified instructions.
- When the historical dataset does not contain a relevant example, the app may return a weak match or request manual review.
- Model performance depends on the quality, coverage, and size of the supplied historical dataset.

## Dataset

This project uses the company-provided historical ticket dataset. Keep the dataset unchanged and share it only where permitted by the hackathon/company rules.
