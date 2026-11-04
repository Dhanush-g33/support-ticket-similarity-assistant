import json
import os
import re

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


st.set_page_config(
    page_title="Solari Analytics",
    page_icon="🎫",
    layout="wide"
)

st.title("Solari Analytics")
st.subheader("Support Ticket Similarity Assistant")
st.write("Find similar historical tickets and get resolution suggestions.")


# Load the company dataset

data_path = os.path.join(
    os.path.dirname(__file__),
    "tickets.json"
)


@st.cache_data
def load_data():
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return data.get("tickets", [])

    return []


try:
    tickets = load_data()
except Exception as error:
    st.error(f"Error loading dataset: {error}")
    st.stop()

if not tickets:
    st.error("No historical tickets found.")
    st.stop()


# Clean the text

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_ticket_text(ticket):
    subject = ticket.get("subject", "")
    description = ticket.get("description", "")
    tags = ticket.get("tags", [])

    if isinstance(tags, list):
        tags = " ".join(str(tag) for tag in tags)

    text = str(subject) + " " + str(description) + " " + str(tags)
    return preprocess_text(text)


ticket_texts = [get_ticket_text(ticket) for ticket in tickets]
categories = [
    str(ticket.get("category", "Unknown")).strip().lower()
    for ticket in tickets
]


# Build TF-IDF vectors for historical tickets

@st.cache_resource
def create_tfidf_model(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    ticket_vectors = vectorizer.fit_transform(texts)
    return vectorizer, ticket_vectors


vectorizer, ticket_vectors = create_tfidf_model(ticket_texts)


# Train the category classifier

@st.cache_resource
def train_category_model(texts, labels):
    if len(set(labels)) < 2:
        return None

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(texts, labels)
    return model


category_model = train_category_model(ticket_texts, categories)


def predict_category(text, similar_tickets):
    best_similarity = (
        similar_tickets[0]["score"]
        if similar_tickets
        else 0.0
    )

    # Don't assign a category if the historical matches are too weak.
    if best_similarity < 0.25:
        return "Unknown / No strong match"

    try:
        cleaned_text = preprocess_text(text)

        if category_model is not None:
            probabilities = category_model.predict_proba([cleaned_text])[0]
            best_index = probabilities.argmax()
            confidence = probabilities[best_index]

            # Use the classifier when it is reasonably confident.
            if confidence >= 0.30:
                return category_model.classes_[best_index]

        # If the classifier is unsure, use the category of the closest
        # historical ticket as a practical fallback.
        closest_ticket = similar_tickets[0]["ticket"]
        matched_category = str(
            closest_ticket.get("category", "Unknown")
        ).strip().lower()

        if matched_category and matched_category != "unknown":
            return matched_category

        return "Uncertain"

    except Exception:
        # A fallback category can still be obtained from the closest match.
        if similar_tickets:
            return str(
                similar_tickets[0]["ticket"].get("category", "Unknown")
            ).strip().lower()

        return "Unknown"


# Simple rule-based priority prediction

def predict_priority(text):
    text = text.lower()

    high_words = [
        "critical",
        "urgent",
        "system down",
        "complete outage",
        "security breach",
        "data loss",
        "production down",
        "multiple users unable",
        "payment deducted but refund not received"
    ]

    medium_words = [
        "slow",
        "error",
        "failed",
        "issue",
        "problem",
        "delay",
        "refund",
        "payment failure",
        "not working",
        "incorrect",
        "disconnect",
        "disconnection"
    ]

    for word in high_words:
        if word in text:
            return "High"

    for word in medium_words:
        if word in text:
            return "Medium"

    return "Low"


# Sentence Transformer for semantic matching

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def create_embeddings(texts):
    model = load_embedding_model()

    return model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )


# Find historical tickets that are most similar to the new ticket

def find_similar_tickets(new_text, top_k=5):
    cleaned_text = preprocess_text(new_text)

    new_vector = vectorizer.transform([cleaned_text])

    tfidf_scores = cosine_similarity(
        new_vector,
        ticket_vectors
    )[0]

    try:
        model = load_embedding_model()

        new_embedding = model.encode(
            [cleaned_text],
            normalize_embeddings=True
        )

        old_embeddings = create_embeddings(ticket_texts)

        semantic_scores = cosine_similarity(
            new_embedding,
            old_embeddings
        )[0]

        scores = (
            0.40 * tfidf_scores +
            0.60 * semantic_scores
        )

    except Exception:
        # If the embedding model is unavailable, use TF-IDF only.
        scores = tfidf_scores

    indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in indices:
        results.append({
            "ticket": tickets[index],
            "score": float(scores[index])
        })

    return results


# Recommend a resolution only when a match is reasonably strong

def generate_recommendation(results):
    if not results:
        return "No historical tickets were found."

    best_match = results[0]

    if best_match["score"] < 0.25:
        return (
            "No strong historical match was found. "
            "Please review this ticket manually."
        )

    ticket = best_match["ticket"]

    resolution = ticket.get(
        "resolution_notes",
        "No resolution notes available."
    )

    ticket_id = ticket.get("ticket_id", "Unknown")

    return (
        f"Suggested resolution based on historical ticket "
        f"{ticket_id}:\n\n{resolution}"
)


# Ticket input interface

st.divider()
st.header("Submit a New Support Ticket")

subject = st.text_input(
    "Ticket Subject",
    placeholder="Example: Unable to login"
)

description = st.text_area(
    "Ticket Description",
    placeholder="Describe the customer's problem...",
    height=150
)

top_k = st.slider(
    "Number of similar tickets",
    min_value=1,
    max_value=10,
    value=5
)


if st.button("Analyze Ticket", type="primary"):
    if not subject.strip() or not description.strip():
        st.warning("Please enter both subject and description.")

    else:
        new_text = subject + " " + description

        with st.spinner("Analyzing the ticket..."):
            similar_tickets = find_similar_tickets(new_text, top_k)

        best_similarity = (
            similar_tickets[0]["score"]
            if similar_tickets
            else 0
        )

        predicted_category = predict_category(
            new_text,
            similar_tickets
        )

        predicted_priority = predict_priority(new_text)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Category", predicted_category)

        with col2:
            st.metric("Predicted Priority", predicted_priority)

        st.header("Similar Historical Tickets")

        if not similar_tickets or best_similarity < 0.20:
            st.warning(
                "No strong match was found in the company dataset. "
                "The ticket may need manual review."
            )
        else:
            for rank, result in enumerate(similar_tickets, start=1):
                ticket = result["ticket"]
                score = result["score"]

                with st.expander(
                    f"Rank {rank} | "
                    f"{ticket.get('ticket_id', 'Unknown')} | "
                    f"Similarity: {score:.2%}"
                ):
                    st.write("**Subject:**", ticket.get("subject", "N/A"))
                    st.write("**Description:**", ticket.get("description", "N/A"))
                    st.write("**Category:**", ticket.get("category", "N/A"))
                    st.write("**Status:**", ticket.get("status", "N/A"))
                    st.write(
                        "**Resolution Notes:**",
                        ticket.get("resolution_notes", "N/A")
                    )

        st.header("Resolution Recommendation")
        st.info(generate_recommendation(similar_tickets))


# Analytics dashboard

st.divider()
st.header("Support Ticket Analytics Dashboard")

df = pd.DataFrame(tickets)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Historical Tickets", len(df))

with col2:
    if "category" in df.columns:
        st.metric("Total Categories", df["category"].nunique())

with col3:
    if "status" in df.columns:
        resolved_count = (
            df["status"].astype(str).str.lower() == "resolved"
        ).sum()

        st.metric("Resolved Tickets", int(resolved_count))


if "category" in df.columns:
    st.subheader("Tickets by Category")

    category_counts = df["category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]

    fig = px.bar(
        category_counts,
        x="Category",
        y="Count",
        title="Ticket Category Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)


if "status" in df.columns:
    st.subheader("Tickets by Status")

    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig = px.pie(
        status_counts,
        names="Status",
        values="Count",
        title="Ticket Status Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)


st.caption("Solari Analytics - Support Ticket Assistant")
