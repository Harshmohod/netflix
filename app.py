import pandas as pd
import gradio as gr
import os
import re
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util

# ✅ Auto-install en_core_web_sm if missing
import spacy
nlp = spacy.load("en_core_web_sm")

# ✅ Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load Netflix dataset
df = pd.read_csv("netflix_titles.csv")
df.fillna("", inplace=True)
df.columns = [col.lower().strip() for col in df.columns]

# ✅ Convert row to single text string
def row_to_text(row):
    return (
        f"The movie '{row['title']}' is a {row['listed_in']} title from {row['country']} "
        f"released in {row['release_year']}, directed by {row['director']}, starring {row['cast']}. "
        f"Description: {row['description']}"
    )

# ✅ Use cached embeddings if available
if os.path.exists("embeddings.npy") and os.path.exists("embedding_texts.txt"):
    print("🔄 Loading cached embeddings...")
    corpus_embeddings = np.load("embeddings.npy")
    with open("embedding_texts.txt", "r", encoding="utf-8") as f:
        df["embedding_text"] = f.read().splitlines()
else:
    print("⚙️ Generating embeddings for the first time...")
    df["embedding_text"] = df.apply(row_to_text, axis=1)
    corpus_embeddings = model.encode(df["embedding_text"].tolist(), convert_to_tensor=True)
    np.save("embeddings.npy", corpus_embeddings.cpu().numpy())
    with open("embedding_texts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(df["embedding_text"]))

# ✅ Extract filters from user query
def extract_filters(query):
    doc = nlp(query.lower())
    filters = {}

    # Genre detection
    GENRES = ['romantic', 'romance', 'comedy', 'action', 'crime', 'horror',
              'cartoon', 'anime', 'thriller', 'documentary', 'drama']
    filters["genre"] = [g for g in GENRES if g in query]

    # Year filters
    after = re.search(r"after (\d{4})", query)
    if after:
        filters["after_year"] = int(after.group(1))

    before = re.search(r"before (\d{4})", query)
    if before:
        filters["before_year"] = int(before.group(1))

    between = re.search(r"between (\d{4}) and (\d{4})", query)
    if between:
        filters["between_years"] = (int(between.group(1)), int(between.group(2)))

    # Country (GPE)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            filters["country"] = ent.text

    # Director / Actor name via regex
    person_match = re.search(r"(?:directed by|by director|by)\s+([a-zA-Z .]+)", query)
    if person_match:
        filters["person"] = person_match.group(1).strip().title()

    return filters

# ✅ Main chatbot function
def chatbot(query):
    filters = extract_filters(query)
    print("Extracted filters:", filters)

    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_indices = scores.argsort(descending=True)

    filtered = []
    for idx in top_indices[:100]:
        row = df.iloc[int(idx)]

        genre_match = True
        country_match = True
        year_match = True
        person_match = True

        # Genre filter
        if filters.get("genre"):
            genre_match = any(g.lower() in row["listed_in"].lower() for g in filters["genre"])

        # Country filter
        if filters.get("country"):
            country_match = filters["country"].lower() in row["country"].lower()

        # Person filter (in cast or director)
        if filters.get("person"):
            person = filters["person"].lower()
            person_match = person in row["director"].lower() or person in row["cast"].lower()

        # Year filter
        year = int(row["release_year"])
        if "after_year" in filters:
            year_match = year > filters["after_year"]
        if "before_year" in filters:
            year_match = year < filters["before_year"]
        if "between_years" in filters:
            start, end = filters["between_years"]
            year_match = start <= year <= end

        if genre_match and country_match and year_match and person_match:
            filtered.append(row)

        if len(filtered) >= 10:
            break

    if not filtered:
        return "❌ No results found."

    # Format output
    response = ""
    for row in filtered:
        response += f"🎬 **{row['title']}**\n"
        response += f"📅 Year: {row['release_year']} | 🎭 Genre: {row['listed_in']}\n"
        response += f"🌍 Country: {row['country'] or 'N/A'}\n"
        response += f"🎬 Director: {row['director'] or 'N/A'}\n"
        response += f"👥 Cast: {row['cast'] or 'N/A'}\n"
        response += f"📝 {row['description'][:200]}...\n---\n"

    return response

# ✅ Gradio Interface
gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="Ask anything about Netflix titles"),
    outputs="markdown",
    title="🎬 Netflix Smart Chatbot",
    description="Supports genre, year range (after/before/between), country, director, actor — using spaCy + embeddings for smarter filtering."
).launch(share=True, debug=True)



