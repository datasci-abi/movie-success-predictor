import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# --- Load and prepare the model + features ---
@st.cache_data
def load_model():
    df = pd.read_csv("data/tmdb_5000_movies.csv")

    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    df['success'] = df['revenue'] > df['budget']

    # Parse genres
    df['genres'] = df['genres'].apply(lambda x: [g['name'] for g in ast.literal_eval(x)] if pd.notnull(x) else [])
    top_genres = df['genres'].explode().value_counts().head(10).index.tolist()
    df['genres_filtered'] = df['genres'].apply(lambda g: [x for x in g if x in top_genres])
    df.dropna(subset=['popularity', 'runtime', 'vote_average', 'vote_count'], inplace=True)

    # Feature engineering
    df['budget_per_minute'] = df['budget'] / df['runtime']

    # One-hot encode genres
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(mlb.fit_transform(df['genres_filtered']), columns=mlb.classes_)
    df = pd.concat([df.reset_index(drop=True), genre_dummies], axis=1)

    # Features
    features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'budget_per_minute'] + top_genres

    # Balance the dataset
    hit_df = df[df['success'] == True]
    flop_df = df[df['success'] == False]
    hit_sample = hit_df.sample(n=len(flop_df), random_state=42)
    df_balanced = pd.concat([hit_sample, flop_df]).sample(frac=1, random_state=42)

    X = df_balanced[features]
    y = df_balanced['success']

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, features, top_genres

# --- Load model and features ---
model, features, genre_list = load_model()

# --- UI ---
st.title("🎬 Movie Success Predictor")
st.write("Will your movie be a box office hit or flop?")

# --- Inputs ---
budget = st.number_input("Budget ($)", value=50000000, min_value=10000, step=100000)
popularity = st.slider("Popularity", 0.0, 100.0, 30.0)
runtime = st.slider("Runtime (min)", 60, 240, 120)
vote_avg = st.slider("Average Vote", 0.0, 10.0, 7.0)
vote_count = st.slider("Vote Count", 0, 50000, 1000)

# --- Genre selection ---
st.subheader("Genres")
selected_genres = st.multiselect("Choose genres", genre_list)

# --- Feature engineering for prediction ---
budget_per_minute = budget / runtime if runtime > 0 else 0

# Build prediction input
input_data = {f: 0 for f in features}
input_data.update({
    'budget': budget,
    'popularity': popularity,
    'runtime': runtime,
    'vote_average': vote_avg,
    'vote_count': vote_count,
    'budget_per_minute': budget_per_minute
})
for genre in selected_genres:
    input_data[genre] = 1

input_df = pd.DataFrame([input_data])

# --- Prediction ---
if st.button("Predict Movie Success"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")
    st.write("**Probability of Success:** {:.2f}%".format(probability * 100))
    if prediction:
        st.success("✅ This movie is likely to be a **HIT!**")
    else:
        st.error("❌ This movie may **FLOP**.")

    st.subheader("📊 Model Input Features")
    st.dataframe(input_df.T)
