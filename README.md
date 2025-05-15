# 🎬 Movie Success Predictor

A machine learning project that predicts whether a movie will be a **HIT** or **FLOP** based on features like budget, popularity, runtime, votes, and genres.

Built with:
- Python
- Pandas & Scikit-learn (for model training)
- Streamlit (for the web app UI)

---

## 💡 Features

- Uses the TMDB 5000 Movie Dataset
- Feature engineering: `budget_per_minute`, genre encoding
- Balanced dataset (equal hits and flops)
- Predicts in real-time using Streamlit
- Displays confidence (probability) of success

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/datasci-abi/movie-success-predictor.git
cd movie-success-predictor
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/Scripts/activate  # or use venv/bin/activate on Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (optional)

```bash
python main.py
```

### 5. Launch the web app

```bash
streamlit run streamlit_app.py
```

Open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
movie-success-predictor/
├── data/
│   └── tmdb_5000_movies.csv
├── main.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Example Prediction Output

- Probability of Success: **82.4%**
- Result: ✅ HIT

---

## 📝 Credits

- Dataset: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Built by datasci-abi using Python & Streamlit

---

## 🔗 License

MIT — Free to use, modify, and share.
