DEEPFAKE DETECTION MODEL .

# 🕵️ Deepfake Detection using Machine Learning

This project uses **machine learning** to detect whether a piece of media (image, video, or audio) is **Real** or **Fake (deepfake)**, based on metadata features extracted from the file — such as lip-sync quality, visual artifacts, and lighting inconsistencies.

---

## 📖 What is a Deepfake?

A **deepfake** is a piece of media (photo, video, or audio) that has been artificially generated or manipulated using AI — often making it look like a real person said or did something they never actually did. Detecting deepfakes automatically is important for fighting misinformation, fraud, and fake news.

This project builds a simple **Logistic Regression** model that learns to tell real media apart from fake media using a set of numeric and categorical clues (metadata), rather than analyzing the raw image/video/audio itself.

---

## 📂 Dataset

**File:** `deepfake_detection_metadata_dataset.csv`
**Size:** 1000 rows (media samples), 12 columns

| Column | What it means |
|---|---|
| `media_id` | Unique ID for each media file |
| `media_type` | Type of media — Image, Video, or Audio |
| `content_category` | What the content is about (News, Interview, Political Speech, Social Media, etc.) |
| `face_count` | Number of faces detected in the media |
| `audio_present` | Whether the media has audio (Yes/No) |
| `lip_sync_score` | How well lips match the audio (low score = mismatched = likely fake) |
| `visual_artifacts_score` | Amount of visual glitches/artifacts (high score = likely fake) |
| `compression_level` | How compressed the file is |
| `lighting_inconsistency_score` | How inconsistent the lighting looks (high score = likely fake) |
| `source_platform` | Where the media was found (Facebook, YouTube, Instagram, etc.) |
| `generation_method` | How the fake was generated, if applicable (GAN, Diffusion, etc.) — blank for real media |
| `label` | Target: **Real** or **Fake** |

The dataset is almost perfectly balanced: **50.1% Real, 49.9% Fake.**

---

## ⚙️ Project Workflow

The notebook (`Deepfake_detection.ipynb`) follows these steps:

1. **Load the data** with pandas and take a first look with `df.head()`.
2. **Exploratory Data Analysis (EDA)**
   - Check for missing values and duplicate rows
   - Separate numeric vs categorical columns
   - Look at value counts for each column
   - Visualize the target label distribution
   - Check skewness and outliers (boxplots) in numeric columns
   - Plot a correlation heatmap to see which features relate most to the label
   - Plot distribution histograms for each numeric column
3. **Clean the data**
   - Convert `label` to numbers (`Real` → 1, `Fake` → 0)
   - Convert `audio_present` to numbers (`Yes` → 1, `No` → 0)
4. **Prepare features & target**
   - Drop non-useful/leaky columns and split into training (80%) and test (20%) sets
5. **Build a preprocessing + model pipeline**
   - Numeric columns: fill missing values with the mean, then scale them (`StandardScaler`)
   - Categorical columns: fill missing values with the most frequent value, then one-hot encode them
   - Combine both with `ColumnTransformer`
6. **Train the model**: a `LogisticRegression` classifier inside an sklearn `Pipeline`
7. **Evaluate the model**: predict on the test set and check accuracy

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, matplotlib, seaborn, scikit-learn
- **Model:** Logistic Regression (with a preprocessing pipeline)
- **Environment:** Jupyter / Google Colab

---

## 📊 Current Results

| Metric | Value |
|---|---|
| Model | Logistic Regression |
| Train/Test split | 80% / 20% |
| **Test Accuracy** | **43%** |

**Note:** 43% accuracy is *below* what a random guess would give on this balanced dataset (~50%). This happened because the final feature set (`media_type`, `content_category`, `face_count`, `audio_present`, `compression_level`) accidentally **excludes the most informative columns** — `lip_sync_score`, `visual_artifacts_score`, and `lighting_inconsistency_score` — which the correlation heatmap in the EDA step shows are the columns most related to the label. These were dropped in the `prepare_data()` step along with the truly non-useful ones (`media_id`, `source_platform`, `generation_method`).

### 💡 How to improve it
- Keep `lip_sync_score`, `visual_artifacts_score`, and `lighting_inconsistency_score` in the feature set — they carry the real predictive signal.
- Try other models (Random Forest, Gradient Boosting) to capture non-linear patterns.
- Add a confusion matrix, precision, recall, and F1-score for a fuller evaluation (not just accuracy).
- Handle the missing `generation_method` values (501 missing) instead of dropping the column outright, since it may still carry useful signal for fake samples.

---

## 📁 Repository Structure

```
├── Deepfake_detection.ipynb                     # Main notebook: EDA, preprocessing, model, evaluation
├── deepfake_detection_metadata_dataset.csv       # Dataset
└── README.md                                     # Project documentation
```

---

## ▶️ How to Run

1. Clone this repository
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open `Deepfake_detection.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells in order

---

## 🎯 Key Takeaway

This project demonstrates a complete, beginner-friendly machine learning workflow — from raw data to a trained model — for a deepfake detection use case. It also highlights an important real-world lesson: **feature selection matters as much as model choice**, since dropping the most predictive columns caused the model to perform worse than random guessing.
