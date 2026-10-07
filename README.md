# Student Feedback Analysis System

A Flask-based student feedback sentiment analysis system using NLTK, TF-IDF and Multinomial Naive Bayes.

## Important CSV formats

### Training dataset
The built-in training dataset must contain:

- ID
- Feedback Text
- Category
- Sentiment
- Date

The `Sentiment` column is the training label.

### New feedback upload
New feedback files should contain:

- ID
- Feedback Text
- Category
- Date

Do **not** add a Sentiment column. The application predicts the sentiment automatically using the trained model.

Approved categories:

- Teacher Feedback
- Course Content
- Examination Pattern
- Laboratory
- Library Facilities

## Train the model

From the project directory:

```powershell
python train_model.py
```

The training script uses an 80:20 stratified train/test split and saves:

- `models/sentiment_model.pkl`
- `models/metadata.json`
- `reports/test_predictions.csv`

## Run the application

```powershell
python app.py
```

Then open `http://127.0.0.1:5000`.

## Model/data-quality improvements

The training dataset contains 3,000 unique feedback records with the project's 62% positive, 23% neutral and 15% negative distribution. The generated records use varied wording, contrastive statements and harder examples rather than repeating identical feedback.

The preprocessing also preserves common negation words such as `not`, `never`, `cannot` and `wouldn't`, because removing negation can reverse sentiment meaning.

The TF-IDF configuration uses unigrams and bigrams with `min_df=3` to reduce the influence of extremely rare terms.

The training script checks for duplicate feedback text before training.

## Notes

The project is intended for local demonstration and academic evaluation. Authentication is not included in the current demo configuration.
