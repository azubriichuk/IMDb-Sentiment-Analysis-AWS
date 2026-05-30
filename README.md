# IMDb Sentiment Analysis — AWS Pipeline

A machine learning project that classifies IMDb movie reviews as **positive**, **negative**, or **neutral** using traditional NLP techniques, with an AWS-based deployment pipeline.

---

## Overview

The project trains two classifiers — Logistic Regression and Random Forest — on the [IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) using TF-IDF features. After training, the best model is deployed to AWS Lambda for real-time inference, with data processed via AWS Glue and the model stored in S3.

## Architecture

```
IMDb CSV ──► AWS Glue (ETL) ──► S3 (cleaned data)
                                      │
                              SageMaker (training)
                                      │
                              S3 (model artifacts)
                                      │
                           AWS Lambda (REST API)
```

For local development, `main.py` runs the full pipeline end-to-end without any AWS dependencies.

## Project Structure

```
├── main.py                   # Local end-to-end pipeline
├── src/
│   ├── preprocessor.py       # Text cleaning, EDA, TF-IDF vectorization
│   └── model_trainer.py      # Model training, evaluation, and visualization
├── lambda/
│   └── handler.py            # AWS Lambda inference handler
├── glue/
│   └── etl_job.py            # AWS Glue ETL job
├── sagemaker/
│   └── sentiment_analysis.ipynb  # SageMaker training notebook
├── scripts/
│   └── upload_to_s3.py       # Upload dataset and artifacts to S3
├── tests/
│   └── test_logic.py
└── requirements.txt
```

## Quickstart (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download `IMDB Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in a `data/` folder:

```
data/IMDB Dataset.csv
```

### 3. Run the pipeline

```bash
python main.py
```

Optional — adjust the neutral zone threshold (default ±10% around 50%):

```bash
python main.py --neutral-threshold 15
```

The script will:
- Clean and preprocess reviews (HTML removal, stopwords, lemmatization)
- Show EDA charts: sentiment distribution, review length histograms, word clouds
- Print top bigrams and trigrams per sentiment class
- Train Logistic Regression (with GridSearch) and Random Forest
- Compare model metrics
- Launch an interactive prompt to classify your own review text

## AWS Deployment

### Prerequisites

- AWS CLI configured with appropriate permissions
- S3 bucket for model artifacts
- IAM role with S3 read access for Lambda

### Steps

1. **Upload data to S3**
   ```bash
   python scripts/upload_to_s3.py
   ```

2. **Run the Glue ETL job** to preprocess the raw dataset stored in S3.

3. **Train via SageMaker** using `sagemaker/sentiment_analysis.ipynb`.

4. **Deploy Lambda** — set the following environment variables on the function:

   | Variable | Description |
   |---|---|
   | `MODEL_BUCKET` | S3 bucket name containing model artifacts |
   | `MODEL_PREFIX` | S3 key prefix (default: `models/`) |
   | `NEUTRAL_LOW` | Lower confidence bound for neutral (default: `0.4`) |
   | `NEUTRAL_HIGH` | Upper confidence bound for neutral (default: `0.6`) |

### Lambda API

**Request:**
```json
POST /
{ "review": "The movie was absolutely fantastic!" }
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 94.7,
  "probabilities": {
    "negative": 5.3,
    "positive": 94.7
  },
  "review_preview": "The movie was absolutely fantastic!"
}
```

## Text Preprocessing

Each review goes through the following steps:

1. Lowercase
2. HTML tag removal (`<br />` etc.)
3. Strip non-alphabetic characters
4. Stopword removal
5. WordNet lemmatization
6. TF-IDF vectorization (unigrams + bigrams, top 5 000 features)

## Models

| Model | Tuning |
|---|---|
| Logistic Regression | GridSearchCV over `C` and `solver` |
| Random Forest | GridSearchCV over `n_estimators` and `max_depth` |

Both models support a configurable **neutral zone**: predictions whose confidence falls between `NEUTRAL_LOW` and `NEUTRAL_HIGH` are labelled `neutral` instead of positive/negative.

## Running Tests

```bash
pytest tests/
```

## Documentation

Full project documentation: https://docs.google.com/document/d/1r9UAiR0aSxLXukS0Asr4nTbm2OxnEKcHeijQZ_cLvN0/edit?usp=sharing
