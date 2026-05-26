import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer

# Neutral sentiment thresholds (default: 40-60%)
NEUTRAL_LOW = 0.4
NEUTRAL_HIGH = 0.6

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='IMDb Sentiment Analysis')
    parser.add_argument('-t', '--neutral-threshold', type=int, default=10,
                        help='Threshold for neutral zone in percent (default: 10, meaning 40-60%%)')
    args = parser.parse_args()

    # Calculate neutral thresholds based on argument
    neutral_margin = args.neutral_threshold / 100.0
    neutral_low = 0.5 - neutral_margin
    neutral_high = 0.5 + neutral_margin

    # Get the folder where main.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
# Строим путь к данным от этой папки
    data_path = os.path.join(base_dir, 'data', 'IMDB Dataset.csv')

    prep = DataPreprocessor(data_path)
    try:
        prep.load_data()
    except FileNotFoundError:
        print(f"Error. File not found at {data_path}")
        return

   # preprocessing and feature engineering
    prep.preprocess_data()

    # EDA
    prep.show_eda()

    # Bigram and Trigram analysis
    print("\n--- Bigram Analysis ---")
    prep.show_ngrams(n=2, top_k=15)

    print("\n--- Trigram Analysis ---")
    prep.show_ngrams(n=3, top_k=15)

    # vectorization and train-test split
    X_train, X_test, y_train, y_test = prep.split_and_vectorize()

    # Store results for model comparison
    model_results = {}

    trainer = ModelTrainer()

    # fine tuning logistic regression
    print("\n--- Logistic Regression ---")
    trainer.train_logistic_regression(X_train, y_train, use_grid_search=True)
    lr_metrics = trainer.evaluate(X_test, y_test)
    model_results['Logistic Regression'] = lr_metrics

    # Feature importance for LR
    print("\n--- Feature Importance (Logistic Regression) ---")
    trainer.plot_feature_importance(prep.vectorizer, top_n=20)

    # Confidence distribution
    print("\n--- Confidence Distribution (Logistic Regression) ---")
    trainer.plot_confidence_distribution(X_test, y_test)

    # Learning curve
    print("\n--- Learning Curve (Logistic Regression) ---")
    trainer.plot_learning_curve(X_train, y_train, cv=5)

    # random forest with hyperparameter tuning (auto-run)
    print("\n--- Random Forest ---")
    trainer.train_random_forest(X_train, y_train, use_grid_search=True)
    rf_metrics = trainer.evaluate(X_test, y_test)
    model_results['Random Forest'] = rf_metrics

    # Feature importance for RF
    print("\n--- Feature Importance (Random Forest) ---")
    trainer.plot_feature_importance(prep.vectorizer, top_n=20)

    # Model comparison
    print("\n--- Model Comparison ---")
    trainer.plot_model_comparison(model_results)

    
    print("\n" + "="*40)
    print("Write your own review to predict sentiment!")
    print("Write 'exit' to exit.")
    print("="*40)

    while True:
        user_input = input("\nEnter review: ")
        if user_input.lower() == 'exit':
            break

        # cleaning
        cleaned = prep.clean_text(user_input)
        # transform
        vectorized = prep.vectorizer.transform([cleaned])
        # predict
        prediction = trainer.model.predict(vectorized)[0]
        proba = trainer.model.predict_proba(vectorized)[0]

        # Get confidence (probability of predicted class)
        confidence = proba[prediction]

        # Check if confidence falls in neutral range
        if confidence >= neutral_low and confidence <= neutral_high:
            label = "NEUTRAL"
            confidence_pct = confidence * 100
        else:
            label = "POSITIVE" if prediction == 1 else "NEGATIVE"
            confidence_pct = confidence * 100

        print(f"Result: {label} (Confidence: {confidence_pct:.2f}%)")

if __name__ == "__main__":
    main()