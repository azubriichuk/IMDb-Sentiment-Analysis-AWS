from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, auc, precision_recall_fscore_support)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None

    def train_logistic_regression(self, X_train, y_train, use_grid_search=False):
        print("\nTraining Logistic Regression...")
        
        if use_grid_search:
            print("Starting Grid Search for Logistic Regression...")
            param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
            grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, verbose=1)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            self.best_params = grid.best_params_
            print(f"Best Params: {self.best_params}")
        else:
            self.model = LogisticRegression(C=1.0, max_iter=1000)
            self.model.fit(X_train, y_train)

    def train_random_forest(self, X_train, y_train, use_grid_search=False):
        print("\nTraining Random Forest...")

        if use_grid_search:
            print("Starting Grid Search for Random Forest (this may take time)...")
            # making the grid smaller for faster processing
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, verbose=1, n_jobs=-1)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            self.best_params = grid.best_params_
            print(f"Best Params: {self.best_params}")
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if not self.model:
            print("Model not trained yet.")
            return

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] # probabilities for ROC

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        print(f"\nAccuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        #  visualizations
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Receiver Operating Characteristic (ROC)')
        axes[1].legend(loc="lower right")

        plt.tight_layout()
        plt.show()

        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}

    def plot_feature_importance(self, vectorizer, top_n=20):
        """Plot top N most important features for sentiment classification."""
        if not self.model:
            print("Model not trained yet.")
            return

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # For Random Forest, use feature_importances_
        # For Logistic Regression, use coefficients
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            title = "Random Forest Feature Importance"
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
            title = "Logistic Regression Feature Importance (Absolute Coefficients)"
        else:
            print("Feature importance not available for this model.")
            return

        # Get top N indices
        top_indices = np.argsort(importances)[-top_n:]

        plt.figure(figsize=(10, 8))
        colors = ['green' if feature_names[i] in [feature_names[j] for j in top_indices[:top_n//2]]
                  else 'red' for i in top_indices]
        plt.barh(range(top_n), importances[top_indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Most Important Words for Sentiment')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_confidence_distribution(self, X_test, y_test):
        """Plot distribution of prediction confidence/probabilities."""
        if not self.model:
            print("Model not trained yet.")
            return

        y_proba = self.model.predict_proba(X_test)
        confidence = np.max(y_proba, axis=1)

        plt.figure(figsize=(10, 6))
        plt.hist(confidence, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Random Guess (50%)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, results_dict):
        """Plot comparison of multiple models."""
        if not results_dict:
            print("No results to compare. Train models first.")
            return

        models = list(results_dict.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        # Prepare data
        data = {metric: [] for metric in metrics}
        for model_name in models:
            for metric in metrics:
                data[metric].append(results_dict[model_name][metric])

        x = np.arange(len(models))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + i * width, data[metric], width, label=metric)
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, X_train, y_train, cv=5):
        """Plot learning curve showing performance vs training size."""
        if not self.model:
            print("Model not trained yet.")
            return

        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X_train, y_train, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.plot(train_sizes, val_mean, 'o-', color='orange', label='Cross-Validation Score')

        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')

        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()