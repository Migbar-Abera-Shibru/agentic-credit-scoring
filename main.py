# main.py
import os
import pandas as pd
import numpy as np
from src.credit_data_processor import CreditDataProcessor
from src.credit_scoring_model import CreditScoringModel
from config import settings

def main():
    """Final production version for Agentic Credit Scoring"""
    print("=== AGENTIC CREDIT SCORING SYSTEM ===")
    print("Predicting Serious Delinquency in 2 Years")
    print("=" * 50)
    
    # Initialize credit data processor
    processor = CreditDataProcessor(
        target_column=settings.TARGET_COLUMN,
        id_column=settings.ID_COLUMN
    )
    
    # Load data
    train_path = os.path.join(settings.DATA_PATH, "training.csv")
    test_path = os.path.join(settings.DATA_PATH, "testing.csv")
    
    print("Loading data...")
    train_df, test_df = processor.load_data(train_path, test_path)
    train_processed, test_processed = processor.preprocess_credit_data(train_df, test_df)
    
    # Prepare features and target
    X_train = train_processed[processor.feature_columns]
    y_train = train_processed[settings.TARGET_COLUMN]
    X_test = test_processed[processor.feature_columns]
    
    print(f"\n Dataset Summary:")
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Default rate: {y_train.mean()*100:.2f}%")
    
    # Train model with sampling for stability
    print("\n=== TRAINING MODEL ===")
    model_trainer = CreditScoringModel()
    
    # Use 50,000 samples for training (balanced between speed and performance)
    results = model_trainer.train_model(X_train, y_train, sample_size=50000)
    
    # Save model
    print("\n=== SAVING MODEL ===")
    model_trainer.save_model()
    
    # Generate predictions
    print("\n=== GENERATING PREDICTIONS ===")
    probabilities = model_trainer.predict_proba(X_test)
    
    # Create submission file
    test_ids = test_processed[settings.ID_COLUMN]
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'Probability': probabilities
    })
    
    submission_path = os.path.join(settings.DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")
    
    # Detailed statistics
    print(f"\n PREDICTION ANALYSIS:")
    print(f"Total predictions: {len(probabilities):,}")
    print(f"Average probability: {probabilities.mean():.4f}")
    print(f"Risk distribution:")
    
    risk_categories = [
        (0.0, 0.1, "Low Risk"),
        (0.1, 0.3, "Medium Risk"), 
        (0.3, 0.7, "High Risk"),
        (0.7, 1.0, "Very High Risk")
    ]
    
    for low, high, category in risk_categories:
        count = ((probabilities >= low) & (probabilities < high)).sum()
        percentage = (count / len(probabilities)) * 100
        print(f"  {category}: {count:>6,} applicants ({percentage:5.1f}%)")
    
    print(f"\n MODEL TRAINING SUCCESSFUL!")
    print("Next: Build the Agentic API with explanation capabilities")

if __name__ == "__main__":
    main()