"""
Train all models - Run this directly from project root
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import StudentDataPreprocessor
from src.model_training import StudentPerformanceModels
from sklearn.model_selection import train_test_split
import joblib

def main():
    print("="*60)
    print("🚀 TRAINING STUDENT PERFORMANCE MODELS")
    print("="*60)
    
    # Initialize
    preprocessor = StudentDataPreprocessor()
    model_trainer = StudentPerformanceModels()
    
    # Load data
    data_path = 'data/raw/dst data.xlsx'
    
    if not os.path.exists(data_path):
        print(f"\n❌ Data not found: {data_path}")
        print("Please place your dataset at: data/raw/dst data.xlsx")
        return
    
    print(f"\n📊 Loading data...")
    df = preprocessor.load_and_preprocess_data(data_path)
    
    if df is None:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {df.shape[0]} records")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X, y, features = preprocessor.prepare_modeling_data(df, target_type='classification')
    
    if X is None:
        print("❌ Failed to prepare data")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training: {X_train.shape[0]} samples")
    print(f"📊 Test: {X_test.shape[0]} samples")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Train Random Forest
    print("\n" + "="*50)
    print("🌲 Training Random Forest...")
    print("="*50)
    rf_model, rf_importance, _ = model_trainer.train_random_forest(
        X_train, X_test, y_train, y_test
    )
    
    if rf_model:
        joblib.dump(rf_model, 'models/random_forest.pkl')
        print("✅ Random Forest saved to: models/random_forest.pkl")
        
        if rf_importance is not None:
            rf_importance.to_csv('reports/feature_importance.csv', index=False)
            print("✅ Feature importance saved to: reports/feature_importance.csv")
    
    # Train Neural Network
    print("\n" + "="*50)
    print("🧠 Training Neural Network...")
    print("="*50)
    nn_model, _ = model_trainer.train_neural_network(
        X_train, X_test, y_train, y_test
    )
    
    if nn_model:
        nn_model.save('models/grade_nn.h5')
        print("✅ Neural Network saved to: models/grade_nn.h5")
    
    # Train LSTM
    print("\n" + "="*50)
    print("🔮 Training LSTM...")
    print("="*50)
    
    X_seq, y_seq = preprocessor.prepare_sequential_data(df, sequence_length=2)
    
    if len(X_seq) > 0:
        X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        lstm_model, _ = model_trainer.train_lstm(
            X_seq_train, X_seq_test, y_seq_train, y_seq_test
        )
        
        if lstm_model:
            lstm_model.save('models/lstm_model.h5')
            print("✅ LSTM saved to: models/lstm_model.h5")
    else:
        print("⚠️  Not enough data for LSTM training")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Show what was created
    print("\n📁 Models created:")
    for f in os.listdir('models'):
        print(f"   - models/{f}")
    
    print("\n📁 Reports created:")
    for f in os.listdir('reports'):
        print(f"   - reports/{f}")

if __name__ == "__main__":
    main()