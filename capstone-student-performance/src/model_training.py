import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

class StudentPerformanceModels:
    def __init__(self):
        self.models = {}
        
    def train_random_forest(self, X_train, X_test, y_train, y_test, problem_type='classification'):
        """Train Random Forest model"""
        if X_train is None or len(X_train) == 0:
            print("No training data available")
            return None, None, None
            
        if problem_type == 'classification':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Random Forest Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
        else:  # regression
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            print(f"Random Forest RMSE: {rmse:.4f}")
            feature_importance = None
        
        self.models['random_forest'] = rf
        return rf, feature_importance, y_pred
    
    def train_neural_network(self, X_train, X_test, y_train, y_test, problem_type='classification'):
        """Train Neural Network model"""
        if X_train is None or len(X_train) == 0:
            print("No training data available")
            return None, None
            
        model = Sequential()
        
        if problem_type == 'classification':
            # Classification network
            model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
            model.add(Dropout(0.3))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
            
            history = model.fit(X_train, y_train, 
                              epochs=100, 
                              batch_size=32,
                              validation_split=0.2,
                              verbose=1)
            
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Neural Network Accuracy: {accuracy:.4f}")
            
        else:  # regression
            # Regression network
            model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='linear'))
            
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mse',
                         metrics=['mae'])
            
            history = model.fit(X_train, y_train,
                              epochs=150,
                              batch_size=32,
                              validation_split=0.2,
                              verbose=1)
            
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            print(f"Neural Network MAE: {mae:.4f}")
        
        self.models['neural_network'] = model
        return model, history
    
    def train_lstm(self, X_seq_train, X_seq_test, y_seq_train, y_seq_test):
        """Train LSTM model for sequential prediction"""
        if X_seq_train is None or len(X_seq_train) == 0:
            print("Not enough sequential data for LSTM training")
            return None, None
            
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, 
                 input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        history = model.fit(X_seq_train, y_seq_train,
                          epochs=100,
                          batch_size=16,
                          validation_split=0.2,
                          verbose=1)
        
        loss, mae = model.evaluate(X_seq_test, y_seq_test, verbose=0)
        print(f"LSTM MAE: {mae:.4f}")
        
        self.models['lstm'] = model
        return model, history
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for Random Forest"""
        if X_train is None or len(X_train) == 0:
            print("No training data for hyperparameter tuning")
            return None
            
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                 cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def plot_feature_importance(self, feature_importance, top_n=15):
        """Plot feature importance"""
        if feature_importance is None or len(feature_importance) == 0:
            print("No feature importance data to plot")
            return
            
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Create reports directory if it doesn't exist
        os.makedirs('../reports', exist_ok=True)
        plt.savefig('../reports/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, model, model_name, save_path):
        """Save model to disk"""
        if model is None:
            print(f"Model {model_name} is None, cannot save")
            return
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if model_name == 'random_forest':
            joblib.dump(model, save_path)
        else:
            model.save(save_path)
        print(f"Model saved to: {save_path}")