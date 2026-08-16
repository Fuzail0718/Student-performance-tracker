import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

class StudentDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the student performance data"""
        print(f"Loading data from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Data cleaning
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.create_features(df)
        
        return df
    
    def clean_data(self, df):
        """Clean the raw data"""
        # Remove any leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        # Handle missing values if any
        df = df.dropna()
        
        # Convert marks to numeric
        df['InternalMarks'] = pd.to_numeric(df['InternalMarks'], errors='coerce')
        df['ExternalMarks'] = pd.to_numeric(df['ExternalMarks'], errors='coerce')
        df['TotalMarks'] = pd.to_numeric(df['TotalMarks'], errors='coerce')
        
        # Remove rows with invalid marks
        df = df.dropna(subset=['InternalMarks', 'ExternalMarks', 'TotalMarks'])
        
        return df
    
    def create_features(self, df):
        """Create features for modeling"""
        # Create binary target variable (Pass/Fail) - assuming 'P' means Pass
        df['pass_fail'] = (df['Result'] == 'P').astype(int)
        
        # Create subject category based on subject code
        df['subject_category'] = df['SubjectCode'].str.extract(r'([A-Z]+)')[0]
        
        # Create semester progression features
        student_semester_stats = df.groupby(['StudentID', 'Semester']).agg({
            'TotalMarks': ['mean', 'std', 'min', 'max'],
            'InternalMarks': 'mean',
            'ExternalMarks': 'mean'
        }).reset_index()
        
        student_semester_stats.columns = ['StudentID', 'Semester', 'avg_semester_marks', 
                                        'std_semester_marks', 'min_semester_marks', 
                                        'max_semester_marks', 'avg_internal_marks', 'avg_external_marks']
        
        # Merge semester stats back to main dataframe
        df = df.merge(student_semester_stats, on=['StudentID', 'Semester'], how='left')
        
        # Create student-level features
        student_features = df.groupby('StudentID').agg({
            'TotalMarks': ['mean', 'std', 'count'],
            'InternalMarks': 'mean',
            'ExternalMarks': 'mean',
            'Semester': 'max'
        }).reset_index()
        
        student_features.columns = ['StudentID', 'overall_avg_marks', 'overall_std_marks', 
                                  'total_subjects', 'overall_avg_internal', 'overall_avg_external', 
                                  'current_semester']
        
        # Merge student features
        df = df.merge(student_features, on='StudentID', how='left')
        
        # Create performance trend
        df['internal_external_ratio'] = df['InternalMarks'] / (df['ExternalMarks'] + 1)
        
        # Drop columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        return df
    
    def prepare_modeling_data(self, df, target_type='classification'):
        """Prepare data for machine learning models"""
        if df is None or len(df) == 0:
            return None, None, None
            
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['subject_category']
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
        
        # Select features for modeling
        feature_columns = [
            'Semester', 'InternalMarks', 'ExternalMarks', 'subject_category',
            'avg_semester_marks', 'std_semester_marks', 'min_semester_marks', 'max_semester_marks',
            'avg_internal_marks', 'avg_external_marks', 'overall_avg_marks', 'overall_std_marks',
            'total_subjects', 'overall_avg_internal', 'overall_avg_external', 'current_semester',
            'internal_external_ratio'
        ]
        
        # Remove any columns that might not exist
        available_features = [col for col in feature_columns if col in df_processed.columns]
        
        if len(available_features) == 0:
            print("No features available for modeling")
            return None, None, None
            
        X = df_processed[available_features]
        
        # Handle NaN values
        X = X.fillna(X.mean())
        
        if target_type == 'classification':
            y = df_processed['pass_fail']
        else:  # regression
            y = df_processed['TotalMarks']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=available_features)
        
        return X_scaled, y, available_features
    
    def prepare_sequential_data(self, df, sequence_length=2):
        """Prepare sequential data for LSTM - predict next semester performance"""
        if df is None or len(df) == 0:
            return np.array([]), np.array([])
            
        sequences = []
        targets = []
        
        for student_id in df['StudentID'].unique():
            student_data = df[df['StudentID'] == student_id].sort_values('Semester')
            
            # Get unique semesters for this student
            semesters = student_data['Semester'].unique()
            
            for i in range(len(semesters) - sequence_length):
                current_sequence_sems = semesters[i:i + sequence_length]
                target_sem = semesters[i + sequence_length]
                
                # Get data for current sequence
                seq_data = []
                for sem in current_sequence_sems:
                    sem_data = student_data[student_data['Semester'] == sem]
                    if len(sem_data) > 0:
                        # Use average marks for the semester
                        avg_marks = sem_data['TotalMarks'].mean()
                        seq_data.append(avg_marks)
                
                # Get target (average marks for target semester)
                target_data = student_data[student_data['Semester'] == target_sem]
                if len(seq_data) == sequence_length and len(target_data) > 0:
                    target_marks = target_data['TotalMarks'].mean()
                    sequences.append(seq_data)
                    targets.append(target_marks)
        
        if len(sequences) > 0:
            X_seq = np.array(sequences).reshape(-1, sequence_length, 1)
            y_seq = np.array(targets)
            return X_seq, y_seq
        else:
            return np.array([]), np.array([])
            
    def save_processed_data(self, df, output_path):
        """Save processed data to CSV"""
        if df is not None:
            df.to_csv(output_path, index=False)
            print(f"Saved processed data to: {output_path}")