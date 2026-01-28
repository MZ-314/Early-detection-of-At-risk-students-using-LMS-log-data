"""
Feature Engineering Module
Transforms raw LMS log data into ML-ready features
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import MIN_DAYS_ACTIVE


class FeatureEngineer:
    """
    Create machine learning features from raw LMS log data
    """
    
    def __init__(self, min_days_active=MIN_DAYS_ACTIVE):
        """
        Initialize the feature engineer
        
        Args:
            min_days_active (int): Minimum days of activity required to create features
        """
        self.min_days_active = min_days_active
        
    def create_features(self, df):
        """
        Create ML features from raw log data
        
        Args:
            df (pd.DataFrame): Raw LMS log data
            
        Returns:
            pd.DataFrame: Feature DataFrame with one row per student
        """
        print(f"Creating features from {len(df)} log entries...")
        print(f"Processing {df['student_id'].nunique()} students...")
        
        features = []
        
        for student_id in df['student_id'].unique():
            student_data = df[df['student_id'] == student_id].sort_values('date')
            
            # Skip students with insufficient data
            if len(student_data) < self.min_days_active:
                continue
            
            feature_dict = self._create_student_features(student_id, student_data)
            features.append(feature_dict)
        
        feature_df = pd.DataFrame(features)
        print(f"Created {len(feature_df)} feature vectors with {len(feature_df.columns)-2} features each")
        print(f"At-risk students: {feature_df['at_risk'].sum()} ({feature_df['at_risk'].mean()*100:.1f}%)")
        
        return feature_df
    
    def _create_student_features(self, student_id, student_data):
        """
        Create features for a single student
        
        Args:
            student_id (str): Student identifier
            student_data (pd.DataFrame): Log data for this student
            
        Returns:
            dict: Dictionary of features for this student
        """
        feature_dict = {
            'student_id': student_id,
            'at_risk': student_data['at_risk'].iloc[0]
        }
        
        # Basic aggregation features
        feature_dict['total_logins'] = student_data['login_count'].sum()
        feature_dict['avg_session_duration'] = student_data['session_duration'].mean()
        feature_dict['total_page_views'] = student_data['page_views'].sum()
        feature_dict['total_forum_posts'] = student_data['forum_posts'].sum()
        feature_dict['total_assignments'] = student_data['assignments_submitted'].sum()
        feature_dict['avg_quiz_score'] = student_data['quiz_score'].mean()
        feature_dict['total_video_time'] = student_data['time_on_videos'].sum()
        feature_dict['total_downloads'] = student_data['downloads'].sum()
        feature_dict['days_active'] = len(student_data)
        
        # Engagement trend features
        if len(student_data) >= 14:
            recent_activity = student_data.tail(14)['login_count'].sum()
            early_activity = student_data.head(14)['login_count'].sum()
            feature_dict['engagement_trend'] = recent_activity / max(early_activity, 1)
        else:
            feature_dict['engagement_trend'] = 1.0
        
        # Consistency metrics (standard deviation)
        feature_dict['login_consistency'] = student_data['login_count'].std()
        feature_dict['session_consistency'] = student_data['session_duration'].std()
        
        # Performance indicators
        if len(student_data) >= 10:
            recent_scores = student_data['quiz_score'].tail(5).mean()
            early_scores = student_data['quiz_score'].head(5).mean()
            feature_dict['quiz_improvement'] = recent_scores - early_scores
        else:
            feature_dict['quiz_improvement'] = 0.0
        
        # Behavioral ratios
        feature_dict['pages_per_session'] = (
            feature_dict['total_page_views'] / max(feature_dict['total_logins'], 1)
        )
        feature_dict['posts_per_login'] = (
            feature_dict['total_forum_posts'] / max(feature_dict['total_logins'], 1)
        )
        
        # Activity rate (per day)
        feature_dict['login_rate'] = feature_dict['total_logins'] / feature_dict['days_active']
        feature_dict['assignment_rate'] = feature_dict['total_assignments'] / feature_dict['days_active']
        
        return feature_dict
    
    def save_features(self, feature_df, filename='features.csv', directory=None):
        """
        Save engineered features to CSV
        
        Args:
            feature_df (pd.DataFrame): Feature DataFrame to save
            filename (str): Name of the output file
            directory (str): Directory to save the file
        """
        if directory is None:
            from src.utils.config import PROCESSED_DATA_DIR
            directory = PROCESSED_DATA_DIR
            
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        feature_df.to_csv(filepath, index=False)
        print(f"Features saved to: {filepath}")
        return filepath
    
    def get_feature_names(self):
        """
        Get list of all feature names (excluding student_id and at_risk)
        
        Returns:
            list: List of feature names
        """
        return [
            'total_logins', 'avg_session_duration', 'total_page_views',
            'total_forum_posts', 'total_assignments', 'avg_quiz_score',
            'total_video_time', 'total_downloads', 'days_active',
            'engagement_trend', 'login_consistency', 'session_consistency',
            'quiz_improvement', 'pages_per_session', 'posts_per_login',
            'login_rate', 'assignment_rate'
        ]


if __name__ == "__main__":
    # Test the feature engineer
    from src.data.data_generator import LMSDataGenerator
    
    generator = LMSDataGenerator(n_students=100)
    raw_data = generator.generate_student_data(n_days=30)
    
    engineer = FeatureEngineer()
    features = engineer.create_features(raw_data)
    
    print("\nSample features:")
    print(features.head())
    print("\nFeature statistics:")
    print(features.describe())