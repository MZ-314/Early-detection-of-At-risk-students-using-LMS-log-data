"""
LMS Data Generator
Generates synthetic Learning Management System log data for testing and development
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import *


class LMSDataGenerator:
    """Generate synthetic LMS log data simulating student behavior"""
    
    def __init__(self, n_students=N_STUDENTS, random_seed=RANDOM_SEED):
        """
        Initialize the data generator
        
        Args:
            n_students (int): Number of students to generate data for
            random_seed (int): Random seed for reproducibility
        """
        self.n_students = n_students
        self.random_seed = random_seed
        self.student_ids = [f"{STUDENT_ID_PREFIX}{i:04d}" for i in range(1, n_students + 1)]
        self.course_ids = COURSE_IDS
        np.random.seed(random_seed)
        
    def generate_student_data(self, n_days=N_DAYS):
        """
        Generate synthetic LMS log data for all students
        
        Args:
            n_days (int): Number of days to generate data for
            
        Returns:
            pd.DataFrame: DataFrame containing log entries for all students
        """
        print(f"Generating data for {self.n_students} students over {n_days} days...")
        
        data = []
        
        for idx, student_id in enumerate(self.student_ids):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{self.n_students} students...")
            
            # Assign risk level (25% at-risk)
            risk_level = np.random.choice([0, 1], p=[0.75, 0.25])
            
            # Generate daily activities for each student
            for day in range(n_days):
                date = datetime.now() - timedelta(days=n_days - day)
                
                # Risk-based behavior patterns
                if risk_level == 1:  # At-risk student
                    login_prob = max(0.2, 0.8 - (day / n_days) * 0.6)  # Decreasing engagement
                    session_duration = np.random.exponential(15)  # Shorter sessions
                    page_views = np.random.poisson(8)
                    forum_posts = np.random.poisson(0.5)
                    assignment_submissions = 1 if np.random.random() < 0.4 else 0
                    quiz_score = np.random.normal(55, 15)
                else:  # Normal student
                    login_prob = 0.85
                    session_duration = np.random.exponential(45)
                    page_views = np.random.poisson(25)
                    forum_posts = np.random.poisson(2)
                    assignment_submissions = 1 if np.random.random() < 0.85 else 0
                    quiz_score = np.random.normal(78, 12)
                
                # Only create log entry if student logged in that day
                if np.random.random() < login_prob:
                    data.append({
                        'student_id': student_id,
                        'date': date.strftime('%Y-%m-%d'),
                        'login_count': np.random.poisson(2) + 1,
                        'session_duration': max(5, session_duration),
                        'page_views': max(1, page_views),
                        'forum_posts': forum_posts,
                        'assignments_submitted': assignment_submissions,
                        'quiz_attempts': np.random.poisson(1),
                        'quiz_score': max(0, min(100, quiz_score)),
                        'time_on_videos': np.random.exponential(20),
                        'downloads': np.random.poisson(3),
                        'at_risk': risk_level
                    })
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} log entries")
        print(f"At-risk students: {df.groupby('student_id')['at_risk'].first().sum()} ({df.groupby('student_id')['at_risk'].first().mean()*100:.1f}%)")
        
        return df
    
    def save_data(self, df, filename='lms_logs.csv', directory=SYNTHETIC_DATA_DIR):
        """
        Save generated data to CSV
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Name of the output file
            directory (str): Directory to save the file
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        return filepath


if __name__ == "__main__":
    # Test the data generator
    generator = LMSDataGenerator(n_students=100)
    data = generator.generate_student_data(n_days=30)
    print("\nSample data:")
    print(data.head(10))
    print("\nData shape:", data.shape)
    print("\nData info:")
    print(data.info())