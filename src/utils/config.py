"""
Configuration settings for the At-Risk Student Detection System
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, 'synthetic')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Data generation parameters
N_STUDENTS = 1000
N_DAYS = 90
RANDOM_SEED = 42

# Student IDs
STUDENT_ID_PREFIX = "STU_"
COURSE_IDS = ['CS101', 'MATH201', 'ENG102', 'PHY301', 'CHEM201']

# Risk level thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.3

# Model parameters
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ESTIMATORS = 100

# Feature engineering
MIN_DAYS_ACTIVE = 5  # Minimum days of activity required for feature creation

# Web app settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True