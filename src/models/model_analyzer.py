"""
Model Analyzer
Visualization and analysis tools for model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


class ModelAnalyzer:
    """
    Analyze and visualize machine learning model performance
    """
    
    @staticmethod
    def plot_model_comparison(results, save_path=None):
        """
        Compare performance of multiple models
        
        Args:
            results (dict): Dictionary of model results from predictor.train_models()
            save_path (str, optional): Path to save the figure
        """
        model_names = list(results.keys())
        cv_scores = [results[name]['cv_mean'] for name in model_names]
        test_scores = [results[name]['test_auc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, cv_scores, width, label='CV AUC', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, test_scores, width, label='Test AUC', alpha=0.8, color='coral')
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    @staticmethod
    def plot_roc_curves(results, save_path=None):
        """
        Plot ROC curves for all models
        
        Args:
            results (dict): Dictionary of model results from predictor.train_models()
            save_path (str, optional): Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
        
        for idx, (name, result) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
            auc = result['test_auc']
            
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", 
                   linewidth=2, color=colors[idx % len(colors)])
        
        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, labels=['Normal', 'At-Risk'], save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_test (array): True labels
            y_pred (array): Predicted labels
            labels (list): Class labels
            save_path (str, optional): Path to save the figure
        """
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    @staticmethod
    def plot_feature_importance(importance_df, top_n=10, save_path=None):
        """
        Plot feature importance
        
        Args:
            importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
            top_n (int): Number of top features to display
            save_path (str, optional): Path to save the figure
        """
        if importance_df is None:
            print("Feature importance not available for this model")
            return None
        
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2., 
                   f'{width:.4f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    @staticmethod
    def print_classification_report(y_test, y_pred, target_names=['Normal', 'At-Risk']):
        """
        Print detailed classification report
        
        Args:
            y_test (array): True labels
            y_pred (array): Predicted labels
            target_names (list): Class names
        """
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=target_names))
        print("="*60 + "\n")
    
    @staticmethod
    def plot_risk_distribution(risk_scores, save_path=None):
        """
        Plot distribution of risk scores
        
        Args:
            risk_scores (array): Array of risk probability scores
            save_path (str, optional): Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(risk_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0.3, color='orange', linestyle='--', linewidth=2, label='Medium Risk Threshold')
        ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
        
        ax.set_xlabel('Risk Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Student Risk Scores', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    @staticmethod
    def create_summary_report(results, best_model_name):
        """
        Create a text summary report of model performance
        
        Args:
            results (dict): Dictionary of model results
            best_model_name (str): Name of the best performing model
        """
        print("\n" + "="*60)
        print("MODEL TRAINING SUMMARY REPORT")
        print("="*60)
        
        for name, result in results.items():
            marker = "★" if name == best_model_name else " "
            print(f"\n{marker} {name.upper()}")
            print(f"  Cross-Validation AUC: {result['cv_mean']:.4f} (±{result['cv_std']*2:.4f})")
            print(f"  Test AUC: {result['test_auc']:.4f}")
        
        print("\n" + "="*60)
        print(f"✓ BEST MODEL: {best_model_name.upper()}")
        print(f"  Test AUC: {results[best_model_name]['test_auc']:.4f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test the analyzer
    from src.data.data_generator import LMSDataGenerator
    from src.features.feature_engineer import FeatureEngineer
    from src.models.predictor import StudentRiskPredictor
    
    # Generate and prepare data
    generator = LMSDataGenerator(n_students=500)
    raw_data = generator.generate_student_data(n_days=60)
    
    engineer = FeatureEngineer()
    features = engineer.create_features(raw_data)
    
    # Train models
    predictor = StudentRiskPredictor()
    X, y = predictor.prepare_data(features)
    results = predictor.train_models(X, y)
    
    # Analyze results
    analyzer = ModelAnalyzer()
    analyzer.create_summary_report(results, predictor.best_model_name)
    analyzer.plot_model_comparison(results)
    analyzer.plot_roc_curves(results)
    
    # Get best model results
    best_result = results[predictor.best_model_name]
    analyzer.print_classification_report(best_result['y_test'], best_result['y_pred'])
    analyzer.plot_confusion_matrix(best_result['y_test'], best_result['y_pred'])
    
    # Feature importance
    importance = predictor.get_feature_importance()
    if importance is not None:
        analyzer.plot_feature_importance(importance)