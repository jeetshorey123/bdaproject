"""
Financial Risk Management Project
=================================

This project implements a comprehensive financial risk management system
following the methodology outlined in the project requirements:

Phase 1: Data Collection and Cleaning
Phase 2: Exploratory Data Analysis (EDA)
Phase 3: Predictive Modeling
Phase 4: Visualization and Reporting

Author: Financial Risk Management Team
Date: October 31, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialRiskManager:
    """
    A comprehensive Financial Risk Management system that handles data loading,
    cleaning, analysis, and predictive modeling for loan default prediction.
    """
    
    def __init__(self, accepted_data_path, rejected_data_path):
        """
        Initialize the Financial Risk Manager
        
        Parameters:
        accepted_data_path (str): Path to accepted loans dataset
        rejected_data_path (str): Path to rejected loans dataset
        """
        self.accepted_data_path = accepted_data_path
        self.rejected_data_path = rejected_data_path
        self.accepted_data = None
        self.rejected_data = None
        self.combined_data = None
        self.processed_data = None
        self.models = {}
        self.model_performance = {}
        
    def load_data(self, sample_size=None):
        """
        Load and combine accepted and rejected loan datasets
        
        Parameters:
        sample_size (int): Optional sample size for large datasets
        """
        print("🔄 Phase 1: Data Collection and Loading")
        print("=" * 50)
        
        try:
            # Load accepted loans data
            print("📊 Loading accepted loans data...")
            if sample_size:
                self.accepted_data = pd.read_csv(self.accepted_data_path, nrows=sample_size)
            else:
                self.accepted_data = pd.read_csv(self.accepted_data_path, low_memory=False)
            
            self.accepted_data['loan_status_binary'] = 0  # 0 for accepted (good)
            print(f"✅ Loaded {len(self.accepted_data):,} accepted loan records")
            
            # Load rejected loans data  
            print("📊 Loading rejected loans data...")
            if sample_size:
                self.rejected_data = pd.read_csv(self.rejected_data_path, nrows=sample_size)
            else:
                self.rejected_data = pd.read_csv(self.rejected_data_path, low_memory=False)
                
            self.rejected_data['loan_status_binary'] = 1  # 1 for rejected (risky)
            print(f"✅ Loaded {len(self.rejected_data):,} rejected loan records")
            
            # Get common columns for merging
            common_columns = list(set(self.accepted_data.columns) & set(self.rejected_data.columns))
            print(f"📋 Found {len(common_columns)} common columns between datasets")
            
            # Combine datasets using common columns
            accepted_subset = self.accepted_data[common_columns]
            rejected_subset = self.rejected_data[common_columns]
            
            self.combined_data = pd.concat([accepted_subset, rejected_subset], 
                                         ignore_index=True, sort=False)
            
            print(f"🎯 Combined dataset shape: {self.combined_data.shape}")
            print(f"📈 Total records: {len(self.combined_data):,}")
            
            return self.combined_data
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def clean_data(self):
        """
        Clean and preprocess the combined dataset
        """
        print("\n🧹 Data Cleaning and Preprocessing")
        print("=" * 40)
        
        if self.combined_data is None:
            print("❌ No data loaded. Please load data first.")
            return None
        
        # Create a copy for processing
        self.processed_data = self.combined_data.copy()
        
        # Display initial data info
        print(f"📊 Initial dataset shape: {self.processed_data.shape}")
        print(f"🔍 Missing values per column:")
        missing_data = self.processed_data.isnull().sum()
        missing_percentage = (missing_data / len(self.processed_data)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percentage
        }).sort_values('Missing_Percentage', ascending=False)
        print(missing_df.head(10))
        
        # Remove columns with high missing values (>80%)
        high_missing_cols = missing_df[missing_df['Missing_Percentage'] > 80].index.tolist()
        if high_missing_cols:
            print(f"\n🗑️ Removing {len(high_missing_cols)} columns with >80% missing values")
            self.processed_data = self.processed_data.drop(columns=high_missing_cols)
        
        # Handle numeric columns
        numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'loan_status_binary' in numeric_columns:
            numeric_columns.remove('loan_status_binary')
        
        print(f"🔢 Processing {len(numeric_columns)} numeric columns...")
        
        # Impute missing values for numeric columns
        if numeric_columns:
            numeric_imputer = SimpleImputer(strategy='median')
            self.processed_data[numeric_columns] = numeric_imputer.fit_transform(
                self.processed_data[numeric_columns]
            )
        
        # Handle categorical columns
        categorical_columns = self.processed_data.select_dtypes(include=['object']).columns.tolist()
        print(f"📝 Processing {len(categorical_columns)} categorical columns...")
        
        if categorical_columns:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.processed_data[categorical_columns] = categorical_imputer.fit_transform(
                self.processed_data[categorical_columns]
            )
        
        # Remove duplicates
        initial_rows = len(self.processed_data)
        self.processed_data = self.processed_data.drop_duplicates()
        removed_duplicates = initial_rows - len(self.processed_data)
        
        if removed_duplicates > 0:
            print(f"🗑️ Removed {removed_duplicates:,} duplicate rows")
        
        print(f"✅ Final cleaned dataset shape: {self.processed_data.shape}")
        print(f"🎯 Cleaning completed successfully!")
        
        return self.processed_data
    
    def exploratory_data_analysis(self):
        """
        Perform comprehensive Exploratory Data Analysis
        """
        print("\n📊 Phase 2: Exploratory Data Analysis (EDA)")
        print("=" * 50)
        
        if self.processed_data is None:
            print("❌ No processed data available. Please clean data first.")
            return
        
        # Create visualizations directory
        import os
        viz_dir = "financial_risk_visualizations"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # 1. Target Variable Distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        target_counts = self.processed_data['loan_status_binary'].value_counts()
        plt.pie(target_counts.values, labels=['Accepted (Low Risk)', 'Rejected (High Risk)'], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Loan Status Distribution')
        
        plt.subplot(2, 2, 2)
        sns.countplot(data=self.processed_data, x='loan_status_binary')
        plt.title('Loan Status Count')
        plt.xlabel('Loan Status (0=Accepted, 1=Rejected)')
        
        # 2. Numeric feature distributions
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'loan_status_binary' in numeric_cols:
            numeric_cols.remove('loan_status_binary')
        
        # Select top numeric features for analysis
        top_numeric_cols = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        
        if len(top_numeric_cols) >= 2:
            plt.subplot(2, 2, 3)
            self.processed_data[top_numeric_cols[0]].hist(bins=30, alpha=0.7)
            plt.title(f'Distribution of {top_numeric_cols[0]}')
            plt.xlabel(top_numeric_cols[0])
            plt.ylabel('Frequency')
            
            if len(top_numeric_cols) >= 2:
                plt.subplot(2, 2, 4)
                self.processed_data[top_numeric_cols[1]].hist(bins=30, alpha=0.7, color='orange')
                plt.title(f'Distribution of {top_numeric_cols[1]}')
                plt.xlabel(top_numeric_cols[1])
                plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/basic_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Correlation Analysis
        print("\n🔗 Correlation Analysis")
        correlation_cols = numeric_cols[:10] + ['loan_status_binary']  # Top 10 numeric + target
        correlation_data = self.processed_data[correlation_cols]
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = correlation_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Feature Importance Analysis
        print("\n📈 Top Correlations with Loan Status:")
        target_correlations = correlation_matrix['loan_status_binary'].abs().sort_values(ascending=False)
        print(target_correlations.head(10))
        
        # 5. Risk Factor Analysis
        plt.figure(figsize=(15, 10))
        
        # Top correlated features with target
        top_features = target_correlations.head(6).index.tolist()
        if 'loan_status_binary' in top_features:
            top_features.remove('loan_status_binary')
        
        for i, feature in enumerate(top_features[:4], 1):
            plt.subplot(2, 2, i)
            if feature in self.processed_data.columns:
                # Box plot for numeric features
                sns.boxplot(data=self.processed_data, x='loan_status_binary', y=feature)
                plt.title(f'Risk Analysis: {feature}')
                plt.xlabel('Loan Status (0=Accepted, 1=Rejected)')
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/risk_factor_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. Summary Statistics
        print("\n📊 Summary Statistics by Loan Status:")
        summary_stats = self.processed_data.groupby('loan_status_binary')[numeric_cols[:5]].agg([
            'mean', 'median', 'std'
        ]).round(2)
        print(summary_stats)
        
        # Save EDA summary
        self.save_eda_summary(target_correlations, summary_stats, viz_dir)
        
        print(f"\n✅ EDA completed! Visualizations saved in '{viz_dir}' directory")
    
    def save_eda_summary(self, correlations, summary_stats, viz_dir):
        """
        Save EDA summary to a text file
        """
        with open(f'{viz_dir}/eda_summary.txt', 'w') as f:
            f.write("FINANCIAL RISK MANAGEMENT - EDA SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Dataset Overview:\n")
            f.write(f"Total Records: {len(self.processed_data):,}\n")
            f.write(f"Features: {self.processed_data.shape[1]}\n")
            f.write(f"Accepted Loans: {(self.processed_data['loan_status_binary'] == 0).sum():,}\n")
            f.write(f"Rejected Loans: {(self.processed_data['loan_status_binary'] == 1).sum():,}\n\n")
            
            f.write("Top Risk Factors (Correlation with Loan Status):\n")
            f.write("-" * 40 + "\n")
            for feature, corr in correlations.head(10).items():
                if feature != 'loan_status_binary':
                    f.write(f"{feature}: {corr:.3f}\n")
    
    def prepare_modeling_data(self):
        """
        Prepare data for machine learning modeling
        """
        print("\n🔧 Preparing Data for Machine Learning")
        print("=" * 40)
        
        if self.processed_data is None:
            print("❌ No processed data available.")
            return None, None, None, None
        
        # Select features for modeling
        numeric_features = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'loan_status_binary' in numeric_features:
            numeric_features.remove('loan_status_binary')
        
        # Limit features to prevent overfitting and improve performance
        selected_features = numeric_features[:15]  # Top 15 numeric features
        
        print(f"📊 Selected {len(selected_features)} features for modeling:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i:2d}. {feature}")
        
        # Prepare feature matrix and target
        X = self.processed_data[selected_features]
        y = self.processed_data['loan_status_binary']
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data prepared successfully!")
        print(f"📈 Training set: {X_train.shape[0]:,} samples")
        print(f"📊 Test set: {X_test.shape[0]:,} samples")
        print(f"🎯 Class distribution in training:")
        print(f"   - Accepted (0): {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
        print(f"   - Rejected (1): {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate machine learning models
        """
        print("\n🤖 Phase 3: Predictive Modeling")
        print("=" * 40)
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n🔄 Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # Store results
            results[model_name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'confusion_matrix': confusion_matrix(y_test, y_pred_test),
                'classification_report': classification_report(y_test, y_pred_test)
            }
            
            print(f"✅ {model_name} Results:")
            print(f"   📈 Training Accuracy: {train_accuracy:.4f}")
            print(f"   📊 Test Accuracy: {test_accuracy:.4f}")
            print(f"   🎯 Generalization Gap: {(train_accuracy - test_accuracy):.4f}")
        
        self.models = results
        return results
    
    def evaluate_models(self):
        """
        Evaluate and visualize model performance
        """
        print("\n📊 Model Evaluation and Comparison")
        print("=" * 40)
        
        if not self.models:
            print("❌ No trained models available.")
            return
        
        # Create evaluation visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model accuracy comparison
        model_names = list(self.models.keys())
        train_accuracies = [self.models[name]['train_accuracy'] for name in model_names]
        test_accuracies = [self.models[name]['test_accuracy'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_accuracies, width, label='Training', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_accuracies, width, label='Testing', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confusion matrices
        for i, (model_name, results) in enumerate(self.models.items()):
            ax = axes[0, 1] if i == 0 else axes[1, 0]
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name} - Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Accepted', 'Rejected'])
            ax.set_yticklabels(['Accepted', 'Rejected'])
        
        # Feature importance for Random Forest
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']['model']
            feature_importance = rf_model.feature_importances_
            feature_names = rf_model.feature_names_in_
            
            # Get top 10 features
            top_indices = np.argsort(feature_importance)[-10:]
            top_features = feature_names[top_indices]
            top_importance = feature_importance[top_indices]
            
            axes[1, 1].barh(range(len(top_features)), top_importance)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features)
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('Top 10 Feature Importance (Random Forest)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('financial_risk_visualizations/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed classification reports
        print("\n📋 Detailed Classification Reports:")
        print("=" * 50)
        
        for model_name, results in self.models.items():
            print(f"\n🤖 {model_name}:")
            print("-" * 30)
            print(results['classification_report'])
    
    def generate_risk_insights(self):
        """
        Generate actionable insights for risk management
        """
        print("\n💡 Risk Management Insights")
        print("=" * 40)
        
        insights = []
        
        # Model performance insights
        best_model = max(self.models.keys(), 
                        key=lambda x: self.models[x]['test_accuracy'])
        best_accuracy = self.models[best_model]['test_accuracy']
        
        insights.append(f"🏆 Best performing model: {best_model} with {best_accuracy:.2%} accuracy")
        
        # Risk distribution insights
        total_loans = len(self.processed_data)
        rejected_rate = (self.processed_data['loan_status_binary'] == 1).mean()
        
        insights.append(f"📊 Overall rejection rate: {rejected_rate:.2%}")
        insights.append(f"💰 Potential financial exposure from {total_loans:,} loan applications")
        
        # Feature importance insights
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']['model']
            feature_importance = rf_model.feature_importances_
            feature_names = rf_model.feature_names_in_
            
            top_feature = feature_names[np.argmax(feature_importance)]
            insights.append(f"🔍 Most important risk factor: {top_feature}")
        
        # Business recommendations
        recommendations = [
            "🎯 Implement automated screening using the trained models",
            "📈 Focus on monitoring high-importance risk factors",
            "🔄 Regularly retrain models with new data",
            "⚠️ Set up alerts for high-risk loan applications",
            "📋 Create tiered risk assessment categories"
        ]
        
        print("Key Insights:")
        for insight in insights:
            print(f"  {insight}")
        
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        
        # Save insights to file
        with open('financial_risk_visualizations/risk_insights.txt', 'w') as f:
            f.write("FINANCIAL RISK MANAGEMENT INSIGHTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Key Insights:\n")
            for insight in insights:
                f.write(f"  {insight}\n")
            
            f.write("\nRecommendations:\n")
            for rec in recommendations:
                f.write(f"  {rec}\n")
        
        print(f"\n✅ Insights saved to 'financial_risk_visualizations/risk_insights.txt'")
    
    def create_risk_dashboard_data(self):
        """
        Prepare data for dashboard visualization (would be used with Power BI)
        """
        print("\n📊 Preparing Dashboard Data")
        print("=" * 30)
        
        # Create summary datasets for visualization
        summary_data = {}
        
        # Risk distribution summary
        risk_summary = self.processed_data['loan_status_binary'].value_counts()
        summary_data['risk_distribution'] = {
            'Accepted': risk_summary[0],
            'Rejected': risk_summary[1],
            'Total': len(self.processed_data),
            'Rejection_Rate': risk_summary[1] / len(self.processed_data)
        }
        
        # Feature correlation summary
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'loan_status_binary' in numeric_cols:
            numeric_cols.remove('loan_status_binary')
        
        correlations = self.processed_data[numeric_cols + ['loan_status_binary']].corr()['loan_status_binary']
        summary_data['top_risk_factors'] = correlations.abs().sort_values(ascending=False).head(10).to_dict()
        
        # Model performance summary
        model_performance = {}
        for model_name, results in self.models.items():
            model_performance[model_name] = {
                'train_accuracy': results['train_accuracy'],
                'test_accuracy': results['test_accuracy'],
                'generalization_gap': results['train_accuracy'] - results['test_accuracy']
            }
        summary_data['model_performance'] = model_performance
        
        # Save summary data
        import json
        with open('financial_risk_visualizations/dashboard_data.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print("✅ Dashboard data prepared and saved")
        return summary_data
    
    def run_complete_analysis(self, sample_size=None):
        """
        Run the complete financial risk management analysis pipeline
        """
        print("🚀 FINANCIAL RISK MANAGEMENT SYSTEM")
        print("=" * 60)
        print("Starting comprehensive risk analysis pipeline...")
        print()
        
        try:
            # Phase 1: Data Loading and Cleaning
            self.load_data(sample_size)
            if self.combined_data is None:
                return
            
            self.clean_data()
            if self.processed_data is None:
                return
            
            # Phase 2: Exploratory Data Analysis
            self.exploratory_data_analysis()
            
            # Phase 3: Predictive Modeling
            X_train, X_test, y_train, y_test = self.prepare_modeling_data()
            if X_train is None:
                return
            
            self.train_models(X_train, X_test, y_train, y_test)
            self.evaluate_models()
            
            # Phase 4: Insights and Reporting
            self.generate_risk_insights()
            self.create_risk_dashboard_data()
            
            print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print("📁 All results saved in 'financial_risk_visualizations' directory")
            print("📊 Check the generated visualizations and reports")
            print("💡 Review risk insights for actionable recommendations")
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()

# Main execution
if __name__ == "__main__":
    # Initialize the Financial Risk Manager
    risk_manager = FinancialRiskManager(
        accepted_data_path="accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
        rejected_data_path="rejected_2007_to_2018q4.csv/rejected_2007_to_2018Q4.csv"
    )
    
    # Run complete analysis
    # Use sample_size for faster processing with large datasets
    # For full analysis, remove the sample_size parameter
    risk_manager.run_complete_analysis(sample_size=50000)