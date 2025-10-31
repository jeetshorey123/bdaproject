"""
Financial Risk Management Project - Complete Analysis
=====================================================

This project implements a comprehensive financial risk management system
that analyzes both accepted and rejected loan datasets separately and 
provides insights for risk management.

Phase 1: Data Collection and Cleaning
Phase 2: Exploratory Data Analysis (EDA) 
Phase 3: Predictive Modeling (on accepted loans with loan outcomes)
Phase 4: Risk Assessment and Business Insights

Author: Financial Risk Management Team
Date: October 31, 2025
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import json
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FinancialRiskManager:
    """
    A comprehensive Financial Risk Management system that analyzes
    loan data for risk assessment and predictive modeling.
    """
    
    def __init__(self, accepted_data_path, rejected_data_path):
        """
        Initialize the Financial Risk Manager
        """
        self.accepted_data_path = accepted_data_path
        self.rejected_data_path = rejected_data_path
        self.accepted_data = None
        self.rejected_data = None
        self.processed_accepted = None
        self.processed_rejected = None
        self.models = {}
        self.analysis_results = {}
        
    def load_data(self, sample_size=None):
        """
        Load accepted and rejected loan datasets
        """
        print("🔄 Phase 1: Data Collection and Loading")
        print("=" * 50)
        
        try:
            # Load accepted loans data
            print("📊 Loading accepted loans data...")
            if sample_size:
                print(f"   Using sample size: {sample_size:,} records")
                self.accepted_data = pd.read_csv(self.accepted_data_path, nrows=sample_size)
            else:
                print("   Loading COMPLETE dataset (this may take time for large files)...")
                # Use chunked reading for very large files to manage memory
                chunk_size = 50000  # Process in chunks of 50k rows
                chunks = []
                total_rows = 0
                
                print("   Reading in chunks for memory efficiency...")
                try:
                    for chunk in pd.read_csv(self.accepted_data_path, chunksize=chunk_size, low_memory=False):
                        chunks.append(chunk)
                        total_rows += len(chunk)
                        if len(chunks) % 10 == 0:  # Print progress every 10 chunks
                            print(f"   Processed {total_rows:,} accepted loan rows so far...")
                    
                    self.accepted_data = pd.concat(chunks, ignore_index=True)
                    print(f"   ✅ Completed loading {total_rows:,} accepted loan records")
                except Exception as chunk_error:
                    print(f"   ⚠️  Chunked reading failed, trying direct load: {chunk_error}")
                    self.accepted_data = pd.read_csv(self.accepted_data_path, low_memory=False)
            
            print(f"✅ Loaded {len(self.accepted_data):,} accepted loan records")
            print(f"📋 Accepted loans have {self.accepted_data.shape[1]} features")
            print(f"📋 Sample features: {list(self.accepted_data.columns[:10])}")
            
            # Load rejected loans data  
            print("\n📊 Loading rejected loans data...")
            if sample_size:
                print(f"   Using sample size: {sample_size:,} records")
                self.rejected_data = pd.read_csv(self.rejected_data_path, nrows=sample_size)
            else:
                print("   Loading COMPLETE dataset (this may take time for large files)...")
                # Use chunked reading for very large files
                chunk_size = 50000
                chunks = []
                total_rows = 0
                
                print("   Reading in chunks for memory efficiency...")
                try:
                    for chunk in pd.read_csv(self.rejected_data_path, chunksize=chunk_size, low_memory=False):
                        chunks.append(chunk)
                        total_rows += len(chunk)
                        if len(chunks) % 20 == 0:  # Print progress every 20 chunks
                            print(f"   Processed {total_rows:,} rejected loan rows so far...")
                    
                    self.rejected_data = pd.concat(chunks, ignore_index=True)
                    print(f"   ✅ Completed loading {total_rows:,} rejected loan records")
                except Exception as chunk_error:
                    print(f"   ⚠️  Chunked reading failed, trying direct load: {chunk_error}")
                    self.rejected_data = pd.read_csv(self.rejected_data_path, low_memory=False)
                
            print(f"✅ Loaded {len(self.rejected_data):,} rejected loan records")
            print(f"📋 Rejected loans have {self.rejected_data.shape[1]} features")
            print(f"📋 Sample features: {list(self.rejected_data.columns[:10])}")
            
            # Calculate total memory usage
            accepted_memory = self.accepted_data.memory_usage(deep=True).sum() / 1024**2
            rejected_memory = self.rejected_data.memory_usage(deep=True).sum() / 1024**2
            total_memory = accepted_memory + rejected_memory
            
            print(f"\n💾 Memory Usage Summary:")
            print(f"   Accepted data: {accepted_memory:.1f} MB")
            print(f"   Rejected data: {rejected_memory:.1f} MB")
            print(f"   Total memory: {total_memory:.1f} MB")
            
            if total_memory > 1000:  # Warn if over 1GB
                print(f"   ⚠️  Large dataset detected ({total_memory:.1f} MB)")
                print(f"   💡 Consider using sample_size if memory issues occur")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            print("💡 If memory error occurs, try using sample_size parameter")
            print("💡 Example: risk_manager.run_complete_analysis(sample_size=100000)")
            return False
    
    def clean_accepted_data(self):
        """
        Clean and preprocess the accepted loans dataset
        """
        print("\n🧹 Cleaning Accepted Loans Data")
        print("=" * 40)
        
        if self.accepted_data is None:
            print("❌ No accepted data loaded.")
            return None
        
        # Create a copy for processing
        self.processed_accepted = self.accepted_data.copy()
        
        print(f"📊 Initial shape: {self.processed_accepted.shape}")
        
        # Remove columns with high missing values (>80%)
        missing_percentage = (self.processed_accepted.isnull().sum() / len(self.processed_accepted)) * 100
        high_missing_cols = missing_percentage[missing_percentage > 80].index.tolist()
        
        if high_missing_cols:
            print(f"🗑️ Removing {len(high_missing_cols)} columns with >80% missing values")
            self.processed_accepted = self.processed_accepted.drop(columns=high_missing_cols)
        
        # Focus on key financial features
        key_features = []
        all_columns = self.processed_accepted.columns.tolist()
        
        # Identify important loan features
        important_patterns = ['loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'grade', 
                            'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 
                            'verification_status', 'loan_status', 'purpose', 'dti', 
                            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 
                            'revol_bal', 'revol_util', 'total_acc', 'fico_range_low', 
                            'fico_range_high']
        
        for pattern in important_patterns:
            matching_cols = [col for col in all_columns if pattern.lower() in col.lower()]
            key_features.extend(matching_cols)
        
        # Remove duplicates and keep available columns
        key_features = list(set(key_features))
        available_features = [col for col in key_features if col in self.processed_accepted.columns]
        
        if available_features:
            self.processed_accepted = self.processed_accepted[available_features]
            print(f"📋 Selected {len(available_features)} key features")
        
        # Handle numeric columns
        numeric_columns = self.processed_accepted.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            numeric_imputer = SimpleImputer(strategy='median')
            self.processed_accepted[numeric_columns] = numeric_imputer.fit_transform(
                self.processed_accepted[numeric_columns]
            )
            print(f"🔢 Processed {len(numeric_columns)} numeric features")
        
        # Handle categorical columns
        categorical_columns = self.processed_accepted.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.processed_accepted[categorical_columns] = categorical_imputer.fit_transform(
                self.processed_accepted[categorical_columns]
            )
            print(f"📝 Processed {len(categorical_columns)} categorical features")
        
        # Remove duplicates
        initial_rows = len(self.processed_accepted)
        self.processed_accepted = self.processed_accepted.drop_duplicates()
        removed_duplicates = initial_rows - len(self.processed_accepted)
        
        print(f"✅ Final accepted data shape: {self.processed_accepted.shape}")
        return self.processed_accepted
    
    def clean_rejected_data(self):
        """
        Clean and preprocess the rejected loans dataset
        """
        print("\n🧹 Cleaning Rejected Loans Data")
        print("=" * 40)
        
        if self.rejected_data is None:
            print("❌ No rejected data loaded.")
            return None
        
        # Create a copy for processing
        self.processed_rejected = self.rejected_data.copy()
        
        print(f"📊 Initial shape: {self.processed_rejected.shape}")
        
        # Handle numeric columns
        numeric_columns = self.processed_rejected.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            numeric_imputer = SimpleImputer(strategy='median')
            self.processed_rejected[numeric_columns] = numeric_imputer.fit_transform(
                self.processed_rejected[numeric_columns]
            )
            print(f"🔢 Processed {len(numeric_columns)} numeric features")
        
        # Handle categorical columns  
        categorical_columns = self.processed_rejected.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.processed_rejected[categorical_columns] = categorical_imputer.fit_transform(
                self.processed_rejected[categorical_columns]
            )
            print(f"📝 Processed {len(categorical_columns)} categorical features")
        
        # Remove duplicates
        initial_rows = len(self.processed_rejected)
        self.processed_rejected = self.processed_rejected.drop_duplicates()
        removed_duplicates = initial_rows - len(self.processed_rejected)
        
        print(f"✅ Final rejected data shape: {self.processed_rejected.shape}")
        return self.processed_rejected
    
    def analyze_accepted_loans(self):
        """
        Perform detailed analysis of accepted loans
        """
        print("\n📊 Phase 2A: Analyzing Accepted Loans")
        print("=" * 45)
        
        if self.processed_accepted is None:
            print("❌ No processed accepted data available.")
            return {}
        
        analysis = {}
        
        # Basic statistics
        analysis['total_accepted'] = len(self.processed_accepted)
        analysis['features_count'] = self.processed_accepted.shape[1]
        
        print(f"📈 Total accepted loans: {analysis['total_accepted']:,}")
        print(f"📋 Available features: {analysis['features_count']}")
        
        # Loan amount analysis
        if 'loan_amnt' in self.processed_accepted.columns:
            loan_stats = {
                'mean': float(self.processed_accepted['loan_amnt'].mean()),
                'median': float(self.processed_accepted['loan_amnt'].median()),
                'min': float(self.processed_accepted['loan_amnt'].min()),
                'max': float(self.processed_accepted['loan_amnt'].max()),
                'std': float(self.processed_accepted['loan_amnt'].std())
            }
            analysis['loan_amount_stats'] = loan_stats
            
            print(f"\n💰 Loan Amount Analysis:")
            print(f"   Average: ${loan_stats['mean']:,.2f}")
            print(f"   Median: ${loan_stats['median']:,.2f}")
            print(f"   Range: ${loan_stats['min']:,.2f} - ${loan_stats['max']:,.2f}")
        
        # Interest rate analysis
        if 'int_rate' in self.processed_accepted.columns:
            # Convert interest rate to numeric if it's a string with %
            int_rate_col = self.processed_accepted['int_rate']
            if int_rate_col.dtype == 'object':
                int_rate_col = pd.to_numeric(int_rate_col.str.replace('%', ''), errors='coerce')
            
            rate_stats = {
                'mean': float(int_rate_col.mean()),
                'median': float(int_rate_col.median()),
                'min': float(int_rate_col.min()),
                'max': float(int_rate_col.max())
            }
            analysis['interest_rate_stats'] = rate_stats
            
            print(f"\n📈 Interest Rate Analysis:")
            print(f"   Average: {rate_stats['mean']:.2f}%")
            print(f"   Median: {rate_stats['median']:.2f}%")
            print(f"   Range: {rate_stats['min']:.2f}% - {rate_stats['max']:.2f}%")
        
        # Grade distribution
        if 'grade' in self.processed_accepted.columns:
            grade_dist = self.processed_accepted['grade'].value_counts().to_dict()
            analysis['grade_distribution'] = grade_dist
            
            print(f"\n🏆 Loan Grade Distribution:")
            for grade, count in sorted(grade_dist.items()):
                percentage = (count / len(self.processed_accepted)) * 100
                print(f"   Grade {grade}: {count:,} ({percentage:.1f}%)")
        
        # Loan status analysis (if available)
        if 'loan_status' in self.processed_accepted.columns:
            status_dist = self.processed_accepted['loan_status'].value_counts().to_dict()
            analysis['loan_status_distribution'] = status_dist
            
            print(f"\n📊 Loan Status Distribution:")
            for status, count in status_dist.items():
                percentage = (count / len(self.processed_accepted)) * 100
                print(f"   {status}: {count:,} ({percentage:.1f}%)")
        
        return analysis
    
    def analyze_rejected_loans(self):
        """
        Perform detailed analysis of rejected loans
        """
        print("\n📊 Phase 2B: Analyzing Rejected Loans")
        print("=" * 45)
        
        if self.processed_rejected is None:
            print("❌ No processed rejected data available.")
            return {}
        
        analysis = {}
        
        # Basic statistics
        analysis['total_rejected'] = len(self.processed_rejected)
        analysis['features_count'] = self.processed_rejected.shape[1]
        
        print(f"📈 Total rejected loans: {analysis['total_rejected']:,}")
        print(f"📋 Available features: {analysis['features_count']}")
        print(f"📋 Rejection features: {list(self.processed_rejected.columns)}")
        
        # Amount requested analysis
        amount_cols = [col for col in self.processed_rejected.columns 
                      if 'amount' in col.lower() or 'requested' in col.lower()]
        
        if amount_cols:
            amount_col = amount_cols[0]
            if self.processed_rejected[amount_col].dtype in ['object']:
                # Try to convert to numeric
                amount_values = pd.to_numeric(self.processed_rejected[amount_col], errors='coerce')
            else:
                amount_values = self.processed_rejected[amount_col]
            
            amount_values = amount_values.dropna()
            if len(amount_values) > 0:
                amount_stats = {
                    'mean': float(amount_values.mean()),
                    'median': float(amount_values.median()),
                    'min': float(amount_values.min()),
                    'max': float(amount_values.max())
                }
                analysis['requested_amount_stats'] = amount_stats
                
                print(f"\n💰 Requested Amount Analysis:")
                print(f"   Average: ${amount_stats['mean']:,.2f}")
                print(f"   Median: ${amount_stats['median']:,.2f}")
                print(f"   Range: ${amount_stats['min']:,.2f} - ${amount_stats['max']:,.2f}")
        
        # Risk score analysis
        risk_cols = [col for col in self.processed_rejected.columns 
                    if 'risk' in col.lower() or 'score' in col.lower()]
        
        if risk_cols:
            risk_col = risk_cols[0]
            risk_values = pd.to_numeric(self.processed_rejected[risk_col], errors='coerce')
            risk_values = risk_values.dropna()
            
            if len(risk_values) > 0:
                risk_stats = {
                    'mean': float(risk_values.mean()),
                    'median': float(risk_values.median()),
                    'min': float(risk_values.min()),
                    'max': float(risk_values.max())
                }
                analysis['risk_score_stats'] = risk_stats
                
                print(f"\n⚠️ Risk Score Analysis:")
                print(f"   Average: {risk_stats['mean']:.2f}")
                print(f"   Median: {risk_stats['median']:.2f}")
                print(f"   Range: {risk_stats['min']:.2f} - {risk_stats['max']:.2f}")
        
        # State distribution
        state_cols = [col for col in self.processed_rejected.columns 
                     if 'state' in col.lower()]
        
        if state_cols:
            state_col = state_cols[0]
            state_dist = self.processed_rejected[state_col].value_counts().head(10).to_dict()
            analysis['top_rejection_states'] = state_dist
            
            print(f"\n🗺️ Top 10 States by Rejections:")
            for state, count in state_dist.items():
                percentage = (count / len(self.processed_rejected)) * 100
                print(f"   {state}: {count:,} ({percentage:.1f}%)")
        
        return analysis
    
    def create_risk_model(self):
        """
        Create a risk prediction model using accepted loans data
        """
        print("\n🤖 Phase 3: Building Risk Prediction Model")
        print("=" * 50)
        
        if self.processed_accepted is None:
            print("❌ No processed accepted data available for modeling.")
            return {}
        
        # Check if we have loan status for binary classification
        if 'loan_status' not in self.processed_accepted.columns:
            print("⚠️ No loan status column found. Creating synthetic risk labels...")
            # Create synthetic risk based on other factors
            return self.create_synthetic_risk_model()
        
        # Prepare target variable
        loan_status = self.processed_accepted['loan_status'].copy()
        
        # Define good vs bad loans
        good_statuses = ['Fully Paid', 'Current']
        bad_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)']
        
        # Filter to loans with clear outcomes
        clear_outcomes = self.processed_accepted[
            self.processed_accepted['loan_status'].isin(good_statuses + bad_statuses)
        ].copy()
        
        if len(clear_outcomes) == 0:
            print("⚠️ No clear loan outcomes found. Creating alternative model...")
            return self.create_alternative_risk_model()
        
        # Create binary target (0 = good, 1 = bad/risky)
        clear_outcomes['risk_label'] = clear_outcomes['loan_status'].apply(
            lambda x: 0 if x in good_statuses else 1
        )
        
        print(f"📊 Model training data: {len(clear_outcomes):,} loans with clear outcomes")
        
        risk_distribution = clear_outcomes['risk_label'].value_counts()
        print(f"   Good loans (0): {risk_distribution[0]:,} ({risk_distribution[0]/len(clear_outcomes)*100:.1f}%)")
        print(f"   Risky loans (1): {risk_distribution[1]:,} ({risk_distribution[1]/len(clear_outcomes)*100:.1f}%)")
        
        # Prepare features
        feature_columns = clear_outcomes.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in ['risk_label']]
        
        if len(feature_columns) == 0:
            print("❌ No numeric features available for modeling.")
            return {}
        
        X = clear_outcomes[feature_columns].fillna(clear_outcomes[feature_columns].median())
        y = clear_outcomes['risk_label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"📈 Training set: {len(X_train):,} loans")
        print(f"📊 Test set: {len(X_test):,} loans")
        print(f"🔢 Features used: {len(feature_columns)}")
        
        # Optimize models for large datasets
        if len(X_train) > 100000:
            print("📊 Large dataset detected - optimizing model parameters...")
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, solver='liblinear'),
                'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=15, 
                                                       min_samples_split=20, n_jobs=-1)
            }
        else:
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
            }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n🔄 Training {model_name}...")
            
            # Train model
            if model_name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            results[model_name] = {
                'accuracy': float(accuracy),
                'confusion_matrix': cm.tolist(),
                'feature_count': len(feature_columns),
                'training_samples': len(X_train)
            }
            
            print(f"✅ {model_name} Accuracy: {accuracy:.4f}")
            print(f"   True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
            print(f"   False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
        
        return results
    
    def create_alternative_risk_model(self):
        """
        Create alternative risk analysis when loan status is not available
        """
        print("🔄 Creating alternative risk analysis...")
        
        # Use grade and interest rate as proxy for risk
        risk_analysis = {}
        
        if 'grade' in self.processed_accepted.columns and 'int_rate' in self.processed_accepted.columns:
            # Convert interest rate to numeric
            int_rate = self.processed_accepted['int_rate']
            if int_rate.dtype == 'object':
                int_rate = pd.to_numeric(int_rate.str.replace('%', ''), errors='coerce')
            
            grade_risk = self.processed_accepted.groupby('grade').agg({
                'loan_amnt': 'mean',
                'int_rate': 'mean' if int_rate.dtype == 'object' else lambda x: x.mean()
            }).round(2)
            
            risk_analysis['grade_risk_profile'] = grade_risk.to_dict()
            
            print("📊 Risk Profile by Grade:")
            for grade in sorted(grade_risk.index):
                avg_amount = grade_risk.loc[grade, 'loan_amnt']
                avg_rate = grade_risk.loc[grade, 'int_rate']
                print(f"   Grade {grade}: Avg Amount ${avg_amount:,.2f}, Avg Rate {avg_rate:.2f}%")
        
        return risk_analysis
    
    def generate_comprehensive_insights(self, accepted_analysis, rejected_analysis, model_results):
        """
        Generate comprehensive business insights and recommendations
        """
        print("\n💡 Phase 4: Business Insights and Risk Management Strategy")
        print("=" * 60)
        
        insights = {
            'executive_summary': {},
            'risk_assessment': {},
            'business_recommendations': [],
            'key_findings': []
        }
        
        # Executive Summary
        total_applications = accepted_analysis.get('total_accepted', 0) + rejected_analysis.get('total_rejected', 0)
        rejection_rate = rejected_analysis.get('total_rejected', 0) / total_applications if total_applications > 0 else 0
        
        insights['executive_summary'] = {
            'total_applications_analyzed': total_applications,
            'total_accepted': accepted_analysis.get('total_accepted', 0),
            'total_rejected': rejected_analysis.get('total_rejected', 0),
            'overall_rejection_rate': rejection_rate
        }
        
        print("📊 EXECUTIVE SUMMARY:")
        print(f"   Total Applications Analyzed: {total_applications:,}")
        print(f"   Accepted Loans: {accepted_analysis.get('total_accepted', 0):,}")
        print(f"   Rejected Applications: {rejected_analysis.get('total_rejected', 0):,}")
        print(f"   Overall Rejection Rate: {rejection_rate:.1%}")
        
        # Risk Assessment
        print(f"\n⚠️ RISK ASSESSMENT:")
        
        if 'loan_amount_stats' in accepted_analysis:
            avg_loan = accepted_analysis['loan_amount_stats']['mean']
            print(f"   Average Approved Loan Amount: ${avg_loan:,.2f}")
            
        if 'requested_amount_stats' in rejected_analysis:
            avg_rejected = rejected_analysis['requested_amount_stats']['mean']
            print(f"   Average Rejected Request Amount: ${avg_rejected:,.2f}")
            
            if 'loan_amount_stats' in accepted_analysis:
                amount_difference = avg_rejected - avg_loan
                print(f"   Amount Gap (Rejected vs Accepted): ${amount_difference:,.2f}")
                insights['key_findings'].append(
                    f"Rejected applications request ${amount_difference:,.2f} more on average"
                )
        
        if 'interest_rate_stats' in accepted_analysis:
            avg_rate = accepted_analysis['interest_rate_stats']['mean']
            print(f"   Average Interest Rate (Accepted): {avg_rate:.2f}%")
        
        # Model Performance
        if model_results:
            best_model = max(model_results.keys(), key=lambda x: model_results[x].get('accuracy', 0))
            best_accuracy = model_results[best_model]['accuracy']
            
            print(f"\n🤖 PREDICTIVE MODEL PERFORMANCE:")
            print(f"   Best Model: {best_model}")
            print(f"   Accuracy: {best_accuracy:.1%}")
            print(f"   Can predict risk with {best_accuracy:.1%} accuracy")
            
            insights['risk_assessment']['model_accuracy'] = best_accuracy
            insights['key_findings'].append(f"Risk prediction model achieves {best_accuracy:.1%} accuracy")
        
        # Business Recommendations
        recommendations = [
            "🎯 STRATEGIC RECOMMENDATIONS:",
            "",
            "1. AUTOMATED RISK SCREENING:",
            "   - Implement ML model for initial application screening",
            "   - Set risk thresholds based on business risk tolerance",
            "   - Route high-risk applications for manual review",
            "",
            "2. PORTFOLIO OPTIMIZATION:",
            "   - Monitor loan grade distribution for risk balance",
            "   - Adjust pricing based on risk assessment",
            "   - Track performance by loan characteristics",
            "",
            "3. OPERATIONAL IMPROVEMENTS:",
            "   - Standardize rejection criteria and documentation",
            "   - Implement real-time risk monitoring dashboard",
            "   - Regular model retraining with new loan performance data",
            "",
            "4. REGULATORY COMPLIANCE:",
            "   - Ensure fair lending practices across all demographics",
            "   - Document risk assessment methodology",
            "   - Regular audit of decision-making processes",
            "",
            "5. BUSINESS GROWTH:",
            f"   - Current rejection rate of {rejection_rate:.1%} indicates room for growth",
            "   - Consider expanding to lower-risk segments",
            "   - Develop products for different risk profiles"
        ]
        
        insights['business_recommendations'] = recommendations
        
        print("\n" + "\n".join(recommendations))
        
        # Save comprehensive results
        self.analysis_results = insights
        
        # Create results directory
        if not os.path.exists("financial_analysis_results"):
            os.makedirs("financial_analysis_results")
        
        # Save all analysis results
        all_results = {
            'executive_summary': insights['executive_summary'],
            'accepted_loans_analysis': accepted_analysis,
            'rejected_loans_analysis': rejected_analysis,
            'model_results': model_results,
            'business_insights': insights,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open('financial_analysis_results/comprehensive_analysis.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Create executive report
        self.create_executive_report(all_results)
        
        return insights
    
    def create_executive_report(self, results):
        """
        Create an executive summary report
        """
        report_content = f"""
FINANCIAL RISK MANAGEMENT ANALYSIS
Executive Summary Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
========
Total Applications Analyzed: {results['executive_summary']['total_applications_analyzed']:,}
Accepted Loans: {results['executive_summary']['total_accepted']:,}
Rejected Applications: {results['executive_summary']['total_rejected']:,}
Overall Rejection Rate: {results['executive_summary']['overall_rejection_rate']:.1%}

ACCEPTED LOANS ANALYSIS
======================
"""
        
        if 'loan_amount_stats' in results['accepted_loans_analysis']:
            stats = results['accepted_loans_analysis']['loan_amount_stats']
            report_content += f"""
Loan Amounts:
- Average: ${stats['mean']:,.2f}
- Median: ${stats['median']:,.2f}
- Range: ${stats['min']:,.2f} - ${stats['max']:,.2f}
"""
        
        if 'interest_rate_stats' in results['accepted_loans_analysis']:
            rates = results['accepted_loans_analysis']['interest_rate_stats']
            report_content += f"""
Interest Rates:
- Average: {rates['mean']:.2f}%
- Median: {rates['median']:.2f}%
- Range: {rates['min']:.2f}% - {rates['max']:.2f}%
"""
        
        report_content += f"""

REJECTED LOANS ANALYSIS
=======================
"""
        
        if 'requested_amount_stats' in results['rejected_loans_analysis']:
            stats = results['rejected_loans_analysis']['requested_amount_stats']
            report_content += f"""
Requested Amounts:
- Average: ${stats['mean']:,.2f}
- Median: ${stats['median']:,.2f}
- Range: ${stats['min']:,.2f} - ${stats['max']:,.2f}
"""
        
        if results.get('model_results'):
            best_model = max(results['model_results'].keys(), 
                           key=lambda x: results['model_results'][x].get('accuracy', 0))
            accuracy = results['model_results'][best_model]['accuracy']
            
            report_content += f"""

PREDICTIVE MODEL PERFORMANCE
============================
Best Model: {best_model}
Accuracy: {accuracy:.1%}
Business Impact: Can automate {accuracy:.1%} of risk assessment decisions
"""
        
        report_content += f"""

KEY RECOMMENDATIONS
==================
1. Implement automated risk screening using machine learning
2. Monitor and optimize loan portfolio balance
3. Establish real-time risk monitoring systems
4. Ensure regulatory compliance and fair lending practices
5. Explore opportunities for responsible business growth

NEXT STEPS
==========
1. Review detailed analysis in comprehensive_analysis.json
2. Implement recommended risk management strategies
3. Set up regular model monitoring and retraining
4. Establish KPI tracking for risk management effectiveness
"""
        
        with open('financial_analysis_results/executive_report.txt', 'w') as f:
            f.write(report_content)
        
        print(f"\n📄 Executive report saved to 'financial_analysis_results/executive_report.txt'")
    
    def run_complete_analysis(self, sample_size=None):
        """
        Run the complete financial risk management analysis pipeline
        """
        print("🚀 FINANCIAL RISK MANAGEMENT SYSTEM")
        print("=" * 60)
        print("Starting comprehensive risk analysis pipeline...")
        print("📊 Processing COMPLETE dataset (all records)")
        print()
        
        try:
            # Phase 1: Data Loading and Cleaning
            if not self.load_data(sample_size):
                return
            
            # Data Cleaning
            self.clean_accepted_data()
            self.clean_rejected_data()
            
            # Phase 2: Analysis
            accepted_analysis = self.analyze_accepted_loans()
            rejected_analysis = self.analyze_rejected_loans()
            
            # Phase 3: Modeling
            model_results = self.create_risk_model()
            
            # Phase 4: Insights
            insights = self.generate_comprehensive_insights(
                accepted_analysis, rejected_analysis, model_results
            )
            
            print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print("📁 All results saved in 'financial_analysis_results' directory")
            print("📊 Files generated:")
            print("   - comprehensive_analysis.json: Complete analysis results")
            print("   - executive_report.txt: Executive summary")
            print("\n🎯 NEXT STEPS:")
            print("   1. Review the executive report for key insights")
            print("   2. Implement recommended risk management strategies")
            print("   3. Set up ongoing monitoring and model maintenance")
            print("   4. Plan regular analysis updates with new data")
            
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
    
    # Run complete analysis with FULL dataset (no sampling)
    print("🔧 Processing COMPLETE dataset - all available records")
    print("⚠️  This may take several minutes depending on dataset size")
    print("� For faster testing, add sample_size parameter (e.g., sample_size=50000)")
    print()
    
    risk_manager.run_complete_analysis()  # No sample_size = full dataset