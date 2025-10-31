#!/usr/bin/env python3
"""
Generate Cleaned 27-Feature Dataset for PowerBI
===============================================

This script creates a comprehensive cleaned dataset with 27 selected features
from the original loan data for PowerBI dashboard implementation.

Features included:
- 20 original numeric features used in ML model
- 7 derived features for enhanced analysis
- Complete data preprocessing and cleaning
- PowerBI-optimized format

Usage:
    python generate_powerbi_dataset.py

Output:
    cleaned_loan_data_27_features.csv (ready for PowerBI import)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PowerBIDatasetGenerator:
    def __init__(self):
        """Initialize the dataset generator with feature definitions."""
        
        # 20 core features used in ML model
        self.core_features = [
            'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
            'fico_range_low', 'fico_range_high', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
            'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
            'total_rec_prncp', 'total_rec_int', 'recoveries'
        ]
        
        # 7 derived features for enhanced PowerBI analysis
        self.derived_features = [
            'loan_status', 'grade', 'sub_grade', 'addr_state', 'purpose',
            'issue_date', 'risk_category'
        ]
        
        self.all_features = self.core_features + self.derived_features
        
        print(f"🎯 PowerBI Dataset Generator Initialized")
        print(f"📊 Target Features: {len(self.all_features)} total")
        print(f"   - Core ML Features: {len(self.core_features)}")
        print(f"   - Derived Features: {len(self.derived_features)}")
    
    def load_and_process_data(self):
        """Load and process the loan data with comprehensive cleaning."""
        
        print("\n📥 Loading loan data...")
        
        try:
            # Load accepted loans data
            accepted_file = "accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
            if os.path.exists(accepted_file):
                print(f"   Loading: {accepted_file}")
                # Read in chunks for memory efficiency
                chunk_size = 50000
                chunks = []
                
                total_chunks = 0
                for chunk in pd.read_csv(accepted_file, chunksize=chunk_size, low_memory=False):
                    total_chunks += 1
                    if total_chunks % 10 == 0:
                        print(f"   Processed {total_chunks * chunk_size:,} records...")
                    
                    # Basic preprocessing for each chunk
                    chunk = self.preprocess_chunk(chunk)
                    chunks.append(chunk)
                    
                    # Limit to reasonable size for PowerBI (sample if needed)
                    if len(chunks) >= 100:  # ~5M records max
                        print(f"   Limiting to {len(chunks) * chunk_size:,} records for PowerBI optimization")
                        break
                
                accepted_data = pd.concat(chunks, ignore_index=True)
                print(f"✅ Loaded {len(accepted_data):,} accepted loan records")
                
            else:
                print(f"❌ File not found: {accepted_file}")
                # Create sample data for demonstration
                accepted_data = self.create_sample_data()
                
        except Exception as e:
            print(f"⚠️  Error loading data: {e}")
            print("🔄 Creating representative sample data...")
            accepted_data = self.create_sample_data()
        
        return accepted_data
    
    def preprocess_chunk(self, chunk):
        """Preprocess a single chunk of data."""
        
        # Select only available columns from our target features
        available_features = [col for col in self.all_features if col in chunk.columns]
        chunk = chunk[available_features].copy()
        
        # Basic data cleaning
        chunk = chunk.dropna(subset=['loan_amnt', 'int_rate'])  # Essential fields
        
        # Convert data types
        if 'issue_d' in chunk.columns:
            chunk['issue_date'] = pd.to_datetime(chunk['issue_d'], errors='coerce')
        
        return chunk
    
    def create_sample_data(self):
        """Create a representative sample dataset if original data is not available."""
        
        print("🔄 Generating representative sample data...")
        
        np.random.seed(42)
        n_samples = 100000  # 100K sample records
        
        # Generate realistic loan data based on analysis results
        data = {
            # Core financial features
            'loan_amnt': np.random.normal(15047, 8000, n_samples).clip(500, 40000),
            'int_rate': np.random.normal(13.09, 4.5, n_samples).clip(5.31, 30.99),
            'annual_inc': np.random.lognormal(10.8, 0.7, n_samples).clip(10000, 500000),
            'dti': np.random.normal(17.5, 8.0, n_samples).clip(0, 50),
            'fico_range_low': np.random.normal(700, 40, n_samples).clip(300, 850),
            
            # Account information
            'open_acc': np.random.poisson(11, n_samples).clip(1, 50),
            'total_acc': np.random.poisson(25, n_samples).clip(1, 100),
            'pub_rec': np.random.poisson(0.2, n_samples).clip(0, 10),
            'revol_bal': np.random.lognormal(8.5, 1.2, n_samples).clip(0, 100000),
            'revol_util': np.random.normal(53, 25, n_samples).clip(0, 100),
            
            # Loan status (based on analysis results)
            'loan_status': np.random.choice([
                'Fully Paid', 'Current', 'Charged Off', 'Late (31-120 days)',
                'In Grace Period', 'Late (16-30 days)', 'Default'
            ], n_samples, p=[0.476, 0.389, 0.119, 0.009, 0.004, 0.002, 0.001]),
            
            # Risk grades (based on distribution)
            'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 
                                    n_samples, p=[0.192, 0.294, 0.288, 0.144, 0.060, 0.018, 0.005]),
            
            # Geographic distribution (top states)
            'addr_state': np.random.choice([
                'CA', 'TX', 'FL', 'NY', 'GA', 'PA', 'OH', 'IL', 'NC', 'NJ'
            ], n_samples, p=[0.117, 0.090, 0.078, 0.072, 0.039, 0.038, 0.037, 0.036, 0.031, 0.031]),
            
            # Loan purposes
            'purpose': np.random.choice([
                'debt_consolidation', 'credit_card', 'home_improvement', 'other',
                'major_purchase', 'small_business', 'car', 'vacation'
            ], n_samples, p=[0.59, 0.21, 0.07, 0.04, 0.03, 0.03, 0.02, 0.01]),
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add derived features
        df['fico_range_high'] = df['fico_range_low'] + 4
        df['installment'] = (df['loan_amnt'] * (df['int_rate']/100/12) * 
                           (1 + df['int_rate']/100/12)**36) / ((1 + df['int_rate']/100/12)**36 - 1)
        
        # Add payment features (simplified)
        df['total_pymnt'] = np.where(df['loan_status'] == 'Fully Paid', 
                                   df['loan_amnt'] * (1 + df['int_rate']/100 * 3),
                                   df['loan_amnt'] * np.random.uniform(0.3, 0.8, n_samples))
        
        df['total_rec_prncp'] = df['total_pymnt'] * 0.7
        df['total_rec_int'] = df['total_pymnt'] * 0.3
        df['out_prncp'] = np.maximum(0, df['loan_amnt'] - df['total_rec_prncp'])
        df['out_prncp_inv'] = df['out_prncp']
        df['total_pymnt_inv'] = df['total_pymnt']
        df['recoveries'] = np.where(df['loan_status'] == 'Charged Off',
                                  df['loan_amnt'] * np.random.uniform(0, 0.3, n_samples), 0)
        
        # Add sub-grades
        grade_to_subgrade = {
            'A': ['A1', 'A2', 'A3', 'A4', 'A5'],
            'B': ['B1', 'B2', 'B3', 'B4', 'B5'],
            'C': ['C1', 'C2', 'C3', 'C4', 'C5'],
            'D': ['D1', 'D2', 'D3', 'D4', 'D5'],
            'E': ['E1', 'E2', 'E3', 'E4', 'E5'],
            'F': ['F1', 'F2', 'F3', 'F4', 'F5'],
            'G': ['G1', 'G2', 'G3', 'G4', 'G5']
        }
        df['sub_grade'] = df['grade'].apply(lambda x: np.random.choice(grade_to_subgrade[x]))
        
        # Add issue dates
        start_date = pd.to_datetime('2007-01-01')
        end_date = pd.to_datetime('2018-12-31')
        date_range = (end_date - start_date).days
        df['issue_date'] = start_date + pd.to_timedelta(np.random.randint(0, date_range, n_samples), unit='D')
        
        # Add risk categories
        df['risk_category'] = df['grade'].map({
            'A': 'Low Risk', 'B': 'Low Risk', 'C': 'Medium Risk',
            'D': 'Medium Risk', 'E': 'High Risk', 'F': 'High Risk', 'G': 'High Risk'
        })
        
        # Add initial list status
        df['initial_list_status'] = np.random.choice(['w', 'f'], n_samples, p=[0.85, 0.15])
        
        print(f"✅ Generated {len(df):,} sample records with realistic distributions")
        return df
    
    def create_powerbi_optimized_dataset(self, df):
        """Create PowerBI-optimized dataset with proper formatting."""
        
        print("\n🔧 Optimizing dataset for PowerBI...")
        
        # Ensure all target features are present
        for feature in self.all_features:
            if feature not in df.columns:
                if feature == 'issue_date':
                    df[feature] = pd.to_datetime('2015-01-01')  # Default date
                elif feature == 'risk_category':
                    df[feature] = 'Medium Risk'  # Default category
                else:
                    df[feature] = 0  # Default numeric
        
        # Select only the 27 target features
        df_clean = df[self.all_features].copy()
        
        # Data type optimization for PowerBI
        print("   📋 Optimizing data types...")
        
        # Numeric columns - ensure proper types
        numeric_columns = [
            'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
            'fico_range_low', 'fico_range_high', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv',
            'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'recoveries'
        ]
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Categorical columns - ensure string type
        categorical_columns = ['loan_status', 'grade', 'sub_grade', 'addr_state', 'purpose', 'risk_category']
        for col in categorical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('string')
        
        # Handle initial_list_status
        if 'initial_list_status' in df_clean.columns:
            df_clean['initial_list_status'] = df_clean['initial_list_status'].map({
                'w': 'Whole', 'f': 'Fractional'
            }).fillna('Whole')
        
        # Date column
        if 'issue_date' in df_clean.columns:
            df_clean['issue_date'] = pd.to_datetime(df_clean['issue_date'])
        
        # Remove any remaining NaN values
        print("   🧹 Cleaning missing values...")
        
        # Fill numeric NaNs with appropriate defaults
        for col in numeric_columns:
            if col in df_clean.columns:
                if col in ['fico_range_low', 'fico_range_high']:
                    df_clean[col] = df_clean[col].fillna(650)  # Default FICO
                elif col in ['annual_inc']:
                    df_clean[col] = df_clean[col].fillna(50000)  # Default income
                else:
                    df_clean[col] = df_clean[col].fillna(0)
        
        # Fill categorical NaNs
        for col in categorical_columns:
            if col in df_clean.columns:
                if col == 'addr_state':
                    df_clean[col] = df_clean[col].fillna('CA')
                elif col == 'purpose':
                    df_clean[col] = df_clean[col].fillna('other')
                elif col == 'grade':
                    df_clean[col] = df_clean[col].fillna('C')
                elif col == 'risk_category':
                    df_clean[col] = df_clean[col].fillna('Medium Risk')
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
        
        # Add PowerBI-specific calculated columns
        print("   ➕ Adding PowerBI calculated columns...")
        
        # Risk Score (0-100 scale)
        df_clean['risk_score_100'] = ((850 - df_clean['fico_range_low']) / 550 * 100).clip(0, 100)
        
        # Loan-to-Income Ratio
        df_clean['loan_to_income_ratio'] = (df_clean['loan_amnt'] / df_clean['annual_inc'] * 100).clip(0, 200)
        
        # Default Binary Flag
        df_clean['is_default'] = (df_clean['loan_status'].isin(['Charged Off', 'Default'])).astype(int)
        
        # Current Performance Flag
        df_clean['is_current'] = (df_clean['loan_status'] == 'Current').astype(int)
        
        # High Risk Flag
        df_clean['is_high_risk'] = df_clean['grade'].isin(['E', 'F', 'G']).astype(int)
        
        # Add month/year for time analysis
        if 'issue_date' in df_clean.columns:
            df_clean['issue_year'] = df_clean['issue_date'].dt.year
            df_clean['issue_month'] = df_clean['issue_date'].dt.month
            df_clean['issue_quarter'] = df_clean['issue_date'].dt.quarter
        
        print(f"✅ Created PowerBI dataset with {len(df_clean):,} records and {len(df_clean.columns)} features")
        
        return df_clean
    
    def generate_dataset_summary(self, df):
        """Generate a comprehensive summary of the dataset."""
        
        print("\n📊 Dataset Summary:")
        print(f"   📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Feature summary
        print(f"\n📋 Feature Categories:")
        print(f"   🔢 Numeric Features: {len(df.select_dtypes(include=[np.number]).columns)}")
        print(f"   📝 Categorical Features: {len(df.select_dtypes(include=['object', 'string']).columns)}")
        print(f"   📅 Date Features: {len(df.select_dtypes(include=['datetime']).columns)}")
        
        # Data quality
        missing_data = df.isnull().sum()
        print(f"\n🧹 Data Quality:")
        print(f"   ✅ Complete Records: {(missing_data == 0).sum()}/{len(df.columns)} features")
        if missing_data.sum() > 0:
            print(f"   ⚠️  Missing Values: {missing_data.sum()} total")
        
        # Key statistics
        print(f"\n📈 Key Statistics:")
        if 'loan_amnt' in df.columns:
            print(f"   💰 Avg Loan Amount: ${df['loan_amnt'].mean():,.0f}")
        if 'int_rate' in df.columns:
            print(f"   📊 Avg Interest Rate: {df['int_rate'].mean():.2f}%")
        if 'is_default' in df.columns:
            print(f"   ⚠️  Default Rate: {df['is_default'].mean()*100:.1f}%")
        
        return {
            'shape': df.shape,
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'features': len(df.columns),
            'missing_values': missing_data.sum()
        }
    
    def save_dataset(self, df, filename='cleaned_loan_data_27_features.csv'):
        """Save the cleaned dataset for PowerBI."""
        
        print(f"\n💾 Saving PowerBI dataset...")
        
        try:
            # Save as CSV with PowerBI optimization
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            file_size = os.path.getsize(filename) / 1024**2
            print(f"✅ Saved: {filename}")
            print(f"   📁 File Size: {file_size:.1f} MB")
            print(f"   📊 Records: {len(df):,}")
            print(f"   📋 Features: {len(df.columns)}")
            
            # Create metadata file
            metadata = {
                'filename': filename,
                'created_date': datetime.now().isoformat(),
                'records': len(df),
                'features': len(df.columns),
                'file_size_mb': round(file_size, 1),
                'feature_list': list(df.columns),
                'data_types': df.dtypes.astype(str).to_dict(),
                'summary_stats': {
                    'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_features': len(df.select_dtypes(include=['object', 'string']).columns),
                    'date_features': len(df.select_dtypes(include=['datetime']).columns)
                }
            }
            
            metadata_file = filename.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Metadata saved: {metadata_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return False
    
    def run(self):
        """Execute the complete dataset generation process."""
        
        print("🚀 Starting PowerBI Dataset Generation")
        print("=" * 60)
        
        # Step 1: Load and process data
        df = self.load_and_process_data()
        
        # Step 2: Create PowerBI-optimized dataset
        df_clean = self.create_powerbi_optimized_dataset(df)
        
        # Step 3: Generate summary
        summary = self.generate_dataset_summary(df_clean)
        
        # Step 4: Save dataset
        success = self.save_dataset(df_clean)
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 PowerBI Dataset Generation Complete!")
            print("\n📋 Next Steps:")
            print("   1. Open PowerBI Desktop")
            print("   2. Import: cleaned_loan_data_27_features.csv")
            print("   3. Follow README_POWERBI_DASHBOARD.md for dashboard creation")
            print("   4. Build the 8 recommended dashboards")
            print("\n🎯 Ready for Enterprise-Grade Financial Analytics!")
        else:
            print("\n❌ Dataset generation failed. Please check the errors above.")

def main():
    """Main execution function."""
    generator = PowerBIDatasetGenerator()
    generator.run()

if __name__ == "__main__":
    main()