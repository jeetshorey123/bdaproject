"""
Financial Risk Management - Data Visualization Module
====================================================

This module creates comprehensive visualizations for the financial risk analysis
using matplotlib and other plotting libraries that work with the current environment.

Author: Financial Risk Management Team
Date: October 31, 2025
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class RiskVisualizationCreator:
    """
    Creates comprehensive visualizations for financial risk analysis
    """
    
    def __init__(self, results_file_path="financial_analysis_results/comprehensive_analysis.json"):
        """
        Initialize with analysis results
        """
        self.results_file = results_file_path
        self.results = None
        self.load_results()
        
    def load_results(self):
        """
        Load analysis results from JSON file
        """
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            print("✅ Loaded analysis results successfully")
        except FileNotFoundError:
            print("❌ Results file not found. Please run the main analysis first.")
        except Exception as e:
            print(f"❌ Error loading results: {str(e)}")
    
    def create_loan_amount_comparison(self):
        """
        Create visualization comparing accepted vs rejected loan amounts
        """
        if not self.results:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accepted loan amounts
        accepted_stats = self.results['accepted_loans_analysis'].get('loan_amount_stats', {})
        if accepted_stats:
            amounts = ['Min', 'Mean', 'Median', 'Max']
            values = [accepted_stats['min'], accepted_stats['mean'], 
                     accepted_stats['median'], accepted_stats['max']]
            
            bars1 = ax1.bar(amounts, values, color=['lightblue', 'blue', 'darkblue', 'navy'])
            ax1.set_title('Accepted Loan Amounts Distribution')
            ax1.set_ylabel('Amount ($)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars1, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'${value:,.0f}', ha='center', va='bottom')
        
        # Rejected loan amounts
        rejected_stats = self.results['rejected_loans_analysis'].get('requested_amount_stats', {})
        if rejected_stats:
            amounts = ['Min', 'Mean', 'Median', 'Max']
            values = [rejected_stats['min'], rejected_stats['mean'], 
                     rejected_stats['median'], rejected_stats['max']]
            
            bars2 = ax2.bar(amounts, values, color=['lightcoral', 'red', 'darkred', 'maroon'])
            ax2.set_title('Rejected Loan Request Amounts Distribution')
            ax2.set_ylabel('Amount ($)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars2, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'${value:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('financial_analysis_results/loan_amount_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created loan amount comparison visualization")
    
    def create_risk_distribution_chart(self):
        """
        Create risk distribution visualization
        """
        if not self.results:
            return
            
        # Get grade distribution from accepted loans
        grade_dist = self.results['accepted_loans_analysis'].get('grade_distribution', {})
        
        if not grade_dist:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Grade distribution bar chart
        grades = sorted(grade_dist.keys())
        counts = [grade_dist[grade] for grade in grades]
        total = sum(counts)
        percentages = [(count/total)*100 for count in counts]
        
        bars = ax1.bar(grades, counts, color=['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred', 'maroon'][:len(grades)])
        ax1.set_title('Loan Distribution by Risk Grade')
        ax1.set_xlabel('Loan Grade')
        ax1.set_ylabel('Number of Loans')
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        # Grade distribution pie chart
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred', 'maroon'][:len(grades)]
        wedges, texts, autotexts = ax2.pie(counts, labels=grades, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('Risk Grade Distribution')
        
        plt.tight_layout()
        plt.savefig('financial_analysis_results/risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created risk distribution visualization")
    
    def create_model_performance_chart(self):
        """
        Create model performance visualization
        """
        if not self.results or 'model_results' not in self.results:
            return
            
        model_results = self.results['model_results']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Model accuracy comparison
        models = list(model_results.keys())
        accuracies = [model_results[model]['accuracy'] for model in models]
        
        bars = ax1.bar(models, accuracies, color=['lightblue', 'lightgreen'])
        ax1.set_title('Model Performance Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add accuracy labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Confusion matrix for best model
        best_model = max(models, key=lambda x: model_results[x]['accuracy'])
        cm = model_results[best_model]['confusion_matrix']
        
        im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
        ax2.set_title(f'Confusion Matrix - {best_model}')
        
        # Add text annotations
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                ax2.text(j, i, str(cm[i][j]), ha='center', va='center', 
                        color='white' if cm[i][j] > max(cm[0][0], cm[1][1])/2 else 'black')
        
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Good', 'Risky'])
        ax2.set_yticklabels(['Good', 'Risky'])
        
        plt.tight_layout()
        plt.savefig('financial_analysis_results/model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created model performance visualization")
    
    def create_geographic_distribution(self):
        """
        Create geographic distribution of rejections
        """
        if not self.results:
            return
            
        rejection_states = self.results['rejected_loans_analysis'].get('top_rejection_states', {})
        
        if not rejection_states:
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        states = list(rejection_states.keys())
        counts = list(rejection_states.values())
        
        # Create horizontal bar chart
        bars = ax.barh(states, counts, color='lightcoral')
        ax.set_title('Top 10 States by Loan Rejections')
        ax.set_xlabel('Number of Rejections')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('financial_analysis_results/geographic_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created geographic distribution visualization")
    
    def create_executive_dashboard(self):
        """
        Create a comprehensive executive dashboard
        """
        if not self.results:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Financial Risk Management - Executive Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Application Overview
        exec_summary = self.results['executive_summary']
        categories = ['Accepted', 'Rejected']
        values = [exec_summary['total_accepted'], exec_summary['total_rejected']]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax1.pie(values, labels=categories, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Loan Application Outcomes')
        
        # 2. Risk Grade Distribution (if available)
        grade_dist = self.results['accepted_loans_analysis'].get('grade_distribution', {})
        if grade_dist:
            grades = sorted(grade_dist.keys())
            counts = [grade_dist[grade] for grade in grades]
            
            bars = ax2.bar(grades, counts, color=['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred', 'maroon'][:len(grades)])
            ax2.set_title('Accepted Loans by Risk Grade')
            ax2.set_xlabel('Grade')
            ax2.set_ylabel('Count')
        
        # 3. Model Performance
        if 'model_results' in self.results:
            model_results = self.results['model_results']
            models = list(model_results.keys())
            accuracies = [model_results[model]['accuracy'] for model in models]
            
            bars = ax3.bar(models, accuracies, color=['lightblue', 'lightgreen'])
            ax3.set_title('ML Model Performance')
            ax3.set_ylabel('Accuracy')
            ax3.set_ylim(0, 1)
            
            for bar, acc in zip(bars, accuracies):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.1%}', ha='center', va='bottom')
        
        # 4. Key Metrics Summary
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""
KEY METRICS SUMMARY

Total Applications: {exec_summary['total_applications_analyzed']:,}
Rejection Rate: {exec_summary['overall_rejection_rate']:.1%}

ACCEPTED LOANS:
"""
        
        if 'loan_amount_stats' in self.results['accepted_loans_analysis']:
            loan_stats = self.results['accepted_loans_analysis']['loan_amount_stats']
            summary_text += f"Average Amount: ${loan_stats['mean']:,.0f}\n"
        
        if 'interest_rate_stats' in self.results['accepted_loans_analysis']:
            rate_stats = self.results['accepted_loans_analysis']['interest_rate_stats']
            summary_text += f"Average Rate: {rate_stats['mean']:.1f}%\n"
        
        summary_text += "\nREJECTED APPLICATIONS:\n"
        
        if 'requested_amount_stats' in self.results['rejected_loans_analysis']:
            req_stats = self.results['rejected_loans_analysis']['requested_amount_stats']
            summary_text += f"Average Request: ${req_stats['mean']:,.0f}\n"
        
        if 'model_results' in self.results:
            best_model = max(self.results['model_results'].keys(), 
                           key=lambda x: self.results['model_results'][x]['accuracy'])
            best_accuracy = self.results['model_results'][best_model]['accuracy']
            summary_text += f"\nBest Model: {best_model}\nAccuracy: {best_accuracy:.1%}"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('financial_analysis_results/executive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created executive dashboard")
    
    def create_all_visualizations(self):
        """
        Create all visualizations
        """
        print("\n📊 Creating Comprehensive Visualizations")
        print("=" * 50)
        
        if not self.results:
            print("❌ No results available for visualization")
            return
        
        self.create_loan_amount_comparison()
        self.create_risk_distribution_chart()
        self.create_model_performance_chart()
        self.create_geographic_distribution()
        self.create_executive_dashboard()
        
        print("\n✅ All visualizations created successfully!")
        print("📁 Visualization files saved in 'financial_analysis_results' directory:")
        print("   - loan_amount_comparison.png")
        print("   - risk_distribution.png")
        print("   - model_performance.png")
        print("   - geographic_distribution.png")
        print("   - executive_dashboard.png")

# Main execution
if __name__ == "__main__":
    print("🎨 FINANCIAL RISK MANAGEMENT VISUALIZATIONS")
    print("=" * 60)
    
    # Create visualizations
    viz_creator = RiskVisualizationCreator()
    viz_creator.create_all_visualizations()
    
    print("\n🎯 Visualizations complete! Use these charts for:")
    print("   - Executive presentations")
    print("   - Risk assessment reports") 
    print("   - Stakeholder communications")
    print("   - Regulatory documentation")