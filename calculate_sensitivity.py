import pandas as pd
import numpy as np
from itertools import product
from tennis_video_analysis_pnl import generate_video_analysis_pnl
import os

def calculate_sensitivity_results():
    """
    Calculate sensitivity analysis results and save to CSV
    """
    # Define sensitivity parameters
    sensitivity_params = {
        "Development Months": {
            "param": "development_months",
            "base": 6,
            "variations": list(range(3, 13)),  # [3,4,5,6,7,8,9,10,11,12]
            "description": "Time needed for initial development"
        },
        "Monthly Growth Rate": {
            "param": "monthly_growth_rate",
            "base": 0.05,
            "variations": [0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15],
            "description": "Monthly customer growth rate"
        },
        "Monthly Churn Rate": {
            "param": "churn_rate",
            "base": 0.03,
            "variations": [0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.10],
            "description": "Monthly customer churn rate"
        },
        "Monthly Team Cost": {
            "param": ["founder_salary", "monthly_contractor_budget"],
            "base": 25000,
            "variations": [15000, 20000, 25000, 30000, 35000, 40000],
            "description": "Combined monthly cost for team",
            "split_ratio": [0.7, 0.3]
        },
        "Starting Coaches": {
            "param": "starting_coaches",
            "base": 50,
            "variations": [20, 50, 100, 150, 200, 300, 400, 500],
            "description": "Initial number of coaches at launch"
        },
        "Basic Plan Price": {
            "param": "basic_plan_price",
            "base": 150,
            "variations": [150, 175, 200, 225, 250, 275, 300],
            "description": "Monthly subscription price"
        },
        "Pay-per-Video Price": {
            "param": "pay_per_video_price",
            "base": 50,
            "variations": [50, 65, 80, 95, 110, 120, 130],
            "description": "Price per individual video analysis"
        }
    }

    # Create parameter ranges for grid search
    param_ranges = {}
    for param_name, param_info in sensitivity_params.items():
        variations = param_info["variations"]
        param_ranges[param_name] = variations  # Use the actual variations instead of linspace

    # Generate all combinations of parameter values
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]
    combinations = list(product(*param_values))
    
    # Store all results
    all_results = []
    
    # Test each combination
    total_combinations = len(combinations)
    print(f"Processing {total_combinations} combinations...")
    
    for i, combo in enumerate(combinations):
        # Create parameter dictionary for this combination
        params = {}
        for name, value in zip(param_names, combo):
            param_info = sensitivity_params[name]
            
            if isinstance(param_info["param"], list):
                # Handle Monthly Team Cost special case
                for param_name, ratio in zip(param_info["param"], param_info["split_ratio"]):
                    params[param_name] = value * ratio
            elif "development_months" in param_info["param"].lower():
                params[param_info["param"]] = int(value)
            elif "coaches" in param_info["param"].lower():
                params[param_info["param"]] = int(value)
            else:
                params[param_info["param"]] = value
        
        # Generate P&L with these parameters
        test_df = generate_video_analysis_pnl(**params)
        
        # Calculate metrics for each parameter
        for param_idx, param_name in enumerate(param_names):
            result_dict = {
                'Parameter': param_name,
                'Parameter Value': combo[param_idx],
                'Total Investment': test_df['Investment Required'].max(),
                'Months to Breakeven': next((i + 1 for i, row in enumerate(test_df['Cumulative Profit/Loss']) if row > 0), float('inf')),
                'Peak Monthly Revenue': test_df['Total Revenue'].max(),
                'Year 1 Revenue': test_df['Total Revenue'].iloc[12:24].sum() if len(test_df) > 24 else None,
                'Final Monthly Revenue': test_df['Total Revenue'].iloc[-1],
                'Final Monthly Costs': test_df['Operating Costs'].iloc[-1] + test_df['Personnel Costs'].iloc[-1],
                'Max Monthly Profit': test_df['Net Income'].max()
            }
            all_results.append(result_dict)
        
        if i % 10 == 0:
            print(f"Processed {i+1} of {total_combinations} combinations")
    
    # Convert to DataFrame and save
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv("sensitivity_results.csv", index=False)
    print("Sensitivity analysis complete. Results saved to sensitivity_results.csv")

if __name__ == "__main__":
    calculate_sensitivity_results() 