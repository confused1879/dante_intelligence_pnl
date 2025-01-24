import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import product
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calculate_marketing_budget(monthly_revenue, base_marketing_budget):
    """
    Calculate marketing budget based on revenue milestones
    """
    if monthly_revenue >= 100000:  # >$100k MRR
        return max(
            base_marketing_budget,
            monthly_revenue * 0.15  # 15% of revenue for marketing at scale
        )
    elif monthly_revenue >= 50000:  # >$50k MRR
        return max(
            base_marketing_budget,
            monthly_revenue * 0.10  # 10% of revenue for marketing
        )
    else:
        return base_marketing_budget  # Base marketing budget during early stage

def generate_video_analysis_pnl(
    # Development Phase
    development_months=6,           # Months needed before launch
    development_contractor_hours_monthly=160,  # Hours of contractor help during development
    development_costs_monthly={
        'cloud_infrastructure': 500,    # AWS/GCP costs during development
        'data_collection': 1000,        # Cost to acquire training data
        'testing_equipment': 500,       # Testing devices, cameras, etc.
        'software_licenses': 200,       # Development tools, APIs, etc.
    },
    # Market & Customer Assumptions
    months=24,
    starting_coaches=50,  # Starting number of coach customers
    monthly_growth_rate=0.05,
    churn_rate=0.03,
    target_market_size=50000,  # Total addressable tennis coaches
    
    # Product Pricing & Usage
    pay_per_video_price=25.0,     # Price per individual video analysis
    basic_plan_price=150.0,       # Monthly subscription for basic tier
    premium_plan_price=299.99,    # Monthly subscription for advanced tier
    premium_plan_ratio=0.2,       # Percentage of subscribers on premium
    pay_per_video_ratio=0.4,      # Percentage of users who pay per video
    videos_per_coach_monthly=20,  # Average videos analyzed per coach
    
    # COGS - Per Video
    inference_cost_per_video=2.0,  # GPU compute cost
    storage_cost_per_video=0.5,    # Storage cost per video
    video_length_minutes=10,       # Average video length
    storage_cost_per_minute=0.03,  # Storage cost per minute
    
    # Infrastructure Costs
    gpu_inference_cost_per_video=0.15,  # Cost to process each video
    storage_cost_per_gb=0.02,           # Monthly storage cost per GB
    video_processing_server_monthly=500, # Fixed server costs
    
    # Development Team
    ml_engineers=1,
    ml_engineer_salary=120000,
    data_scientists=1,
    data_scientist_salary=80000,
    backend_developers=1,
    backend_salary=110000,
    frontend_developers=1,
    frontend_salary=100000,
    
    # Sales & Marketing
    sales_staff=1,
    sales_salary=60000,
    marketing_staff=1,
    marketing_salary=80000,
    marketing_budget_monthly=5000,  # Ongoing marketing spend
    
    # Management & Support
    executive_count=1,
    executive_salary=150000,
    support_staff=1,
    support_salary=50000,
    training_materials_monthly=500,
    
    # Office & Operations
    office_rent_monthly=3000,
    utilities_monthly=500,
    software_licenses_monthly=500,
    insurance_monthly=1500,
    legal_accounting_monthly=1500,
    
    # Benefits & Equipment
    benefits_percent=0.20,
    wfh_stipend_monthly=100,
    travel_budget_monthly=2000,
    laptop_cost=2500,
    desk_setup_cost=1000,
    
    # Financial Assumptions
    tax_rate=0.25,
    depreciation_years=3,
    
    # Lean Startup Team
    founder_salary=17000,  # Monthly
    contractor_ml_rate=75,
    contractor_frontend_rate=60,
    monthly_contractor_budget=5000,
    
    # Legal & HR Support
    legal_monthly_retainer=1000,     # Basic legal services retainer
    hr_monthly_retainer=500,         # Fractional HR support
    rippling_monthly_base=40,        # Rippling base monthly fee
    rippling_monthly_per_person=8,   # Rippling per person monthly fee
    company_setup_costs={            # One-time company setup costs
        'incorporation': 500,         # Delaware C-Corp filing
        'legal_setup': 2500,         # Initial legal documentation
        'accounting_setup': 1000,     # Accounting system setup
        'banking_setup': 200,         # Business banking setup
        'licenses_permits': 500,      # Basic business licenses
        'trademark': 1000,           # Basic trademark registration
    },
    # Federation & Academy Licensing
    enable_enterprise_licensing=True,  # Toggle for enterprise licensing model
    starting_enterprise_clients=1,     # Initial number of federation/academy clients
    enterprise_monthly_growth=0.02,    # Monthly growth rate for enterprise clients
    enterprise_churn_rate=0.01,        # Lower churn due to longer contracts
    enterprise_license_tiers={
        'small': {
            'price': 999.99,           # Monthly price for small academies (up to 10 coaches)
            'ratio': 0.5,              # 50% of enterprise clients are small academies
            'videos_per_month': 200    # Average videos analyzed per month
        },
        'medium': {
            'price': 2499.99,          # Monthly price for medium academies/federations (up to 30 coaches)
            'ratio': 0.3,              # 30% are medium-sized
            'videos_per_month': 600    # Average videos analyzed per month
        },
        'large': {
            'price': 4999.99,          # Monthly price for large federations (unlimited coaches)
            'ratio': 0.2,              # 20% are large federations
            'videos_per_month': 1500   # Average videos analyzed per month
        }
    },
):
    """
    Generate P&L including initial development phase and enterprise licensing
    """
    total_months = months + development_months
    months_list = list(range(1, total_months + 1))
    
    # Initialize all tracking arrays for full timeline
    active_coaches = [0] * total_months
    subscription_users_list = [0] * total_months
    pay_per_video_users = [0] * total_months
    videos_processed_subscription = [0] * total_months
    videos_processed_pay_per_video = [0] * total_months
    monthly_videos_list = [0] * total_months
    contractor_hours_list = [0] * total_months
    
    # Revenue tracking
    subscription_revenue = [0] * total_months
    pay_per_video_revenue = [0] * total_months
    total_revenue = [0] * total_months
    
    # Cost tracking
    personnel_costs = [0] * total_months
    infrastructure_costs = [0] * total_months
    processing_costs = [0] * total_months
    storage_costs = [0] * total_months
    inference_costs = [0] * total_months
    variable_support_costs = [0] * total_months
    operating_costs = [0] * total_months
    cogs_list = [0] * total_months
    total_videos_list = [0] * total_months
    
    # Initialize profit tracking arrays
    gross_profit = [0] * total_months
    operating_profit = [0] * total_months
    net_income = [0] * total_months
    
    # Track marketing spend
    marketing_spend = [0] * total_months
    
    # Development phase calculations
    for m in range(development_months):
        # Personnel costs during development
        month_dev_personnel = (
            founder_salary +  # Your salary
            (development_contractor_hours_monthly * contractor_ml_rate * 0.7) +  # ML/Backend work
            (development_contractor_hours_monthly * contractor_frontend_rate * 0.3)  # Frontend work
        )
        personnel_costs[m] = month_dev_personnel
        
        # Infrastructure and other costs during development
        month_dev_infrastructure = sum(development_costs_monthly.values())
        infrastructure_costs[m] = month_dev_infrastructure
        
        # Track contractor hours during development
        contractor_hours_list[m] = development_contractor_hours_monthly
        
        # Calculate total costs for development phase
        operating_costs[m] = month_dev_personnel + month_dev_infrastructure
        
        # All costs, no revenue during development
        gross_profit[m] = 0
        operating_profit[m] = -operating_costs[m]
        net_income[m] = operating_profit[m]
    
    # Add company setup costs to first month's expenses
    if len(operating_costs) > 0:
        operating_costs[0] += sum(company_setup_costs.values())

    # Add monthly legal and HR costs to operating costs
    for i in range(total_months):
        operating_costs[i] += legal_monthly_retainer + hr_monthly_retainer
    
    # Calculate Rippling costs based on team size
    for i in range(total_months):
        # Calculate total team size (including contractors)
        total_team_size = 1  # Start with founder
        if i >= development_months:  # Add team members after development phase
            total_team_size += (
                ml_engineers +
                data_scientists +
                backend_developers +
                frontend_developers +
                sales_staff +
                support_staff
            )
        
        # Add Rippling costs to operating costs
        operating_costs[i] += rippling_monthly_base + (rippling_monthly_per_person * total_team_size)
    
    # Enterprise client calculations
    if enable_enterprise_licensing:
        # Initialize all enterprise tracking arrays
        enterprise_clients = [0] * development_months  # Start with zero during development
        enterprise_clients.append(starting_enterprise_clients)  # Add initial clients at launch
        
        # Initialize tracking arrays with zeros for development phase
        small_academy_clients = [0] * development_months
        medium_academy_clients = [0] * development_months
        large_federation_clients = [0] * development_months
        small_academy_revenue = [0] * development_months
        medium_academy_revenue = [0] * development_months
        large_federation_revenue = [0] * development_months
        enterprise_revenue = [0] * development_months
        enterprise_videos = [0] * development_months
        
        # Calculate growth after development phase
        for m in range(development_months + 1, total_months):
            current_clients = enterprise_clients[-1]
            new_clients = current_clients * enterprise_monthly_growth
            churned_clients = current_clients * enterprise_churn_rate
            enterprise_clients.append(max(0, current_clients + new_clients - churned_clients))
        
        # Calculate revenue and videos for operational phase
        for clients in enterprise_clients[development_months:]:
            # Calculate clients by tier
            small_clients = clients * enterprise_license_tiers['small']['ratio']
            medium_clients = clients * enterprise_license_tiers['medium']['ratio']
            large_clients = clients * enterprise_license_tiers['large']['ratio']
            
            # Store client numbers
            small_academy_clients.append(small_clients)
            medium_academy_clients.append(medium_clients)
            large_federation_clients.append(large_clients)
            
            # Calculate and store revenue by tier
            small_rev = small_clients * enterprise_license_tiers['small']['price']
            medium_rev = medium_clients * enterprise_license_tiers['medium']['price']
            large_rev = large_clients * enterprise_license_tiers['large']['price']
            
            small_academy_revenue.append(small_rev)
            medium_academy_revenue.append(medium_rev)
            large_federation_revenue.append(large_rev)
            
            monthly_enterprise_revenue = small_rev + medium_rev + large_rev
            monthly_enterprise_videos = (
                small_clients * enterprise_license_tiers['small']['videos_per_month'] +
                medium_clients * enterprise_license_tiers['medium']['videos_per_month'] +
                large_clients * enterprise_license_tiers['large']['videos_per_month']
            )
            
            enterprise_revenue.append(monthly_enterprise_revenue)
            enterprise_videos.append(monthly_enterprise_videos)
    else:
        # Initialize all arrays with zeros if enterprise licensing is disabled
        enterprise_revenue = [0] * total_months
        enterprise_videos = [0] * total_months
        small_academy_clients = [0] * total_months
        medium_academy_clients = [0] * total_months
        large_federation_clients = [0] * total_months
        small_academy_revenue = [0] * total_months
        medium_academy_revenue = [0] * total_months
        large_federation_revenue = [0] * total_months
    
    # Operational phase calculations
    current_coaches = starting_coaches
    for m in range(development_months, total_months):
        # Growth calculations
        current_coaches = current_coaches * (1 + monthly_growth_rate - churn_rate)
        active_coaches[m] = current_coaches
        
        # Calculate users by type
        subscription_users = current_coaches * (1 - pay_per_video_ratio)
        pay_per_video_users_count = current_coaches * pay_per_video_ratio
        
        # Split subscription users into basic and premium
        premium_users = subscription_users * premium_plan_ratio
        basic_users = subscription_users * (1 - premium_plan_ratio)
        
        # Calculate videos processed
        subscription_videos = subscription_users * videos_per_coach_monthly
        pay_per_video_count = pay_per_video_users_count * videos_per_coach_monthly
        total_videos = subscription_videos + pay_per_video_count
        
        # Calculate processing and storage costs
        month_processing_cost = total_videos * gpu_inference_cost_per_video
        month_storage_cost = total_videos * storage_cost_per_video * storage_cost_per_gb
        month_infrastructure = video_processing_server_monthly + month_processing_cost + month_storage_cost
        
        # Track costs
        processing_costs[m] = month_processing_cost
        storage_costs[m] = month_storage_cost
        infrastructure_costs[m] = month_infrastructure
        
        # Calculate monthly revenue
        month_subscription = basic_users * basic_plan_price + premium_users * premium_plan_price
        month_pay_per_video = pay_per_video_count * pay_per_video_price
        month_total_revenue = month_subscription + month_pay_per_video
        
        # Track revenue
        subscription_revenue[m] = month_subscription
        pay_per_video_revenue[m] = month_pay_per_video
        total_revenue[m] = month_total_revenue
        
        # Calculate COGS
        month_cogs = (
            month_processing_cost +  # GPU inference
            month_storage_cost +     # Storage costs
            month_infrastructure     # Fixed infrastructure
        )
        cogs_list[m] = month_cogs
        
        # Calculate marketing budget based on revenue
        month_marketing = calculate_marketing_budget(month_total_revenue, marketing_budget_monthly)
        marketing_spend[m] = month_marketing
        
        # Calculate profits with all costs included
        gross_profit[m] = month_total_revenue - month_cogs
        
        total_monthly_costs = (
            month_cogs +            # Cost of goods sold
            personnel_costs[m] +    # Staff & contractors
            operating_costs[m] +    # Office, equipment, etc.
            month_marketing +       # Marketing spend
            infrastructure_costs[m] # Infrastructure costs
        )
        
        operating_profit[m] = month_total_revenue - total_monthly_costs
        net_income[m] = operating_profit[m] * (1 - tax_rate)  # Apply taxes
        
        # Track usage
        videos_processed_subscription[m] = subscription_videos
        videos_processed_pay_per_video[m] = pay_per_video_count
        monthly_videos_list[m] = total_videos
        
        # Update operating costs to include scaled marketing
        operating_costs[m] = (
            office_rent_monthly +
            month_marketing +  # Now using scaled marketing budget
            wfh_stipend_monthly +
            travel_budget_monthly
        )
        
        # Update total revenue and video processing calculations
        total_revenue[m] += enterprise_revenue[m]
        monthly_videos_list[m] += enterprise_videos[m]
        
        # Update processing costs for enterprise videos
        processing_costs[m] += enterprise_videos[m] * (
            inference_cost_per_video +
            storage_cost_per_video
        )
    
    # Create DataFrame with all metrics
    data = {
        'Month': months_list,
        'Active Coaches': [round(x) for x in active_coaches],
        'Market Penetration (%)': [round(x/target_market_size*100, 2) for x in active_coaches],
        
        # Revenue metrics
        'Subscription Revenue': subscription_revenue,
        'Pay-per-Video Revenue': pay_per_video_revenue,
        'Total Revenue': total_revenue,
        
        # Cost metrics
        'COGS': cogs_list,
        'Processing Costs': processing_costs,
        'Storage Costs': storage_costs,
        'Personnel Costs': personnel_costs,
        'Infrastructure Costs': infrastructure_costs,
        'Operating Costs': operating_costs,
        
        # Profit metrics
        'Gross Profit': gross_profit,
        'Operating Profit': operating_profit,
        'Net Income': net_income,
        
        # Usage metrics
        'Pay-per-Video Users': [round(x) for x in pay_per_video_users],
        'Subscription Users': [round(x) for x in subscription_users_list],
        'Videos (Subscription)': [round(x) for x in videos_processed_subscription],
        'Videos (Pay-per-Video)': [round(x) for x in videos_processed_pay_per_video],
        
        # Contractor metrics
        'Contractor Hours': contractor_hours_list,
        'Contractor Costs': [min(monthly_contractor_budget, hours * ((contractor_ml_rate * 0.8) + (contractor_frontend_rate * 0.2))) 
                           for hours in contractor_hours_list],
        
        # Marketing metrics
        'Marketing Spend': marketing_spend,
        'Marketing % of Revenue': [
            (spend / rev * 100) if rev > 0 else 0 
            for spend, rev in zip(marketing_spend, total_revenue)
        ]
    }
    
    # Create cumulative metrics
    data.update({
        # Cumulative Revenue
        'Cumulative Revenue': np.cumsum(total_revenue),
        'Cumulative Costs': np.cumsum([x + y + z + m + i for x, y, z, m, i in zip(
            cogs_list,
            operating_costs,
            personnel_costs,
            marketing_spend,
            infrastructure_costs
        )]),
        'Cumulative Profit/Loss': [rev - cost for rev, cost in zip(
            np.cumsum(total_revenue),
            np.cumsum([x + y + z + m + i for x, y, z, m, i in zip(
                cogs_list,
                operating_costs,
                personnel_costs,
                marketing_spend,
                infrastructure_costs
            )])
        )],
        
        # Running cash position
        'Cash Position': np.cumsum(net_income),
        'Investment Required': [-min(0, x) for x in np.cumsum(net_income)]
    })
    
    # Add enterprise-specific metrics
    data.update({
        'Enterprise Revenue': enterprise_revenue,
        'Enterprise Videos': enterprise_videos,
        'Monthly Videos': monthly_videos_list,
        'Small Academy Clients': small_academy_clients,
        'Medium Academy Clients': medium_academy_clients,
        'Large Federation Clients': large_federation_clients,
        'Small Academy Revenue': small_academy_revenue,
        'Medium Academy Revenue': medium_academy_revenue,
        'Large Federation Revenue': large_federation_revenue,
    })
    
    # Add validation check to ensure P&L is correct
    for i in range(len(data['Cumulative Revenue'])):
        if data['Cumulative Revenue'][i] < data['Cumulative Costs'][i]:
            assert data['Cumulative Profit/Loss'][i] <= 0, f"Error: Month {i+1} shows positive P&L despite cumulative costs exceeding revenue"
    
    return pd.DataFrame(data)

def calculate_required_staff(monthly_videos, active_coaches, monthly_revenue):
    """
    Calculate required staff based on business metrics - lean startup approach
    """
    # Contractor/Part-time Needs (hours per month)
    contractor_hours_needed = max(40, monthly_videos // 100)  # 1 hour per 100 videos, min 40hrs
    
    # Full-time Staff Scaling
    if monthly_revenue < 50000:  # Under $50k monthly revenue
        return {
            'full_time_staff': {
                'technical_founder': 1,  # You as CTO
                'ml_engineers': 0,
                'backend_developers': 0,
                'frontend_developers': 0,
                'sales_staff': 0,
                'support_staff': 0
            },
            'contractor_hours': {
                'ml_development': contractor_hours_needed * 0.4,  # 40% of contractor time
                'backend_development': contractor_hours_needed * 0.4,
                'frontend_development': contractor_hours_needed * 0.2
            }
        }
    else:  # Begin scaling with full-time hires
        return {
            'full_time_staff': {
                'technical_founder': 1,
                'ml_engineers': max(0, monthly_videos // 10000),
                'backend_developers': max(0, monthly_videos // 15000),
                'frontend_developers': max(0, active_coaches // 2000),
                'sales_staff': max(0, monthly_revenue // 300000),
                'support_staff': max(0, active_coaches // 1000)
            },
            'contractor_hours': {
                'ml_development': max(0, contractor_hours_needed * 0.2),
                'backend_development': max(0, contractor_hours_needed * 0.2),
                'frontend_development': max(0, contractor_hours_needed * 0.1)
            }
        }

def calculate_sensitivity_metrics(df):
    """Calculate key metrics from a P&L dataframe"""
    metrics = {
        'Total Investment': df['Investment Required'].max(),
        'Months to Breakeven': next((i + 1 for i, val in enumerate(df['Cumulative Profit/Loss']) if val > 0)),
        'Peak Monthly Revenue': df['Total Revenue'].max(),
        'Year 1 Revenue': df.iloc[:12]['Total Revenue'].sum(),
        'Final Monthly Revenue': df['Total Revenue'].iloc[-1],
        'Max Monthly Profit': df['Net Income'].max()
    }
    return metrics

def run_sensitivity_analysis(base_params, param_ranges):
    """
    Run sensitivity analysis with given parameter ranges
    """
    results = []
    
    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]
    combinations = list(product(*param_values))
    
    # Create progress bar
    progress_bar = st.progress(0)
    total_combinations = len(combinations)
    
    for i, combo in enumerate(combinations):
        # Create parameter dictionary for this combination
        test_params = base_params.copy()
        for name, value in zip(param_names, combo):
            if name == "team_and_ops_monthly":
                # Distribute the team_and_ops_monthly value across its components
                current_total = (
                    base_params.get('founder_salary', 0) +
                    base_params.get('monthly_contractor_budget', 0) +
                    base_params.get('office_rent_monthly', 0) +
                    base_params.get('utilities_monthly', 0) +
                    base_params.get('software_licenses_monthly', 0) +
                    base_params.get('insurance_monthly', 0) +
                    base_params.get('legal_accounting_monthly', 0) +
                    base_params.get('wfh_stipend_monthly', 0) +
                    base_params.get('travel_budget_monthly', 0)
                )
                
                if current_total > 0:
                    ratio = value / current_total
                    test_params.update({
                        'founder_salary': base_params.get('founder_salary', 0) * ratio,
                        'monthly_contractor_budget': base_params.get('monthly_contractor_budget', 0) * ratio,
                        'office_rent_monthly': base_params.get('office_rent_monthly', 0) * ratio,
                        'utilities_monthly': base_params.get('utilities_monthly', 0) * ratio,
                        'software_licenses_monthly': base_params.get('software_licenses_monthly', 0) * ratio,
                        'insurance_monthly': base_params.get('insurance_monthly', 0) * ratio,
                        'legal_accounting_monthly': base_params.get('legal_accounting_monthly', 0) * ratio,
                        'wfh_stipend_monthly': base_params.get('wfh_stipend_monthly', 0) * ratio,
                        'travel_budget_monthly': base_params.get('travel_budget_monthly', 0) * ratio
                    })
            else:
                test_params[name] = value
        
        # Generate P&L with these parameters
        df_pnl = generate_video_analysis_pnl(**test_params)
        
        # Calculate metrics
        metrics = calculate_sensitivity_metrics(df_pnl)
        
        # Store results
        for param_idx, param_name in enumerate(param_names):
            results.append({
                'Parameter': param_name,
                'Parameter Value': combo[param_idx],
                **metrics
            })
        
        # Update progress
        progress_bar.progress((i + 1) / total_combinations)
    
    return pd.DataFrame(results)

def get_sensitivity_ranges(ui_inputs):
    """
    Generate parameter ranges for sensitivity analysis based on UI inputs
    """
    return {
        "development_months": np.linspace(
            ui_inputs["min_dev_months"],
            ui_inputs["max_dev_months"],
            ui_inputs["dev_month_steps"]
        ).astype(int),
        
        "monthly_growth_rate": np.linspace(
            ui_inputs["min_growth_rate"],
            ui_inputs["max_growth_rate"],
            ui_inputs["growth_steps"]
        ),
        
        "churn_rate": np.linspace(
            ui_inputs["min_churn_rate"],
            ui_inputs["max_churn_rate"],
            ui_inputs["churn_steps"]
        ),
        
        "starting_coaches": np.linspace(
            ui_inputs["min_coaches"],
            ui_inputs["max_coaches"],
            ui_inputs["coach_steps"]
        ).astype(int),
        
        "basic_plan_price": np.linspace(
            ui_inputs["min_basic_price"],
            ui_inputs["max_basic_price"],
            ui_inputs["price_steps"]
        ),
        
        "pay_per_video_price": np.linspace(
            ui_inputs["min_video_price"],
            ui_inputs["max_video_price"],
            ui_inputs["video_price_steps"]
        )
    }

def main():
    st.title(" Video Analysis - Financial Model")
    # Update scaling triggers explanation
    st.sidebar.markdown("""
        ### Lean Startup Team Structure
        
        **Phase 1 (Pre-$50k MRR):**
        - Founder/CTO (or whatever title we decide)
        - Contractors via Upwork/similar platforms
        - Estimated contractor needs:
            - ML/Backend: 40-80 hours/month
            - Frontend: 20-40 hours/month
        
        **Phase 2 (Post-$50k MRR):**
        - Begin transitioning to full-time hires
        - Reduce contractor reliance
        - First hires typically in ML/Backend
        """)
    # Add market context at the top
    st.sidebar.markdown("""
    ### Market Context & Assumptions
    
    **Current Market Price Points:**
    - Standard Per-Match Rate: \$60-\$120
    - Tennis Analytics Example:
        - Single match: \$90-\$110
        - 10 matches: \$800-\$1,000 (\$80-\$100/match)
        - 100 matches: \$9,000-\$11,000 (\$90-\$110/match)
        - 200 matches: \$18,500-\$22,000 (\$92.50-\$110/match)
    
    **Cost Structure Per Match:**
    - Cloud GPU Inference: \$4-\$6
    - Storage & Bandwidth: \$2-\$4
    - Optional QA/Human Review: \$10-\$15
    - Customer Support (prorated): \$3-\$5
    
    **Target Gross Margins:**
    - Fully Automated: 85-90%
    - With Human QA: 70-75%
    """)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Market & Product",
        "Technical Infrastructure",
        "Team & Operations",
        "Financial Results",
        "Investment Analysis",
        "Sensitivity Analysis"
    ])

    
    
    with tab1:
        st.header("Market & Product Assumptions")
        
        # Add Development Phase section
        st.subheader("Development Phase")
        col1, col2 = st.columns(2)
        
        with col1:
            development_months = st.number_input(
                "Development Months Before Launch", 
                3, 12, 6,
                help="Months needed to develop MVP"
            )
            
            development_contractor_hours = st.number_input(
                "Monthly Contractor Hours During Development",
                40, 200, 160,
                help="Hours of contractor help needed monthly during development"
            )
            
            st.info("""
            **Development Timeline:**
            - Month 1-2: Core ML pipeline
            - Month 3-4: Video processing
            - Month 5: User interface
            - Month 6: Testing & refinement
            """)
        
        with col2:
            st.subheader("Development Costs")
            cloud_costs = st.number_input(
                "Monthly Cloud Infrastructure ($)", 
                100, 2000, 500,
                help="AWS/GCP costs during development"
            )
            
            data_collection = st.number_input(
                "Monthly Data Collection ($)",
                0, 5000, 1000,
                help="Cost to acquire training data/videos"
            )
            
            testing_equipment = st.number_input(
                "Monthly Testing Equipment ($)",
                0, 2000, 500,
                help="Cameras, testing devices, etc."
            )
            
            dev_software = st.number_input(
                "Monthly Software Licenses ($)",
                0, 1000, 200,
                help="Development tools, APIs, etc."
            )
            
            st.info("""
            **Development Phase Strategy:**
            - Focus on MVP features
            - Collect minimal training data
            - Use cloud resources efficiently
            - Test with friendly users
            """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Market")
            months = st.slider("Projection Months", 12, 60, 24)
            starting_coaches = st.number_input("Starting Coaches", 10, 1000, 20)
            monthly_growth_rate = st.slider("Monthly Growth Rate (%)", 0.0, 0.5, 0.05)
            churn_rate = st.slider("Monthly Churn Rate (%)", 0.0, 0.2, 0.03)
            target_market_size = st.number_input("Target Market Size (Coaches)", 1000, 100000, 50000)
            
            st.info("""
            **Market Size Note:** 
            - Targeting tennis coaches, academies, and college programs.
            - Globally, there are an estimated 149,110 tennis coaches
            - Typical coach analyzes 20-40 matches per month.
            """)
        
        with col2:
            st.subheader("Subscription Pricing")
            basic_plan_price = st.number_input("Basic Plan Monthly ($)", 0.0, 500.0, 300.0, 
                help="Monthly subscription for up to 5 matches ($60/match)")
            premium_plan_price = st.number_input("Premium Plan Monthly ($)", 0.0, 1000.0, 500.0,
                help="Premium plan with advanced analytics and more matches")
            premium_plan_ratio = st.slider("Premium Plan Users (%)", 0.0, 1.0, 0.2,
                help="Percentage of users opting for premium features")
            
            st.info("""
            **Pricing Strategy:**
            - Basic Plan: $300/month for 5 matches
            - Premium: Additional features like biomechanics
            - Annual plans available at discount
            """)
            
        with col3:
            st.subheader("Pay-per-Video")
            pay_per_video_price = st.number_input("Price per Video ($)", 0.0, 150.0, 90.0,
                help="Single match analysis price")
            pay_per_video_ratio = st.slider("Pay-per-Video Users (%)", 0.0, 1.0, 0.4,
                help="Percentage of users preferring pay-per-video vs subscription")
            videos_per_coach_monthly = st.number_input("Avg Videos per Coach", 1, 100, 20,
                help="Average number of matches analyzed per coach per month")
            
            st.info("""
            **Volume Pricing:**
            - Single match: \$90
            - 10 match package: \$800 (\$80 per match)
            - 100 matches: \$7,500-\$8,500 (\$75-\$85 per match)
            """)
        
        # Add Enterprise Licensing section after the existing columns
        st.subheader("Enterprise Licensing")
        enable_enterprise = st.checkbox("Enable Federation/Academy Licensing", value=True,
            help="Include revenue from tennis federations and large academies")
        
        if enable_enterprise:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Small Academy License**")
                small_academy_price = st.number_input("Monthly Price ($)", 500, 2000, 1000,
                    help="Price for academies with up to 10 coaches")
                small_academy_ratio = st.slider("% of Enterprise Clients", 0.0, 1.0, 0.5,
                    help="Percentage of clients in this tier")
                small_academy_videos = st.number_input("Monthly Videos", 100, 500, 200,
                    help="Average videos analyzed per month")
                
            with col2:
                st.markdown("**Medium Academy/Federation**")
                medium_academy_price = st.number_input("Medium Tier Monthly Price ($)", 1500, 4000, 2500,
                    help="Price for organizations with up to 30 coaches")
                medium_academy_ratio = st.slider("% of Medium Clients", 0.0, 1.0, 0.3,
                    help="Percentage of clients in this tier")
                medium_academy_videos = st.number_input("Monthly Videos (Medium)", 300, 1000, 600,
                    help="Average videos analyzed per month")
                
            with col3:
                st.markdown("**Large Federation License**")
                large_federation_price = st.number_input("Large Federation Monthly Price ($)", 3000, 8000, 5000,
                    help="Price for large federations (unlimited coaches)")
                large_federation_ratio = st.slider("% of Large Clients", 0.0, 1.0, 0.2,
                    help="Percentage of clients in this tier")
                large_federation_videos = st.number_input("Monthly Videos (Large)", 1000, 3000, 1500,
                    help="Average videos analyzed per month")
            
            col1, col2 = st.columns(2)
            with col1:
                starting_enterprise = st.number_input("Initial Enterprise Clients", 0, 10, 1,
                    help="Number of federation/academy clients at launch")
                enterprise_growth = st.slider("Monthly Enterprise Growth Rate (%)", 0.0, 0.10, 0.02,
                    help="Monthly growth rate for enterprise clients")
            
            with col2:
                enterprise_churn = st.slider("Enterprise Churn Rate (%)", 0.0, 0.05, 0.01,
                    help="Monthly churn rate for enterprise clients")
            
            st.info("""
            **Enterprise Licensing Model:**
            - Small Academy: Perfect for local tennis academies with limited coach count
            - Medium Tier: Ideal for larger academies and small national federations
            - Large Federation: Unlimited usage for major tennis organizations
            
            Each tier includes:
            - Bulk video analysis
            - Advanced analytics dashboard
            - Priority support
            - Custom branding options
            """)
    
    with tab2:
        st.header("Technical Infrastructure")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Processing Costs")
            gpu_inference_cost_per_video = st.number_input("GPU Cost per Video ($)", 0.0, 10.0, 5.0,
                help="Cloud GPU inference cost per 2-hour match")
            video_processing_server_monthly = st.number_input("Fixed Server Costs Monthly ($)", 0, 5000, 2000,
                help="Fixed infrastructure costs")
            
            st.info("""
            **Processing Assumptions:**
            - 2-hour average match length
            - Scene detection & stroke classification
            - Advanced metrics extraction
            - Temporary storage during processing
            """)
        
        with col2:
            st.subheader("Storage")
            storage_per_video_gb = st.number_input("Storage per Video (GB)", 0.0, 10.0, 2.0,
                help="Average storage needed per match video")
            storage_cost_per_gb = st.number_input("Storage Cost per GB ($)", 0.0, 1.0, 0.02,
                help="Cloud storage cost per GB per month")
            
            st.info("""
            **Storage Strategy:**
            - Temporary high-res storage for analysis
            - Compressed long-term storage
            - Optional video archival for customers
            """)
    
    with tab3:
        st.header("Team & Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Legal & HR Support")
            legal_monthly_retainer = st.number_input(
                "Monthly Legal Retainer ($)", 
                0, 5000, 1000,
                help="Basic legal services and compliance support"
            )
            hr_monthly_retainer = st.number_input(
                "Monthly HR Support ($)", 
                0, 2000, 500,
                help="Fractional HR services and compliance"
            )
            
            st.subheader("HR Software")
            rippling_monthly_base = st.number_input(
                "Rippling Base Monthly Fee ($)", 
                0, 200, 40,
                help="Rippling platform base monthly fee"
            )
            rippling_monthly_per_person = st.number_input(
                "Rippling Per Person Monthly Fee ($)", 
                0, 50, 8,
                help="Additional monthly fee per team member"
            )
            
            st.subheader("Company Setup Costs")
            incorporation_cost = st.number_input(
                "Incorporation Costs ($)", 
                0, 2000, 500,
                help="Delaware C-Corp filing fees"
            )
            legal_setup = st.number_input(
                "Legal Setup ($)", 
                0, 5000, 2500,
                help="Initial legal documentation and agreements"
            )
            accounting_setup = st.number_input(
                "Accounting Setup ($)", 
                0, 2000, 1000,
                help="Accounting system and initial setup"
            )
            banking_setup = st.number_input(
                "Banking Setup ($)", 
                0, 500, 200,
                help="Business banking and payment processing setup"
            )
            licenses_permits = st.number_input(
                "Licenses & Permits ($)", 
                0, 1000, 500,
                help="Basic business licenses and permits"
            )
            trademark_cost = st.number_input(
                "Trademark Registration ($)", 
                0, 2000, 1000,
                help="Basic trademark registration costs"
            )

            # Add info box explaining these costs
            st.info("""
            **Legal & HR Support:**
            - Legal retainer covers basic contract review, compliance, and legal guidance
            - HR support includes policy development, compliance, and basic employee relations
            - Rippling provides payroll, benefits, and HR management automation
            
            **Company Setup:**
            - One-time costs for proper business formation
            - Includes essential legal, banking, and compliance setup
            - Trademark protection for brand and IP
            """)

        with col2:
            st.subheader("Technical Team")
            
            # Founder/CTO Compensation
            founder_salary = st.number_input("Founder/CTO Monthly Salary", 0, 20000, 17000,
                help="Monthly compensation - technical founder")
            
            # Contractor Rates & Hours
            st.subheader("Contractors")
            contractor_ml_rate = st.number_input("ML/Backend Contractor Rate ($/hr)", 30, 150, 75)
            contractor_frontend_rate = st.number_input("Frontend Contractor Rate ($/hr)", 25, 120, 60)
            
            monthly_contractor_budget = st.number_input("Monthly Contractor Budget ($)", 
                0, 20000, 5000,
                help="Maximum monthly spend on contractors")
            
            st.info("""
            **Initial Technical Strategy:**
            - You handle core ML/backend architecture
            - Use contractors for specific features/tasks
            - Gradually reduce contractor reliance
            """)
        
        with col3:
            st.subheader("Future Hires (Post-$50k MRR)")
            show_future_hires = st.checkbox("Show Future Hire Planning", value=False)
            
            if show_future_hires:
                ml_engineer_salary = st.number_input("Future ML Engineer Salary", 80000, 200000, 120000)
                backend_salary = st.number_input("Future Backend Dev Salary", 80000, 200000, 110000)
                frontend_salary = st.number_input("Future Frontend Dev Salary", 80000, 200000, 100000)
            
            st.subheader("Marketing & Sales")
            marketing_budget_monthly = st.number_input("Monthly Marketing Budget ($)", 0, 20000, 2000,
                help="Initial marketing spend (ads, content, events)")
            
            
            st.info("""
            **Marketing Budget Strategy:**
            - Initial Stage: Fixed budget of ${:,.0f}
            - >$50k MRR: Increases to 10% of revenue
            - >$100k MRR: Increases to 15% of revenue
            
            This helps scale marketing with revenue growth while 
            maintaining profitability targets.
            """.format(marketing_budget_monthly))
            
            st.info("""
            **Marketing Strategy:**
            - Focus on digital marketing
            - Tennis academy outreach
            - Content marketing & demos
            - Scale with revenue growth
            """)
            
            st.info("""
            **Hiring Strategy:**
            - First hire: ML/Backend Engineer
            - Second: Customer Support
            - Third: Sales/Business Development
            """)
        
            st.subheader("Office & Equipment")
            office_rent_monthly = st.number_input("Monthly Office Rent ($)", 0, 10000, 1000,
                help="Initial home office or co-working space")
            
            # Add these new UI elements
            utilities_monthly = st.number_input("Monthly Utilities ($)", 0, 2000, 500,
                help="Utilities and internet costs")
            software_licenses_monthly = st.number_input("Monthly Software Licenses ($)", 0, 2000, 500,
                help="Software subscriptions and tools")
            insurance_monthly = st.number_input("Monthly Insurance ($)", 0, 5000, 1500,
                help="Business insurance costs")
            legal_accounting_monthly = st.number_input("Monthly Legal/Accounting ($)", 0, 5000, 1500,
                help="Legal and accounting services")
            
            st.subheader("Benefits & Equipment")
            benefits_percent = st.slider("Benefits (% of salary)", 0.0, 0.4, 0.20,
                help="Percentage of salary for benefits (when applicable)")
            wfh_stipend_monthly = st.number_input("WFH Stipend ($)", 0, 500, 100,
                help="Monthly work from home allowance")
            travel_budget_monthly = st.number_input("Monthly Travel Budget ($)", 0, 10000, 1000,
                help="For client meetings and demos")
            
            st.subheader("One-time Setup")
            laptop_cost = st.number_input("Development Laptop Cost ($)", 0, 7000, 4000,
                help="High-performance development machine")
            desk_setup_cost = st.number_input("Home Office Setup ($)", 0, 2000, 500,
                help="Desk, chair, monitors, etc.")
            
            st.info("""
            **Initial Setup Strategy:**
            - Start with home office/co-working
            - Minimal but effective equipment
            - Scale office space with team growth
            """)
    
    with tab4:
        df_pnl = generate_video_analysis_pnl(
            development_months=development_months,
            development_contractor_hours_monthly=development_contractor_hours,
            development_costs_monthly={
                'cloud_infrastructure': cloud_costs,
                'data_collection': data_collection,
                'testing_equipment': testing_equipment,
                'software_licenses': dev_software
            },
            months=months,
            starting_coaches=starting_coaches,
            monthly_growth_rate=monthly_growth_rate,
            churn_rate=churn_rate,
            target_market_size=target_market_size,
            
            # Product Pricing & Usage
            pay_per_video_price=pay_per_video_price,
            basic_plan_price=basic_plan_price,
            premium_plan_price=premium_plan_price,
            premium_plan_ratio=premium_plan_ratio,
            pay_per_video_ratio=pay_per_video_ratio,
            videos_per_coach_monthly=videos_per_coach_monthly,
            
            # Infrastructure & COGS
            inference_cost_per_video=gpu_inference_cost_per_video,
            storage_cost_per_video=storage_per_video_gb,
            video_processing_server_monthly=video_processing_server_monthly,
            storage_cost_per_gb=storage_cost_per_gb,
            
            # Lean Startup Team
            founder_salary=founder_salary,
            contractor_ml_rate=contractor_ml_rate,
            contractor_frontend_rate=contractor_frontend_rate,
            monthly_contractor_budget=monthly_contractor_budget,
            
            # Office & Operations
            office_rent_monthly=office_rent_monthly,
            utilities_monthly=utilities_monthly,
            software_licenses_monthly=software_licenses_monthly,
            insurance_monthly=insurance_monthly,
            legal_accounting_monthly=legal_accounting_monthly,
            
            # Benefits & Equipment
            benefits_percent=benefits_percent,
            wfh_stipend_monthly=wfh_stipend_monthly,
            travel_budget_monthly=travel_budget_monthly,
            laptop_cost=laptop_cost,
            desk_setup_cost=desk_setup_cost,
            
            # Marketing
            marketing_budget_monthly=marketing_budget_monthly,
            
            # Legal & HR Support
            legal_monthly_retainer=legal_monthly_retainer,
            hr_monthly_retainer=hr_monthly_retainer,
            rippling_monthly_base=rippling_monthly_base,
            rippling_monthly_per_person=rippling_monthly_per_person,
            company_setup_costs={
                'incorporation': incorporation_cost,
                'legal_setup': legal_setup,
                'accounting_setup': accounting_setup,
                'banking_setup': banking_setup,
                'licenses_permits': licenses_permits,
                'trademark': trademark_cost
            },
            # Federation & Academy Licensing
            enable_enterprise_licensing=enable_enterprise,
            starting_enterprise_clients=starting_enterprise if enable_enterprise else 0,
            enterprise_monthly_growth=enterprise_growth if enable_enterprise else 0,
            enterprise_churn_rate=enterprise_churn if enable_enterprise else 0,
            enterprise_license_tiers={
                'small': {
                    'price': small_academy_price,
                    'ratio': small_academy_ratio,
                    'videos_per_month': small_academy_videos
                },
                'medium': {
                    'price': medium_academy_price,
                    'ratio': medium_academy_ratio,
                    'videos_per_month': medium_academy_videos
                },
                'large': {
                    'price': large_federation_price,
                    'ratio': large_federation_ratio,
                    'videos_per_month': large_federation_videos
                }
            } if enable_enterprise else None,
        )
        
        # Development phase visualization
        st.subheader("Development Phase Costs")
        dev_phase_df = df_pnl.iloc[:development_months]
        st.line_chart(dev_phase_df[[
            'Personnel Costs',
            'Infrastructure Costs',
            'Operating Costs'
        ]])
        
        # Alternative visualization with more cost details
        st.subheader("Development Phase Breakdown")
        dev_costs_df = dev_phase_df[[
            'Personnel Costs',
            'Infrastructure Costs',
            'Operating Costs',
            'Contractor Hours',
            'Contractor Costs'
        ]]
        st.dataframe(dev_costs_df.style.format("{:,.2f}"))
        
        st.subheader("Key Metrics")
        metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs(["Growth", "Revenue", "Profitability"])
        
        with metrics_tab1:
            st.line_chart(df_pnl[[
                'Active Coaches', 
                'Pay-per-Video Users', 
                'Subscription Users'
            ]])
        
        with metrics_tab2:
            # Create a revenue breakdown chart including enterprise revenue
            st.subheader("Revenue Breakdown")
            st.line_chart(df_pnl[[
                'Subscription Revenue',
                'Pay-per-Video Revenue',
                'Enterprise Revenue',  # Add enterprise revenue
                'Total Revenue'
            ]])
            
            # Add enterprise metrics section
            if enable_enterprise:
                st.subheader("Enterprise Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.line_chart(df_pnl[[
                        'Small Academy Clients',
                        'Medium Academy Clients',
                        'Large Federation Clients'
                    ]])
                    st.caption("Enterprise Client Growth by Tier")
                
                with col2:
                    st.line_chart(df_pnl[[
                        'Small Academy Revenue',
                        'Medium Academy Revenue',
                        'Large Federation Revenue'
                    ]])
                    st.caption("Enterprise Revenue by Tier")
        
        with metrics_tab3:
            st.line_chart(df_pnl[[
                'Gross Profit', 
                'Operating Profit', 
                'Net Income'
            ]])
        
        # Add usage metrics chart
        st.subheader("Video Usage")
        st.line_chart(df_pnl[[
            'Videos (Subscription)',
            'Videos (Pay-per-Video)'
        ]])
        
        # Cost breakdown chart
        st.subheader("Cost Breakdown")
        st.line_chart(df_pnl[[
            'Processing Costs',
            'Storage Costs',
            'Personnel Costs',
            'Operating Costs'
        ]])

        st.subheader("Financial Projections")
        st.dataframe(df_pnl.style.format("{:,.2f}"))
    
    # New Investment Analysis tab
    with tab5:
        st.header("Investment Analysis")
        # Show burn rates with enterprise costs and revenue included
        st.subheader("Burn Rate Analysis")
        
        # Development phase burn
        dev_phase_costs = df_pnl.iloc[:development_months]
        dev_burn_rate = abs(dev_phase_costs['Net Income'].mean()) if len(dev_phase_costs) > 0 else 0
        
        # Calculate operational burn rate only for the operational phase
        operational_phase = df_pnl.iloc[development_months:]  # Get data after development
        
        # Find the first month of positive cumulative profit (breakeven)
        breakeven_index = next(
            (i for i, val in enumerate(operational_phase['Cumulative Profit/Loss']) if val > 0),
            len(operational_phase)
        )
        
        # Calculate burn rate using months until breakeven
        burn_period = operational_phase.iloc[:breakeven_index]
        
              
        # Calculate operational burn rate
        operational_burn_rate = abs(burn_period['Net Income'].mean()) if len(burn_period) > 0 else 0
        
        # Calculate enterprise-specific metrics for burn period
        if enable_enterprise and len(burn_period) > 0:
            enterprise_burn_contribution = burn_period['Enterprise Revenue'].mean()
            total_revenue = burn_period['Total Revenue'].sum()
            if total_revenue > 0:
                enterprise_revenue_ratio = burn_period['Enterprise Revenue'].sum() / total_revenue
                enterprise_cost_contribution = burn_period['Processing Costs'].mean() * enterprise_revenue_ratio
            else:
                enterprise_revenue_ratio = 0
                enterprise_cost_contribution = 0

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Development Phase Burn Rate",
                f"${dev_burn_rate:,.0f}/month",
                help="Average monthly cash burn during development (including setup costs)"
            )
        with col2:
            st.metric(
                "Operational Burn Rate",
                f"${operational_burn_rate:,.0f}/month",
                help="Average monthly cash burn from launch until breakeven"
            )
            if len(burn_period) > 0:
                st.caption(f"Burn period: {len(burn_period)} months")
        
        if enable_enterprise:
            with col3:
                st.metric(
                    "Enterprise Revenue Offset",
                    f"${enterprise_burn_contribution:,.0f}/month",
                    help="Average monthly enterprise revenue during burn period"
                )
            
            # Only show enterprise impact if there are actual numbers to report
            if enterprise_burn_contribution > 0 or enterprise_cost_contribution > 0:
                st.info(f"""
                **Enterprise Impact on Burn Rate:**
                - Enterprise revenue reduces burn rate by ${enterprise_burn_contribution:,.0f}/month
                - Processing costs for enterprise videos: ${enterprise_cost_contribution:,.0f}/month
                - Net impact: ${(enterprise_burn_contribution - enterprise_cost_contribution):,.0f}/month
                """)

        # Calculate and display key investment metrics right after burn rates
        total_investment = df_pnl['Investment Required'].max()
        dev_phase_cost = abs(df_pnl.iloc[:development_months]['Net Income'].sum())
        
        # Calculate breakeven point consistently
        dev_phase_costs = df_pnl.iloc[:development_months]
        operational_phase = df_pnl.iloc[development_months:]
        
        # Calculate total investment needed to recover
        cumulative_investment = abs(dev_phase_costs['Net Income'].sum())
        
        
        
        # Find breakeven point
        breakeven_month = None
        running_total = cumulative_investment
        
        for i, row in operational_phase.iterrows():
                       
            if running_total <= 0:  # We've recovered our investment
                breakeven_month = i + 1
                break
        
        # Display metrics
        if breakeven_month:
            months_to_profit = breakeven_month  # This is already relative to operational phase
            
            # Calculate actual breakeven date
            current_date = pd.Timestamp.now()
            breakeven_date = current_date + pd.DateOffset(months=breakeven_month + development_months)
            
            st.metric(
                "Months to Breakeven",
                f"{months_to_profit} months after launch",
                help="Time until cumulative profit becomes positive (after development)"
            )
            st.metric(
                "Breakeven Date",
                breakeven_date.strftime('%B %Y'),
                help=f"Estimated breakeven in {breakeven_month + development_months} months total (including {development_months} months development)"
            )

        st.info("""
        **Understanding the Investment Metrics:**
        - Total Investment Required: Maximum funding needed before becoming cash flow positive
        - Development Phase Cost: Total spend during initial product development
        - Months to Breakeven: Time until cumulative revenue exceeds cumulative costs
        - Breakeven Date: Estimated calendar date when business becomes profitable
        """)
        
        # Move investment analysis to new tab
        investment_tab1, investment_tab2 = st.tabs(["Cumulative Metrics", "Monthly Cash Flow"])
        
        with investment_tab1:
            st.subheader("Investment Requirements & Breakeven")
            
            # Show cumulative P&L chart
            st.line_chart(df_pnl[[
                'Cumulative Revenue',
                'Cumulative Costs',
                'Cumulative Profit/Loss'
            ]])
            
            # Add data table for cumulative metrics with revenue breakdown
            st.subheader("Cumulative Metrics Data")
            cumulative_data = df_pnl[[
                'Cumulative Revenue',
                'Cumulative Costs',
                'Cumulative Profit/Loss',
                'Enterprise Revenue',          # Add enterprise revenue
                'Subscription Revenue',        # Add individual revenue streams
                'Pay-per-Video Revenue',
                'Total Revenue'
            ]].round(2)
            
            # Rename columns for clarity
            cumulative_data.columns = [
                'Cumulative Revenue',
                'Cumulative Costs',
                'Cumulative Profit/Loss',
                'Enterprise Revenue',
                'Subscription Revenue',
                'Pay-per-Video Revenue',
                'Total Monthly Revenue'
            ]
            
            # Add month numbers as index
            cumulative_data.index = [f"Month {i+1}" for i in range(len(cumulative_data))]
            st.dataframe(cumulative_data.style.format("${:,.2f}"))
            
            # Calculate key investment metrics
            total_investment = df_pnl['Investment Required'].max()
            
            # Find breakeven month
            breakeven_month = None
            for i, row in df_pnl.iterrows():
                if row['Cumulative Profit/Loss'] > 0 and i >= development_months:
                    breakeven_month = i + 1
                    break
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Total Investment Required",
                    f"${total_investment:,.0f}",
                    help="Maximum cumulative negative cash flow (peak funding needed)"
                )
                
                dev_phase_cost = abs(df_pnl.iloc[:development_months]['Net Income'].sum())
                st.metric(
                    "Development Phase Cost",
                    f"${dev_phase_cost:,.0f}",
                    help="Total cost during initial development phase"
                )
            
            with col2:
                if breakeven_month:
                    months_to_profit = breakeven_month - development_months  # Months after launch
                    
                    # Calculate actual breakeven date
                    current_date = pd.Timestamp.now()
                    breakeven_date = current_date + pd.DateOffset(months=breakeven_month)
                    
                    st.metric(
                        "Months to Breakeven",
                        f"{months_to_profit} months after launch",
                        help="Time until cumulative profit becomes positive (after development)"
                    )
                    st.metric(
                        "Breakeven Date",
                        breakeven_date.strftime('%B %Y'),  # Format as 'Month Year'
                        help=f"Estimated breakeven in {breakeven_month} months (including {development_months} months development)"
                    )
                else:
                    st.metric(
                        "Months to Breakeven",
                        "Not reached",
                        help="Business does not reach breakeven in projection period"
                    )
        
        with investment_tab2:
            st.subheader("Cash Flow Analysis")
            # Show monthly cash flows
            st.line_chart(df_pnl[[
                'Net Income',
                'Operating Profit',
                'Gross Profit'
            ]])
            
            # Add data table for monthly cash flows
            st.subheader("Monthly Cash Flow Data")
            cash_flow_data = df_pnl[[
                'Net Income',
                'Operating Profit',
                'Gross Profit'
            ]].round(2)
            # Add month numbers as index
            cash_flow_data.index = [f"Month {i+1}" for i in range(len(cash_flow_data))]
            st.dataframe(cash_flow_data.style.format("${:,.2f}"))
            
            # Show cumulative profit metrics chart
            st.subheader("Cumulative Profit Metrics")
            st.line_chart(df_pnl[[
                'Cumulative Revenue',
                'Cumulative Costs',
                'Cumulative Profit/Loss'
            ]])
            
            # Add data table for cumulative profit metrics
            st.subheader("Cumulative Profit Data")
            cumulative_profit_data = df_pnl[[
                'Cumulative Revenue',
                'Cumulative Costs',
                'Cumulative Profit/Loss'
            ]].round(2)
            # Add month numbers as index
            cumulative_profit_data.index = [f"Month {i+1}" for i in range(len(cumulative_profit_data))]
            st.dataframe(cumulative_profit_data.style.format("${:,.2f}"))

    with tab6:
        st.header("Sensitivity Analysis Configuration")
        
        # Add controls for sensitivity analysis
        st.subheader("Select Parameters to Analyze")
        
        # Parameter selection
        available_params = {
            "Development Months": {
                "param": "development_months",
                "current": development_months,
                "min": 3.0,
                "max": 12.0,
                "is_integer": True,
                "step": 1.0
            },
            "Monthly Growth Rate": {
                "param": "monthly_growth_rate",
                "current": monthly_growth_rate,
                "min": 0.02,
                "max": 0.15,
                "is_integer": False,
                "step": 0.01
            },
            "Monthly Churn Rate": {
                "param": "churn_rate",
                "current": churn_rate,
                "min": 0.01,
                "max": 0.10,
                "is_integer": False,
                "step": 0.01
            },
            "Starting Coaches": {
                "param": "starting_coaches",
                "current": starting_coaches,
                "min": 20.0,
                "max": 500.0,
                "is_integer": True,
                "step": 1.0
            },
            "Basic Plan Price": {
                "param": "basic_plan_price",
                "current": basic_plan_price,
                "min": 150.0,
                "max": 500.0,
                "is_integer": False,
                "step": 10.0
            },
            "Pay-per-Video Price": {
                "param": "pay_per_video_price",
                "current": pay_per_video_price,
                "min": 50.0,
                "max": 130.0,
                "is_integer": False,
                "step": 5.0
            }
        }

        # After all UI elements are defined, add the Monthly Team & Operations parameter
        if 'founder_salary' in locals():  # Check if UI elements are defined
            available_params["Monthly Team & Operations"] = {
                "param": "team_and_ops_monthly",
                "current": (
                    # Team costs
                    founder_salary +
                    monthly_contractor_budget +
                    # Office & Operations
                    office_rent_monthly +
                    utilities_monthly +
                    software_licenses_monthly +
                    insurance_monthly +
                    legal_accounting_monthly +
                    # Benefits & Equipment
                    wfh_stipend_monthly +
                    travel_budget_monthly
                ),
                "min": 5000.0,
                "max": 50000.0,
                "is_integer": True,
                "step": 1000.0
            }

        # Add enterprise parameters
        available_params.update({
            "Enterprise Growth Rate": {
                "param": "enterprise_monthly_growth",
                "current": enterprise_growth if enable_enterprise else 0.02,
                "min": 0.01,
                "max": 0.10,
                "is_integer": False,
                "step": 0.01
            },
            "Small Academy Price": {
                "param": "small_academy_price",
                "current": small_academy_price if enable_enterprise else 1000,
                "min": 500.0,
                "max": 2000.0,
                "is_integer": False,
                "step": 100.0
            },
            "Medium Academy Price": {
                "param": "medium_academy_price",
                "current": medium_academy_price if enable_enterprise else 2500,
                "min": 1500.0,
                "max": 4000.0,
                "is_integer": False,
                "step": 250.0
            },
            "Large Federation Price": {
                "param": "large_federation_price",
                "current": large_federation_price if enable_enterprise else 5000,
                "min": 3000.0,
                "max": 8000.0,
                "is_integer": False,
                "step": 500.0
            }
        })

        # Let user select parameters to analyze
        selected_params = st.multiselect(
            "Select parameters to analyze (max 4 recommended)",
            list(available_params.keys()),
            default=list(available_params.keys())[:3]
        )

        if selected_params:
            st.subheader("Configure Parameter Ranges")
            
            # Create parameter ranges configuration
            param_configs = {}
            col1, col2 = st.columns(2)
            
            for i, param_name in enumerate(selected_params):
                with col1 if i % 2 == 0 else col2:
                    st.write(f"**{param_name}**")
                    param_info = available_params[param_name]
                    
                    # Get min, max, and steps for each parameter
                    min_val = st.number_input(
                        f"Min {param_name}",
                        value=float(param_info["min"]),  # Convert to float
                        step=param_info["step"],
                        format="%.2f" if not param_info["is_integer"] else "%.0f"
                    )
                    max_val = st.number_input(
                        f"Max {param_name}",
                        value=float(param_info["max"]),  # Convert to float
                        step=param_info["step"],
                        format="%.2f" if not param_info["is_integer"] else "%.0f"
                    )
                    steps = st.slider(
                        f"Steps for {param_name}",
                        min_value=3,
                        max_value=10,
                        value=5
                    )
                    
                    param_configs[param_name] = {
                        "param": param_info["param"],
                        "min": min_val,
                        "max": max_val,
                        "steps": steps,
                        "is_integer": param_info["is_integer"]
                    }

            # Add this before the button
            if "sensitivity_results" not in st.session_state:
                st.session_state.sensitivity_results = None

            # Modify the button section
            if st.button("Run Sensitivity Analysis"):
                with st.spinner("Running sensitivity analysis..."):
                    # Create parameter ranges
                    param_ranges = {}
                    for param_name, config in param_configs.items():
                        variations = np.linspace(config["min"], config["max"], config["steps"])
                        if config["is_integer"]:
                            variations = variations.astype(int)
                        param_ranges[config["param"]] = variations

                    # Get base parameters from current state
                    base_params = {
                        "development_months": development_months,
                        "monthly_growth_rate": monthly_growth_rate,
                        "churn_rate": churn_rate,
                        "starting_coaches": starting_coaches,
                        "basic_plan_price": basic_plan_price,
                        "pay_per_video_price": pay_per_video_price,
                        # Add all the team and ops parameters
                        "founder_salary": founder_salary,
                        "monthly_contractor_budget": monthly_contractor_budget,
                        "office_rent_monthly": office_rent_monthly,
                        "utilities_monthly": utilities_monthly,
                        "software_licenses_monthly": software_licenses_monthly,
                        "insurance_monthly": insurance_monthly,
                        "legal_accounting_monthly": legal_accounting_monthly,
                        "wfh_stipend_monthly": wfh_stipend_monthly,
                        "travel_budget_monthly": travel_budget_monthly
                    }

                    # Run sensitivity analysis and store in session state
                    st.session_state.sensitivity_results = run_sensitivity_analysis(base_params, param_ranges)

            # Move results visualization outside the button condition
            if st.session_state.sensitivity_results is not None:
                results_df = st.session_state.sensitivity_results
                
                # Display results
                st.subheader("Sensitivity Analysis Results")
                 # Parameter impact visualization
                metric_to_analyze = st.selectbox(
                    "Select Metric to Analyze",
                    ["Total Investment", "Months to Breakeven", "Peak Monthly Revenue", 
                     "Year 1 Revenue", "Final Monthly Revenue", "Max Monthly Profit"]
                )

                # Create tornado chart
                st.subheader("Tornado Chart Analysis")

                # Calculate parameter impacts
                tornado_data = []
                baseline_values = {}

                # Calculate baseline values (using middle point for each parameter)
                for param_name, param_data in results_df.groupby('Parameter'):
                    param_values = sorted(param_data['Parameter Value'].unique())
                    baseline_idx = len(param_values) // 2
                    baseline_value = param_data[param_data['Parameter Value'] == param_values[baseline_idx]][metric_to_analyze].iloc[0]
                    baseline_values[param_name] = baseline_value
                    
                    # Calculate min and max impacts
                    min_impact = param_data[param_data['Parameter Value'] == param_values[0]][metric_to_analyze].iloc[0] - baseline_value
                    max_impact = param_data[param_data['Parameter Value'] == param_values[-1]][metric_to_analyze].iloc[0] - baseline_value
                    
                    tornado_data.append({
                        'Parameter': param_name,
                        'Min Impact': min_impact,
                        'Max Impact': max_impact,
                        'Absolute Impact': max(abs(min_impact), abs(max_impact))
                })

                # Convert to DataFrame and sort by impact
                tornado_df = pd.DataFrame(tornado_data)
                tornado_df = tornado_df.sort_values('Absolute Impact', ascending=True)

                # Create tornado chart
                fig_tornado = go.Figure()

                # Add bars for each parameter
                for idx, row in tornado_df.iterrows():
                    # Add bar for negative impact
                    fig_tornado.add_trace(go.Bar(
                        y=[row['Parameter']],
                        x=[row['Min Impact']],
                        orientation='h',
                        name='Negative Impact',
                        marker_color='rgba(219, 64, 82, 0.7)',
                        showlegend=idx == 0
                    ))
                    
                    # Add bar for positive impact
                    fig_tornado.add_trace(go.Bar(
                        y=[row['Parameter']],
                        x=[row['Max Impact']],
                        orientation='h',
                        name='Positive Impact',
                        marker_color='rgba(55, 128, 191, 0.7)',
                        showlegend=idx == 0
                    ))

                # Update layout
                fig_tornado.update_layout(
                    title=f"Parameter Impact on {metric_to_analyze}",
                    barmode='overlay',
                    height=400,
                    yaxis={'categoryorder': 'array', 'categoryarray': tornado_df['Parameter'].tolist()},
                    xaxis_title=f"Change in {metric_to_analyze}",
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig_tornado, use_container_width=True)
               
                # Create parallel coordinates plot
                fig = go.Figure(data=
                    go.Parcoords(
                        line=dict(
                            color=results_df[metric_to_analyze],
                            colorscale='Viridis',
                        ),
                        dimensions=[
                            dict(
                                range=[results_df[metric_to_analyze].min(), results_df[metric_to_analyze].max()],
                                      label=metric_to_analyze,
                                      values=results_df[metric_to_analyze]
                            ),
                            *[
                                dict(
                                    range=[results_df[results_df['Parameter'] == param]['Parameter Value'].min(),
                                          results_df[results_df['Parameter'] == param]['Parameter Value'].max()],
                                    label=param,
                                    values=results_df[results_df['Parameter'] == param]['Parameter Value']
                                )
                                for param in results_df['Parameter'].unique()
                            ]
                        ]
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                

                # Show summary statistics
                st.subheader("Summary Statistics")
                summary_stats = results_df.groupby('Parameter')[metric_to_analyze].agg(['min', 'mean', 'max'])
                st.dataframe(summary_stats)
                
                # Add detailed results table
                st.subheader("Detailed Sensitivity Analysis Results")
                # Pivot the results to show all metrics for each parameter value
                detailed_results = results_df.pivot_table(
                    index=['Parameter', 'Parameter Value'],
                    values=[
                        'Total Investment',
                        'Months to Breakeven',
                        'Peak Monthly Revenue',
                        'Year 1 Revenue',
                        'Final Monthly Revenue',
                        'Max Monthly Profit'
                    ],
                    aggfunc='first'
                ).round(2)
                
                # Format the table
                st.dataframe(
                    detailed_results.style.format({
                        'Total Investment': '${:,.2f}',
                        'Peak Monthly Revenue': '${:,.2f}',
                        'Year 1 Revenue': '${:,.2f}',
                        'Final Monthly Revenue': '${:,.2f}',
                        'Max Monthly Profit': '${:,.2f}',
                        'Months to Breakeven': '{:.1f}'
                    })
                )

        else:
            st.warning("Please select at least one parameter to analyze")

if __name__ == "__main__":
    main() 