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
):
    """
    Generate P&L including initial development phase
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
        
        # Calculate profits
        gross_profit[m] = month_total_revenue - month_cogs
        operating_profit[m] = gross_profit[m] - (
            personnel_costs[m] +     # Staff & contractors
            operating_costs[m] +     # Office, equipment, etc.
            month_marketing  # Now using scaled marketing budget
        )
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
        'Cumulative Costs': np.cumsum([x + y + z for x, y, z in zip(operating_costs, personnel_costs, infrastructure_costs)]),
        'Cumulative Profit/Loss': np.cumsum(net_income),
        
        # Running cash position (negative shows required investment)
        'Cash Position': np.cumsum(net_income),
        'Investment Required': [-min(0, x) for x in np.cumsum(net_income)],  # Shows only negative cash position
        
        # Other cumulative metrics...
    })
    
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
        'Months to Breakeven': next((i + 1 for i, val in enumerate(df['Cumulative Profit/Loss']) if val > 0), len(df)),
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
        
        with col2:
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
        
        with col3:
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
            marketing_budget_monthly=marketing_budget_monthly
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
            st.line_chart(df_pnl[[
                'Subscription Revenue',
                'Pay-per-Video Revenue', 
                'Total Revenue'
            ]])
            
            # Add contractor utilization metrics
            st.subheader("Contractor Utilization")
            st.line_chart(df_pnl[[
                'Contractor Hours',
                'Contractor Costs'
            ]])
        
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
        # Show burn rates
        st.subheader("Burn Rate Analysis")
        dev_burn_rate = abs(df_pnl.iloc[:development_months]['Net Income'].mean())
        operational_burn_rate = abs(df_pnl[df_pnl['Net Income'] < 0]['Net Income'].mean())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Development Phase Burn Rate",
                f"${dev_burn_rate:,.0f}/month",
                help="Average monthly cash burn during development"
            )
        with col2:
            st.metric(
                "Operational Burn Rate",
                f"${operational_burn_rate:,.0f}/month",
                help="Average monthly cash burn until breakeven"
            )
        
        # Move investment analysis to new tab
        investment_tab1, investment_tab2 = st.tabs(["Cumulative Metrics", "Monthly Cash Flow"])
        
        with investment_tab1:
            st.subheader("Investment Requirements & Breakeven")
            
            # Show cumulative P&L
            st.line_chart(df_pnl[[
                'Cumulative Revenue',
                'Cumulative Costs',
                'Cumulative Profit/Loss'
            ]])
            
            # Calculate key investment metrics
            total_investment = df_pnl['Investment Required'].max()
            
            # Find breakeven month
            breakeven_month = None
            for i, row in df_pnl.iterrows():
                if row['Cumulative Profit/Loss'] > 0:
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
                
                # Add development cost breakdown
                dev_phase_cost = abs(df_pnl.iloc[:development_months]['Net Income'].sum())
                st.metric(
                    "Development Phase Cost",
                    f"${dev_phase_cost:,.0f}",
                    help="Total cost during initial development phase"
                )
            
            with col2:
                if breakeven_month:
                    months_to_profit = breakeven_month
                    st.metric(
                        "Months to Breakeven",
                        f"{months_to_profit} months",
                        help="Time until cumulative profit becomes positive"
                    )
                    st.metric(
                        "Breakeven Date",
                        f"{pd.Timestamp.now().strftime('%Y-%m')[:7]} + {months_to_profit} months",
                        help="Approximate calendar date of breakeven"
                    )
                else:
                    st.metric(
                        "Months to Breakeven",
                        "Not reached",
                        help="Business does not reach breakeven in projection period"
                    )
            
            st.info("""
            **Understanding the Investment Metrics:**
            - Total Investment Required: Maximum funding needed before becoming cash flow positive
            - Development Phase Cost: Total spend during initial product development
            - Months to Breakeven: Time until cumulative revenue exceeds cumulative costs
            - Breakeven Date: Estimated calendar date when business becomes profitable
            """)
        
        with investment_tab2:
            st.subheader("Cash Flow Analysis")
            # Show monthly cash flows
            st.line_chart(df_pnl[[
                'Net Income',
                'Operating Profit',
                'Gross Profit'
            ]])
            
            # Add cumulative profit metrics
            st.subheader("Cumulative Profit Metrics")
            st.line_chart(df_pnl[[
                'Cumulative Revenue',
                'Cumulative Costs',
                'Cumulative Profit/Loss'
            ]])
            
            

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

        else:
            st.warning("Please select at least one parameter to analyze")

if __name__ == "__main__":
    main() 