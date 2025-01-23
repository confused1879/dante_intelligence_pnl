import streamlit as st
import pandas as pd
import numpy as np

def generate_pnl_data(
    # Market & Customer Assumptions
    months=24,
    starting_users=100,
    monthly_growth_rate=0.05,
    churn_rate=0.05,
    target_market_size=1000000,
    subscription_fee=25.0,
    
    # Consulting & Partnership Revenue
    consulting_monthly_engagements=2,
    consulting_fee=5000,
    partnership_deals=1,
    partnership_fee=50000,
    
    # Additional Revenue Streams
    model_licensing_fee=10000,
    affiliate_revenue_per_user=1.0,
    
    # COGS
    cogs_hosting_per_user=2.0,
    cogs_data_per_user=1.0,
    cogs_support_per_user=0.5,
    support_engineer_monthly=2500,
    
    # Operating Expenses - R&D
    ai_engineer_count=3,
    ai_engineer_salary=120000,
    data_scientist_count=2,
    data_scientist_salary=100000,
    model_training_monthly=5000,
    
    # Operating Expenses - Sales & Marketing
    sales_team_count=2,
    sales_salary=80000,
    marketing_percent_revenue=0.15,
    event_sponsorship_monthly=5000,
    
    # Operating Expenses - G&A
    executive_salary_monthly=25000,
    office_rent_monthly=3000,
    insurance_monthly=2000,
    professional_fees_monthly=1000,
    
    # Financial Assumptions
    tax_rate=0.25,
    interest_rate=0.05,
    depreciation_years=5
):
    """
    Generate a monthly P&L DataFrame based on comprehensive assumptions.
    """

    # Initialize arrays/lists to store month-by-month data
    months_list = list(range(1, months + 1))
    user_count = []
    subscription_revenue = []
    consulting_revenue = []
    partnership_revenue = []
    total_revenue = []
    cogs_hosting = []
    cogs_data = []
    cogs_support = []
    gross_profit = []
    operating_expenses = []
    ebit = []  # Earnings Before Interest & Taxes
    net_income = []

    # Add new arrays for the additional metrics
    model_licensing_revenue = []
    affiliate_revenue = []
    total_rd_expenses = []
    total_sales_marketing = []
    total_ga_expenses = []
    depreciation = []

    # Starting values
    current_users = starting_users

    for m in months_list:
        # 1. Calculate current month user count
        #    Growth is approximate, factoring in churn net effect
        #    Another approach: new_users = current_users * monthly_growth_rate
        #                     lost_users = current_users * churn_rate
        #                     current_users = current_users + new_users - lost_users
        current_users = current_users * (1 + monthly_growth_rate - churn_rate)
        user_count.append(current_users)

        # 2. Subscription Revenue
        sub_revenue = current_users * subscription_fee
        subscription_revenue.append(sub_revenue)

        # 3. Consulting Revenue (assume monthly engagements * fee)
        #    You might also ramp this up over time.
        consult_revenue = consulting_monthly_engagements * consulting_fee
        consulting_revenue.append(consult_revenue)

        # 4. Partnership Revenue
        #    For simplicity, assume partnership fees are amortized monthly
        #    partnership_deals * (partnership_fee / 12)
        #    or you could add lumps in certain months
        partner_rev = partnership_deals * (partnership_fee / 12)
        partnership_revenue.append(partner_rev)

        # Additional Revenue Streams
        licensing_rev = model_licensing_fee / 12  # Assuming annual fee spread monthly
        affiliate_rev = current_users * affiliate_revenue_per_user
        model_licensing_revenue.append(licensing_rev)
        affiliate_revenue.append(affiliate_rev)
        
        # Update total revenue calculation
        rev_total = (sub_revenue + consult_revenue + partner_rev + 
                    licensing_rev + affiliate_rev)
        
        # R&D Expenses
        monthly_rd = (
            (ai_engineer_count * ai_engineer_salary / 12) +
            (data_scientist_count * data_scientist_salary / 12) +
            model_training_monthly
        )
        total_rd_expenses.append(monthly_rd)
        
        # Sales & Marketing Expenses
        monthly_sales_marketing = (
            (sales_team_count * sales_salary / 12) +
            (rev_total * marketing_percent_revenue) +
            event_sponsorship_monthly
        )
        total_sales_marketing.append(monthly_sales_marketing)
        
        # G&A Expenses
        monthly_ga = (
            executive_salary_monthly +
            office_rent_monthly +
            insurance_monthly +
            professional_fees_monthly
        )
        total_ga_expenses.append(monthly_ga)
        
        # Update operating expenses calculation
        op_exp = monthly_rd + monthly_sales_marketing + monthly_ga
        
        # 5. Cost of Goods Sold
        #    E.g., hosting cost is cogs_hosting_per_user * user_count
        #    data licensing cost is cogs_data_per_user * user_count
        hosting_cost = current_users * cogs_hosting_per_user
        data_cost = current_users * cogs_data_per_user

        # For support, you could scale with user count or keep it fixed monthly
        support_cost = support_engineer_monthly

        cogs_hosting.append(hosting_cost)
        cogs_data.append(data_cost)
        cogs_support.append(support_cost)

        total_cogs = hosting_cost + data_cost + support_cost

        # 6. Gross Profit
        g_profit = rev_total - total_cogs
        gross_profit.append(g_profit)

        # 7. Operating Expenses
        #    Summation of R&D, Sales & Marketing, G&A
        op_exp = op_exp
        operating_expenses.append(op_exp)

        # 8. EBIT
        earnings_before_tax = g_profit - op_exp
        ebit.append(earnings_before_tax)

        # 9. Taxes (assuming profitable months only)
        #    if earnings_before_tax > 0 then tax_rate applies
        if earnings_before_tax > 0:
            tax_payment = earnings_before_tax * tax_rate
        else:
            tax_payment = 0

        net_in = earnings_before_tax - tax_payment
        net_income.append(net_in)

        # Update total revenue calculation
        total_revenue.append(rev_total)

    # Create a DataFrame
    data = {
        'Month': months_list,
        'User Count': user_count,
        'Market Penetration (%)': [u/target_market_size*100 for u in user_count],
        
        # Revenue Metrics
        'Subscription Revenue': subscription_revenue,
        'Consulting Revenue': consulting_revenue,
        'Partnership Revenue': partnership_revenue,
        'Model Licensing Revenue': model_licensing_revenue,
        'Affiliate Revenue': affiliate_revenue,
        'Total Revenue': total_revenue,
        
        # COGS
        'COGS - Hosting': cogs_hosting,
        'COGS - Data': cogs_data,
        'COGS - Support': cogs_support,
        'Gross Profit': gross_profit,
        
        # Operating Expenses
        'R&D Expenses': total_rd_expenses,
        'Sales & Marketing': total_sales_marketing,
        'G&A Expenses': total_ga_expenses,
        'Total Operating Expenses': operating_expenses,
        
        # Bottom Line
        'EBIT': ebit,
        'Net Income': net_income
    }

    df_pnl = pd.DataFrame(data)
    return df_pnl


def main():
    st.title("AI-Driven Tennis Startup - P&L Model")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Market & Customer Assumptions",
        "Revenue Assumptions",
        "Cost Assumptions",
        "Financial Results"
    ])
    
    with tab1:
        st.header("Market & Customer Assumptions")
        months = st.slider("Projection Months", 12, 60, 24)
        starting_users = st.number_input("Starting Users", 0, 10000, 100)
        monthly_growth_rate = st.slider("Monthly Growth Rate (%)", 0.0, 0.5, 0.05)
        churn_rate = st.slider("Monthly Churn Rate (%)", 0.0, 0.2, 0.05)
        target_market_size = st.number_input("Target Market Size", 1000, 10000000, 1000000)
        
    with tab2:
        st.header("Revenue Assumptions")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Subscription")
            subscription_fee = st.number_input("Monthly Subscription Fee ($)", 0, 1000, 25)
            
            st.subheader("Consulting")
            consulting_monthly_engagements = st.number_input("Monthly Consulting Engagements", 0, 100, 2)
            consulting_fee = st.number_input("Consulting Fee per Engagement ($)", 0, 50000, 5000)
        
        with col2:
            st.subheader("Partnerships")
            partnership_deals = st.number_input("Active Partnerships", 0, 50, 1)
            partnership_fee = st.number_input("Annual Partnership Fee ($)", 0, 1000000, 50000)
            
            st.subheader("Additional Revenue")
            model_licensing_fee = st.number_input("Annual Model Licensing Fee ($)", 0, 100000, 10000)
            affiliate_revenue_per_user = st.number_input("Monthly Affiliate Revenue per User ($)", 0.0, 100.0, 1.0)
    
    with tab3:
        st.header("Cost Assumptions")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("COGS")
            cogs_hosting_per_user = st.number_input("Hosting Cost per User ($)", 0.0, 100.0, 2.0)
            cogs_data_per_user = st.number_input("Data Cost per User ($)", 0.0, 100.0, 1.0)
            cogs_support_per_user = st.number_input("Support Cost per User ($)", 0.0, 100.0, 0.5)
            
            st.subheader("R&D")
            ai_engineer_count = st.number_input("Number of AI Engineers", 0, 50, 3)
            ai_engineer_salary = st.number_input("AI Engineer Annual Salary ($)", 0, 200000, 120000)
            data_scientist_count = st.number_input("Number of Data Scientists", 0, 50, 2)
            data_scientist_salary = st.number_input("Data Scientist Annual Salary ($)", 0, 200000, 100000)
        
        with col2:
            st.subheader("Sales & Marketing")
            sales_team_count = st.number_input("Number of Sales Staff", 0, 50, 2)
            sales_salary = st.number_input("Sales Staff Annual Salary ($)", 0, 150000, 80000)
            marketing_percent_revenue = st.slider("Marketing % of Revenue", 0.0, 0.5, 0.15)
            
            st.subheader("G&A")
            executive_salary_monthly = st.number_input("Monthly Executive Salary ($)", 0, 100000, 25000)
            office_rent_monthly = st.number_input("Monthly Office Rent ($)", 0, 20000, 3000)
    
    with tab4:
        # Generate the DataFrame with all parameters
        df_pnl = generate_pnl_data(
            months=months,
            starting_users=starting_users,
            monthly_growth_rate=monthly_growth_rate,
            churn_rate=churn_rate,
            target_market_size=target_market_size,
            subscription_fee=subscription_fee,
            consulting_monthly_engagements=consulting_monthly_engagements,
            consulting_fee=consulting_fee,
            partnership_deals=partnership_deals,
            partnership_fee=partnership_fee,
            model_licensing_fee=model_licensing_fee,
            affiliate_revenue_per_user=affiliate_revenue_per_user,
            cogs_hosting_per_user=cogs_hosting_per_user,
            cogs_data_per_user=cogs_data_per_user,
            cogs_support_per_user=cogs_support_per_user,
            ai_engineer_count=ai_engineer_count,
            ai_engineer_salary=ai_engineer_salary,
            data_scientist_count=data_scientist_count,
            data_scientist_salary=data_scientist_salary,
            sales_team_count=sales_team_count,
            sales_salary=sales_salary,
            marketing_percent_revenue=marketing_percent_revenue,
            executive_salary_monthly=executive_salary_monthly,
            office_rent_monthly=office_rent_monthly
        )

        st.subheader("P&L Table")
        st.dataframe(df_pnl.style.format("{:,.2f}"))

        # Key Metrics Charts
        st.subheader("Key Metrics Over Time")
        metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs(["Revenue", "Costs", "Profitability"])
        
        with metrics_tab1:
            st.line_chart(df_pnl[['Subscription Revenue', 'Consulting Revenue', 
                                 'Partnership Revenue', 'Model Licensing Revenue', 
                                 'Total Revenue']])
        
        with metrics_tab2:
            st.line_chart(df_pnl[['COGS - Hosting', 'COGS - Data', 'COGS - Support',
                                 'R&D Expenses', 'Sales & Marketing', 'G&A Expenses']])
        
        with metrics_tab3:
            st.line_chart(df_pnl[['Gross Profit', 'EBIT', 'Net Income']])

if __name__ == "__main__":
    main()
