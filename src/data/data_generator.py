import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from scipy.stats import norm, beta

class DataGenerator:
    def __init__(self, n_customers=25000, start_date='2023-01-01', months=12):
        self.n_customers = n_customers
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.months = months
        self.customers = []
        
    def generate_customer_ids(self):
        return [f'CUST_{i:05d}' for i in range(1, self.n_customers + 1)]
    
    def generate_demographics(self):
        ages = np.random.normal(35, 10, self.n_customers).astype(int)
        genders = np.random.choice(['M', 'F'], self.n_customers, p=[0.52, 0.48])
        locations = np.random.choice(['Urban', 'Suburban', 'Rural'], 
                                   self.n_customers, 
                                   p=[0.5, 0.3, 0.2])
        return pd.DataFrame({
            'customer_id': self.generate_customer_ids(),
            'age': ages,
            'gender': genders,
            'location': locations
        })
    
    def generate_financial_metrics(self,demographics:pd.DataFrame):
        location_base_income = {
            'Urban': 120_000,   
            'Suburban':  90_000,
            'Rural':     60_000
        }
        gender_multiplier = {
            'M': 1.10, 
            'F': 0.90
        }
        loc_mean = demographics['location'].map(location_base_income).to_numpy()
        gen_mult = demographics['gender'].map(gender_multiplier).to_numpy()
        
        sigma_inc = 0.3
        mu_inc = np.log(loc_mean * gen_mult)
        
        monthly_incomes = np.random.lognormal(mean=mu_inc, sigma=sigma_inc, size=self.n_customers)
        
        # - EMI outflow = 20–35% of income, plus some noise
        emi_ratio = np.random.uniform(0.20, 0.35, size=self.n_customers)
        monthly_emis = monthly_incomes * emi_ratio
        
         #    - Total credit limit = 2×–4× monthly_income
        limit_mult = np.random.uniform(2.0, 4.0, size=self.n_customers)
        total_credit_limit = monthly_incomes * limit_mult
        
        #    - Current outstanding ~ 30–80% of limit
        util_ratio = np.random.beta(2, 5, size=self.n_customers) 
        current_outstandings = total_credit_limit * util_ratio
        credit_utilization = current_outstandings/total_credit_limit
        
        return pd.DataFrame({
            'monthly_income':   monthly_incomes,
            'monthly_emi_outflow': monthly_emis,
            'total_credit_limit':  total_credit_limit,
            'current_outstanding': current_outstandings,
            'credit_utilization_ratio': credit_utilization
        })
    
    def generate_credit_history(self,financials:pd.DataFrame):
        util = financials['credit_utilization_ratio'].to_numpy()
        dpds = np.random.poisson(lam=2, size=self.n_customers)
        lam_msd = 12 / (1 + dpds)                 
        months_since_default = np.random.poisson(lam=lam_msd.clip(min=1e-2))
        lam_inq = 0.5 + util * 5                
        num_hard_inquiries = np.random.poisson(lam=lam_inq)
        months_since_default = np.random.poisson(lam=12, size=self.n_customers)
        
        dpd_norm = np.clip(dpds/ 10, 0, 1)             # ≥10 days
        msd_norm = np.clip(months_since_default / 24, 0, 1)  # ≥24mo
        util_norm = np.clip(util, 0, 1)         
         
        w_dpd = 0.4
        w_msd = 0.3
        w_util = 0.3

        goodness = (w_dpd * (1 - dpd_norm) +
                    w_msd * msd_norm +
                    w_util * (1 - util_norm))

        noise = np.random.normal(0, 5, size=self.n_customers)
        repayment_score = np.clip(goodness * 100 + noise, 0, 100)
        
        return pd.DataFrame({
            'repayment_history_score': repayment_score,
            'dpd_last_3_months': dpds,
            'num_hard_inquiries_last_6m': num_hard_inquiries,
            'months_since_last_default': months_since_default
        })
    
    def generate_loan_estimate(self,financials:pd.DataFrame):
        
        util = financials['credit_utilization_ratio'].to_numpy()     
        base_rate = 1.0  

        λ = base_rate + util * 3.0  

        num_open_loans = np.random.poisson(lam=λ, size= self.n_customers)
                
        return pd.DataFrame({
            'num_open_loans': num_open_loans
        })
    
    def determine_credit_movement(self, row):
        # Risk factors
        risk_score = 0
        
        # High risk indicators
        if row['dpd_last_3_months'] > 3:
            risk_score += 1
        if row['credit_utilization_ratio'] > 0.8:
            risk_score += 1
        if row['num_hard_inquiries_last_6m'] > 2:
            risk_score += 1
        if row['months_since_last_default'] < 6:
            risk_score += 1
        
        cc_usage_ratio = row['recent_credit_card_usage'] / row['total_credit_limit']
        if cc_usage_ratio > 0.3:
            risk_score += 1
            
        if row['recent_loan_disbursed_amount'] > 0.5 * row['monthly_income']:
            risk_score += 1
            
        # Positive indicators
        if row['repayment_history_score'] > 80:
            risk_score -= 1
        if row['monthly_income'] / row['monthly_emi_outflow'] > 2:
            risk_score -= 1
        if row['num_open_loans'] > 3 and row['repayment_history_score'] > 90:
            risk_score -= 1 
        # Determine movement
        if risk_score >= 2:
            return 'decrease'
        elif risk_score <= -2:
            return 'increase'
        else:
            return 'stable'
    
    def generate_dataset(self):
        demographics = self.generate_demographics()
        financial = self.generate_financial_metrics(demographics)
        credit_history = self.generate_credit_history(financial)
        loan_estimation = self.generate_loan_estimate(financial)
        
        # Combine all features
        df = pd.concat([demographics, financial, credit_history,loan_estimation], axis=1)
        df['recent_credit_card_usage'] = df['total_credit_limit'] * df['credit_utilization_ratio']* np.random.uniform(0.4, 0.8, size=self.n_customers)
        df['recent_loan_disbursed_amount'] = df['current_outstanding'] * np.random.uniform(0.2, 0.6, size=self.n_customers)
        # Add target variable
        df['target_credit_score_movement'] = df.apply(self.determine_credit_movement, axis=1)

        
        return df

def main():
    generator = DataGenerator()
    df = generator.generate_dataset()
    
    # Save the dataset
    df.to_csv('data/raw/credit_score_dataset.csv', index=False)
    print(f"Dataset generated with {len(df)} rows")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(df.describe())
    print("\nTarget Distribution:")
    print(df['target_credit_score_movement'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
