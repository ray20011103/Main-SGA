import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings("ignore")

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
data_raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
data_processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

print("Loading Data...")
# Load necessary data
df_sga = pd.read_csv(os.path.join(data_processed_dir, 'main_sga_v2_result.csv'))
df_cpi = pd.read_csv(os.path.join(data_raw_dir, 'CPI.csv'))
df_ret = pd.read_csv(os.path.join(data_raw_dir, 'industry&return.csv'))

# Preprocess CPI
df_cpi['Year'] = df_cpi['年月'].astype(str).str[:4].astype(int)
annual_cpi = df_cpi.groupby('Year')['數值'].mean().reset_index()
annual_cpi.columns = ['Year', 'CPI']
annual_cpi['CPI_Deflator'] = annual_cpi['CPI'] / 100.0

# Preprocess SGA (Investment)
df_sga = pd.merge(df_sga, annual_cpi[['Year', 'CPI_Deflator']], on='Year', how='left')
df_sga['CPI_Deflator'] = df_sga['CPI_Deflator'].fillna(1.0)
df_sga['Nominal_Investment'] = df_sga['Investment_MainSGA'] * df_sga['Avg_Total_Assets']
df_sga['Real_Investment'] = df_sga['Nominal_Investment'] / df_sga['CPI_Deflator']

# Preprocess Returns
df_ret['Company_ID'] = df_ret['證券代碼'].astype(str).str.split().str[0]
df_ret['Date'] = pd.to_datetime(df_ret['年月'], format='%Y%m')
df_ret['Year'] = df_ret['Date'].dt.year
df_ret['Month'] = df_ret['Date'].dt.month
df_ret['Return'] = pd.to_numeric(df_ret['報酬率_月'], errors='coerce') * 100
df_ret['Market_Cap'] = pd.to_numeric(df_ret['市值(百萬元)'], errors='coerce')
df_ret = df_ret.dropna(subset=['Return', 'Market_Cap'])

# ==========================================
# Helper Function: Calculate PIM with specific delta
# ==========================================
def calculate_oc_stock(delta_val, g_val=0.2964):
    def pim_logic(group):
        group = group.sort_values('Year')
        oc_stocks = []
        if len(group) == 0: return group
        
        # Init
        first_flow = group['Real_Investment'].iloc[0]
        current_stock = first_flow / (g_val + delta_val)
        oc_stocks.append(current_stock)
        
        # Loop
        for i in range(1, len(group)):
            flow = group['Real_Investment'].iloc[i]
            current_stock = current_stock * (1 - delta_val) + flow
            oc_stocks.append(current_stock)
        
        group['Real_Stock'] = oc_stocks
        return group

    # Apply PIM
    df_res = df_sga.groupby('Company_ID', group_keys=False).apply(pim_logic)
    
    # Scale by Assets (Nominal/Nominal)
    df_res['Nominal_Stock'] = df_res['Real_Stock'] * df_res['CPI_Deflator']
    df_res['OC_Intensity'] = df_res['Nominal_Stock'] / df_res['Avg_Total_Assets']
    
    return df_res[['Company_ID', 'Year', 'OC_Intensity', 'Industry_Group']].copy()

# ==========================================
# Helper Function: Calculate Small Firm Premium
# ==========================================
def get_small_firm_premium(df_oc_data, rebal_month):
    # Align Data
    # If Month >= rebal_month, use Year-1; else Year-2
    df_ret_temp = df_ret.copy()
    df_ret_temp['Fin_Year'] = np.where(
        df_ret_temp['Month'] >= rebal_month, 
        df_ret_temp['Year'] - 1, 
        df_ret_temp['Year'] - 2
    )
    
    df_oc_data['Company_ID'] = df_oc_data['Company_ID'].astype(str)
    
    # Merge
    merged = pd.merge(
        df_ret_temp, df_oc_data,
        left_on=['Company_ID', 'Fin_Year'], right_on=['Company_ID', 'Year'],
        how='inner'
    )
    merged = merged.dropna(subset=['OC_Intensity', 'TEJ產業_代碼'])
    
    # Double Sort
    # 1. Size Split (Median)
    merged['Size_Group'] = merged.groupby('Date')['Market_Cap'].transform(
        lambda x: pd.qcut(x, 2, labels=['Small', 'Big'])
    )
    
    # 2. OC Quintile (Within Industry)
    def assign_q(x):
        try: return pd.qcut(x, 5, labels=[1,2,3,4,5])
        except: 
            try: return pd.qcut(x, 3, labels=[2,3,4])
            except: return np.nan
            
    merged['OC_Rank'] = merged.groupby(['Date', 'TEJ產業_代碼'])['OC_Intensity'].transform(assign_q)
    
    # Calc Returns
    port_ret = merged.groupby(['Date', 'Size_Group', 'OC_Rank'])[['Return', 'Market_Cap']].apply(
        lambda x: np.average(x['Return'], weights=x['Market_Cap'])
    ).reset_index(name='Ret')
    
    pivot = port_ret.pivot_table(index='Date', columns=['Size_Group', 'OC_Rank'], values='Ret')
    
    # Small Firm High-Low
    try:
        hl_series = pivot[('Small', 5)] - pivot[('Small', 1)]
        mean = hl_series.mean()
        t_stat = sm.OLS(hl_series, sm.add_constant(np.ones(len(hl_series)))).fit(cov_type='HAC', cov_kwds={'maxlags': 6}).tvalues[0]
        return mean, t_stat
    except:
        return np.nan, np.nan

# ==========================================
# Task A: Table 10 (Rebalancing Timing)
# ==========================================
print("\n=== Generating Table 10 (Timing Sensitivity) ===")
# Use default delta = 0.15
df_oc_base = calculate_oc_stock(0.15)

timing_results = []
for m in [4, 5, 6]:
    mean, t = get_small_firm_premium(df_oc_base, m)
    timing_results.append({
        'Rebalancing Month': f'{m} (End)', 
        'Mean Return (%)': mean, 
        't-statistic': t
    })
    print(f"Month {m}: Mean={mean:.3f}, t={t:.3f}")

pd.DataFrame(timing_results).to_csv(os.path.join(data_processed_dir, 'Table10_Timing.csv'), index=False)

# ==========================================
# Task B: Table 11 (Depreciation Rate)
# ==========================================
print("\n=== Generating Table 11 (Depreciation Sensitivity) ===")
# Fix month = 5
base_month = 5

delta_results = []
for d in [0.10, 0.15, 0.20, 0.30]:
    df_oc_d = calculate_oc_stock(d)
    mean, t = get_small_firm_premium(df_oc_d, base_month)
    delta_results.append({
        'Depreciation Rate': f'{d:.2f}', 
        'Mean Return (%)': mean, 
        't-statistic': t
    })
    print(f"Delta {d}: Mean={mean:.3f}, t={t:.3f}")

pd.DataFrame(delta_results).to_csv(os.path.join(data_processed_dir, 'Table11_Delta.csv'), index=False)
print("\nRobustness Tables Generated.")
