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

print("Loading Data for Comparison...")
df_main_oc = pd.read_csv(os.path.join(data_processed_dir, 'org_capital_results.csv'))
df_trad_oc = pd.read_csv(os.path.join(data_processed_dir, 'traditional_oc_results.csv'))
df_ret = pd.read_csv(os.path.join(data_raw_dir, 'industry&return.csv'))

# Preprocess Returns
df_ret['Company_ID'] = df_ret['證券代碼'].astype(str).str.split().str[0]
df_ret['Date'] = pd.to_datetime(df_ret['年月'], format='%Y%m')
df_ret['Year'] = df_ret['Date'].dt.year
df_ret['Month'] = df_ret['Date'].dt.month
df_ret['Return'] = pd.to_numeric(df_ret['報酬率_月'], errors='coerce') * 100
df_ret['Market_Cap'] = pd.to_numeric(df_ret['市值(百萬元)'], errors='coerce')
df_ret = df_ret.dropna(subset=['Return', 'Market_Cap'])

# Fix ID types
df_main_oc['Company_ID'] = df_main_oc['Company_ID'].astype(str)
df_trad_oc['Company_ID'] = df_trad_oc['Company_ID'].astype(str)

def run_double_sort(df_oc_data, oc_col_name):
    # Align Data (May Rebalancing)
    rebal_month = 5
    df_ret_temp = df_ret.copy()
    df_ret_temp['Fin_Year'] = np.where(
        df_ret_temp['Month'] >= rebal_month, 
        df_ret_temp['Year'] - 1, 
        df_ret_temp['Year'] - 2
    )
    
    # Merge
    merged = pd.merge(
        df_ret_temp, df_oc_data[['Company_ID', 'Year', oc_col_name, 'Industry_Group']],
        left_on=['Company_ID', 'Fin_Year'], right_on=['Company_ID', 'Year'],
        how='inner'
    )
    merged = merged.dropna(subset=[oc_col_name, 'Industry_Group'])
    
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
            
    merged['OC_Rank'] = merged.groupby(['Date', 'Industry_Group'])[oc_col_name].transform(assign_q)
    
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

print("\nRunning Comparison (Small Firm High-Low Strategy)...")

# 1. Main SG&A OC
mean_main, t_main = run_double_sort(df_main_oc, 'OC_Intensity')

# 2. Traditional OC
mean_trad, t_trad = run_double_sort(df_trad_oc, 'Traditional_OC_Intensity')

print("\n" + "="*60)
print("【Comparison: Main SG&A vs. Traditional Total SG&A】")
print("Target: Small Firm High-Low Strategy (Monthly Return %)")
print("="*60)
print(f"{ 'Metric':<20} | { 'Main SG&A (New)':<15} | { 'Total SG&A (Trad)':<15}")
print("-" * 60)
print(f"{ 'Mean Return':<20} | {mean_main:>10.3f}%    | {mean_trad:>10.3f}%")
print(f"{ 't-statistic':<20} | {t_main:>10.3f}     | {t_trad:>10.3f}")
print("-" * 60)

if t_main > t_trad:
    print(">> Conclusion: Main SG&A OC provides a stronger/more significant signal.")
else:
    print(">> Conclusion: Traditional OC performs similarly or better.")
print("="*60)

# Save result for LaTeX
res_df = pd.DataFrame({
    'Metric': ['Mean Return (%)', 't-statistic'],
    'Main SG&A (New)': [f"{mean_main:.3f}", f"{t_main:.3f}"],
    'Total SG&A (Trad)': [f"{mean_trad:.3f}", f"{t_trad:.3f}"]
})
res_df.to_csv(os.path.join(data_processed_dir, 'OC_Comparison.csv'), index=False)
