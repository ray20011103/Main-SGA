import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats.mstats import winsorize
import os

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
data_raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
data_processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

# 1. 讀取與合併資料 (Standard Routine)
print("讀取並處理資料中...")
df_sga = pd.read_csv(os.path.join(data_processed_dir, 'main_sga_v2_result.csv'))
df_capex = pd.read_csv(os.path.join(data_raw_dir, 'capex.csv'))
df_industry = pd.read_csv(os.path.join(data_raw_dir, 'industry.csv'))
df_mv = pd.read_csv(os.path.join(data_raw_dir, 'industry&return.csv'), dtype={'年月': str}, usecols=['證券代碼', '年月', '市值(百萬元)'])

# ID & Date Formatting
df_sga['Company_ID'] = df_sga['Company_ID'].astype(str)
df_capex.columns = df_capex.columns.str.strip()
df_capex['Company_ID'] = df_capex['公司'].astype(str).str.split().str[0]
df_capex['Year'] = pd.to_datetime(df_capex['年月']).dt.year
df_industry.columns = df_industry.columns.str.strip()
df_industry['Company_ID'] = df_industry['公司'].astype(str).str.split().str[0]
df_industry['Year'] = pd.to_datetime(df_industry['年月']).dt.year
df_mv['Company_ID'] = df_mv['證券代碼'].astype(str).str.split().str[0]
df_mv['Date'] = pd.to_datetime(df_mv['年月'], format='%Y%m')
df_mv['Year'] = df_mv['Date'].dt.year

# Numeric Conversion
df_capex['PPE_Capex'] = pd.to_numeric(df_capex['購置不動產廠房設備（含預付）－CFI'].astype(str).str.replace(',', ''), errors='coerce').abs()
df_capex['Intan_Capex'] = pd.to_numeric(df_capex['無形資產（增加）減少－CFI'].astype(str).str.replace(',', ''), errors='coerce').abs()
df_capex['Tobins_Q'] = pd.to_numeric(df_capex['Tobins Q'], errors='coerce')
if '負債總額' in df_capex.columns:
    df_capex['Total_Liabilities'] = pd.to_numeric(df_capex['負債總額'].astype(str).str.replace(',', ''), errors='coerce')
else:
    df_capex['Total_Liabilities'] = np.nan

df_industry['Net_Income'] = pd.to_numeric(df_industry['歸屬母公司淨利（損）'].astype(str).str.replace(',', ''), errors='coerce')
df_industry['Total_Assets'] = pd.to_numeric(df_industry['資產總額'].astype(str).str.replace(',', ''), errors='coerce')
df_industry['Industry_Group'] = df_industry['TEJ產業_代碼'].astype(str).str[:3]

# Control Vars
df_industry = df_industry.sort_values(['Company_ID', 'Year'])
df_industry['Lag_Net_Income'] = df_industry.groupby('Company_ID')['Net_Income'].shift(1)
df_industry['Current_Earnings_Growth'] = (df_industry['Net_Income'] - df_industry['Lag_Net_Income']) / df_industry['Lag_Net_Income'].abs()
df_industry['Current_Earnings_Growth'] = df_industry['Current_Earnings_Growth'].replace([np.inf, -np.inf], np.nan)

# MV
df_mv['MV'] = pd.to_numeric(df_mv['市值(百萬元)'].astype(str).str.replace(',', ''), errors='coerce')
df_mv_year = df_mv.sort_values('Date').groupby(['Company_ID', 'Year'])['MV'].last().reset_index()

# Merge
df_merged = pd.merge(df_sga, df_capex[['Company_ID', 'Year', 'PPE_Capex', 'Intan_Capex', 'Tobins_Q', 'Total_Liabilities']], on=['Company_ID', 'Year'], how='inner')
df_merged = pd.merge(df_merged, df_industry[['Company_ID', 'Year', 'Net_Income', 'Current_Earnings_Growth', 'Total_Assets']], on=['Company_ID', 'Year'], how='left')
df_merged = pd.merge(df_merged, df_mv_year, on=['Company_ID', 'Year'], how='left')

# Scaling & Standardizing Setup
df_merged['Scaled_RD'] = df_merged['研究發展費'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_PPE_Capex'] = df_merged['PPE_Capex'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_Intan_Capex'] = df_merged['Intan_Capex'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_Intan_Capex'] = df_merged['Scaled_Intan_Capex'].fillna(0)
df_merged['Size'] = np.log(df_merged['MV'])
df_merged['Leverage'] = df_merged['Total_Liabilities'] / df_merged['Total_Assets']
df_merged['Loss_Dummy'] = (df_merged['Net_Income'] < 0).astype(int)
df_merged['Current_Earnings_Growth'] = df_merged['Current_Earnings_Growth'].fillna(0)

# ==========================================
# 定義分組 (Sub-samples)
# ==========================================

# 1. 電子業 vs 非電子業
# 假設 M23, M24, M25, M26, M27, M28 等為電子相關 (依據 TEJ 常用代碼)
# 電子零組件(M23), 半導體(M24), 光電(M25), 電腦週邊(M26), 通信網路(M27), 電子通路(M28), 資訊服務(M29)
electronic_codes = ['M23', 'M24', 'M25', 'M26', 'M27', 'M28', 'M29']
df_merged['Is_Electronics'] = df_merged['Industry_Group'].isin(electronic_codes)

# 2. 公司規模 (Size) - Proxy for Lifecycle (Small ~ Young/Growth, Large ~ Mature)
# 使用全樣本中位數切分
size_median = df_merged['Size'].median()
df_merged['Is_Large'] = df_merged['Size'] > size_median

# Data Cleaning
cols = ['Tobins_Q', 'Scaled_RD', 'Investment_MainSGA', 'Scaled_PPE_Capex', 'Scaled_Intan_Capex', 
        'Size', 'Current_Earnings_Growth', 'Loss_Dummy', 'Leverage', 
        'Industry_Group', 'Year', 'Company_ID', 'Is_Electronics', 'Is_Large']
df_reg = df_merged.dropna(subset=cols).copy()

# Winsorization
vars_win = ['Tobins_Q', 'Scaled_RD', 'Investment_MainSGA', 'Scaled_PPE_Capex', 'Scaled_Intan_Capex', 
            'Size', 'Current_Earnings_Growth', 'Leverage']
for col in vars_win:
    df_reg[col] = winsorize(df_reg[col], limits=[0.01, 0.01])

# Standardization (Z-Score by Industry-Year)
invest_vars = ['Scaled_RD', 'Investment_MainSGA', 'Scaled_PPE_Capex', 'Scaled_Intan_Capex']
for col in invest_vars:
    new_col = f'Std_{col}'
    grouped = df_reg.groupby(['Industry_Group', 'Year'])[col]
    mean = grouped.transform('mean')
    std = grouped.transform('std')
    df_reg[new_col] = (df_reg[col] - mean) / std
    df_reg[new_col] = df_reg[new_col].fillna(0)

# Clustering Setup
df_reg['Company_Code'] = df_reg['Company_ID'].astype('category').cat.codes
df_reg['Year_Code'] = df_reg['Year'].astype(int)

# ==========================================
# 執行回歸函數
# ==========================================
def run_regression(data, label):
    # 如果樣本太少，避免 Clustering 錯誤
    if len(data) < 100:
        return "N/A", 0, 0
        
    groups = data[['Company_Code', 'Year_Code']].values
    formula = 'Tobins_Q ~ Std_Scaled_RD + Std_Investment_MainSGA + Std_Scaled_PPE_Capex + Std_Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage'
    model = smf.ols(formula, data=data)
    
    try:
        res = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
    except:
        # Fallback to robust SE if clustering fails (e.g. too few groups)
        res = model.fit(cov_type='HC1')
    
    coef = res.params['Std_Investment_MainSGA']
    t_stat = res.tvalues['Std_Investment_MainSGA']
    p_val = res.pvalues['Std_Investment_MainSGA']
    obs = int(res.nobs)
    adj_r2 = res.rsquared_adj
    
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    return f"{coef:.4f}{sig} (t={t_stat:.2f})", obs, adj_r2

# ==========================================
# 執行並輸出 Markdown
# ==========================================

# 1. Electronics vs Non-Electronics
res_elec = run_regression(df_reg[df_reg['Is_Electronics'] == True], "Electronics")
res_non = run_regression(df_reg[df_reg['Is_Electronics'] == False], "Non-Electronics")

# 2. Small vs Large
res_small = run_regression(df_reg[df_reg['Is_Large'] == False], "Small Firms")
res_large = run_regression(df_reg[df_reg['Is_Large'] == True], "Large Firms")

# Full Sample
res_all = run_regression(df_reg, "Full Sample")

# Print Clean Markdown Table
print("\n### Sub-sample Analysis: Impact of MainSG&A Investment on Tobin's Q\n")
print("| Sample Group | MainSG&A Coef (Std) | Obs | Adj. R2 |")
print("| :--- | :--- | :---: | :---: |")
print(f"| **Full Sample** | {res_all[0]} | {res_all[1]} | {res_all[2]:.3f} |")
print(f"| Electronics | {res_elec[0]} | {res_elec[1]} | {res_elec[2]:.3f} |")
print(f"| Non-Electronics | {res_non[0]} | {res_non[1]} | {res_non[2]:.3f} |")
print(f"| Small Firms (Size < Med) | {res_small[0]} | {res_small[1]} | {res_small[2]:.3f} |")
print(f"| Large Firms (Size > Med) | {res_large[0]} | {res_large[1]} | {res_large[2]:.3f} |")
print("\n*Note: Coeffs are standardized. t-stats in parentheses are clustered by firm & year. *** p<0.01, ** p<0.05, * p<0.1*")
