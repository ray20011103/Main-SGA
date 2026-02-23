import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats.mstats import winsorize
import os

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
data_raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
data_processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

# 1. 讀取資料
print("讀取資料中...")
df_sga = pd.read_csv(os.path.join(data_processed_dir, 'main_sga_v2_result.csv'))
df_capex = pd.read_csv(os.path.join(data_raw_dir, 'capex.csv'))
df_industry = pd.read_csv(os.path.join(data_raw_dir, 'industry.csv'))

# 讀取市值資料 (用於計算 Size)
df_mv = pd.read_csv(os.path.join(data_raw_dir, 'industry&return.csv'), 
                    dtype={'年月': str},
                    usecols=['證券代碼', '年月', '市值(百萬元)'])

# 2. 資料清洗與合併準備
df_sga['Company_ID'] = df_sga['Company_ID'].astype(str)

df_capex.columns = df_capex.columns.str.strip()
df_capex['Company_ID'] = df_capex['公司'].astype(str).str.split().str[0]
df_capex['Year'] = pd.to_datetime(df_capex['年月']).dt.year

# 處理數值欄位
df_capex['PPE_Capex'] = pd.to_numeric(df_capex['購置不動產廠房設備（含預付）－CFI'].astype(str).str.replace(',', ''), errors='coerce').abs()
df_capex['Intan_Capex'] = pd.to_numeric(df_capex['無形資產（增加）減少－CFI'].astype(str).str.replace(',', ''), errors='coerce').abs()
df_capex['Tobins_Q'] = pd.to_numeric(df_capex['Tobins Q'], errors='coerce')
# 處理負債總額
if '負債總額' in df_capex.columns:
    df_capex['Total_Liabilities'] = pd.to_numeric(df_capex['負債總額'].astype(str).str.replace(',', ''), errors='coerce')
else:
    df_capex['Total_Liabilities'] = np.nan

# 準備控制變數與產業代碼
df_industry.columns = df_industry.columns.str.strip()
df_industry['Company_ID'] = df_industry['公司'].astype(str).str.split().str[0]
df_industry['Year'] = pd.to_datetime(df_industry['年月']).dt.year
df_industry['Net_Income'] = pd.to_numeric(df_industry['歸屬母公司淨利（損）'].astype(str).str.replace(',', ''), errors='coerce')
df_industry['Total_Assets'] = pd.to_numeric(df_industry['資產總額'].astype(str).str.replace(',', ''), errors='coerce')
# 處理產業代碼
df_industry['Industry_Group'] = df_industry['TEJ產業_代碼'].astype(str).str[:3]

df_industry = df_industry.sort_values(['Company_ID', 'Year'])
df_industry['Lag_Net_Income'] = df_industry.groupby('Company_ID')['Net_Income'].shift(1)
df_industry['Current_Earnings_Growth'] = (df_industry['Net_Income'] - df_industry['Lag_Net_Income']) / df_industry['Lag_Net_Income'].abs()
df_industry['Current_Earnings_Growth'] = df_industry['Current_Earnings_Growth'].replace([np.inf, -np.inf], np.nan)

# 準備市值資料
df_mv['Company_ID'] = df_mv['證券代碼'].astype(str).str.split().str[0]
df_mv['Date'] = pd.to_datetime(df_mv['年月'], format='%Y%m')
df_mv['Year'] = df_mv['Date'].dt.year
df_mv['MV'] = pd.to_numeric(df_mv['市值(百萬元)'].astype(str).str.replace(',', ''), errors='coerce')
# 取每家公司每年的平均市值
df_mv_year = df_mv.sort_values('Date').groupby(['Company_ID', 'Year'])['MV'].last().reset_index()

# 3. 合併所有資料
# 先合併 Capex (含負債總額)
df_merged = pd.merge(df_sga, df_capex[['Company_ID', 'Year', 'PPE_Capex', 'Intan_Capex', 'Tobins_Q', 'Total_Liabilities']], 
                     on=['Company_ID', 'Year'], how='inner')

# 合併 Industry (獲利成長, 資產總額, 產業代碼)
# Removed Industry_Group from here as it is already in df_sga
df_merged = pd.merge(df_merged, df_industry[['Company_ID', 'Year', 'Net_Income', 'Current_Earnings_Growth', 'Total_Assets']], 
                     on=['Company_ID', 'Year'], how='left')

# 合併市值 (MV)
df_merged = pd.merge(df_merged, df_mv_year, on=['Company_ID', 'Year'], how='left')

# 4. 變數標準化 (Scaling by Assets)
df_merged['Scaled_RD'] = df_merged['研究發展費'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_PPE_Capex'] = df_merged['PPE_Capex'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_Intan_Capex'] = df_merged['Intan_Capex'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_Intan_Capex'] = df_merged['Scaled_Intan_Capex'].fillna(0)

# 5. 建構控制變數
df_merged['Size'] = np.log(df_merged['MV'])
df_merged['Leverage'] = df_merged['Total_Liabilities'] / df_merged['Total_Assets']
df_merged['Loss_Dummy'] = (df_merged['Net_Income'] < 0).astype(int)
df_merged['Current_Earnings_Growth'] = df_merged['Current_Earnings_Growth'].fillna(0)

# 6. 準備迴歸資料 (包含 Maintenance_MainSGA)
cols_to_use = ['Tobins_Q', 'Scaled_RD', 'Investment_MainSGA', 'Maintenance_MainSGA',
               'Scaled_PPE_Capex', 'Scaled_Intan_Capex', 
               'Size', 'Current_Earnings_Growth', 'Loss_Dummy', 'Leverage', 
               'Company_ID', 'Year', 'Industry_Group']

df_reg = df_merged.dropna(subset=cols_to_use).copy()

# === Step A: Winsorization (1% & 99%) on RAW values ===
print("正在進行 Winsorization (1%, 99% от...")
vars_to_winsorize = ['Tobins_Q', 'Scaled_RD', 'Investment_MainSGA', 'Maintenance_MainSGA',
                     'Scaled_PPE_Capex', 'Scaled_Intan_Capex', 
                     'Size', 'Current_Earnings_Growth', 'Leverage']

for col in vars_to_winsorize:
    df_reg[col] = winsorize(df_reg[col], limits=[0.01, 0.01])

# === Step B: Industry-Year Standardization (Z-Score) ===
print("正在進行 Industry-Year Standardization...")
# 我們也要標準化 Maintenance_MainSGA 以便公平比較
invest_vars = ['Scaled_RD', 'Investment_MainSGA', 'Maintenance_MainSGA', 'Scaled_PPE_Capex', 'Scaled_Intan_Capex']

for col in invest_vars:
    new_col = f'Std_{col}'
    grouped = df_reg.groupby(['Industry_Group', 'Year'])[col]
    mean = grouped.transform('mean')
    std = grouped.transform('std')
    df_reg[new_col] = (df_reg[col] - mean) / std
    df_reg[new_col] = df_reg[new_col].fillna(0)

# Company & Year Code for Clustering
df_reg['Company_Code'] = df_reg['Company_ID'].astype('category').cat.codes
df_reg['Year_Code'] = df_reg['Year'].astype(int)
groups = df_reg[['Company_Code', 'Year_Code']].values

print(f"\n分析樣本數: {len(df_reg)}")

# Model 1: Investment Component
print("\n=== Model 1: Investment Component (Tobin's Q) ===")
formula_inv = 'Tobins_Q ~ Std_Scaled_RD + Std_Investment_MainSGA + Std_Scaled_PPE_Capex + Std_Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage'
model_inv = smf.ols(formula_inv, data=df_reg)
res_inv = model_inv.fit(cov_type='cluster', cov_kwds={'groups': groups})
print(res_inv.summary())

# Model 2: Maintenance Component
print("\n=== Model 2: Maintenance Component (Tobin's Q) ===")
formula_maint = 'Tobins_Q ~ Std_Scaled_RD + Std_Maintenance_MainSGA + Std_Scaled_PPE_Capex + Std_Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage'
model_maint = smf.ols(formula_maint, data=df_reg)
res_maint = model_maint.fit(cov_type='cluster', cov_kwds={'groups': groups})
print(res_maint.summary())

# 整理係數比較 (Validity Test)
print("\n" + "="*80)
print("【Validity Test】: Investment vs. Maintenance Component")
print("Dependent Variable: Tobin's Q")
print("="*80)
print(f"{ '':<25} | { 'Investment Model':<20} | { 'Maintenance Model':<20}")
print("-" * 80)

# Extract coefficients and stats
# Investment Model
inv_coef = res_inv.params['Std_Investment_MainSGA']
inv_t = res_inv.tvalues['Std_Investment_MainSGA']
inv_p = res_inv.pvalues['Std_Investment_MainSGA']

# Maintenance Model
maint_coef = res_maint.params['Std_Maintenance_MainSGA']
maint_t = res_maint.tvalues['Std_Maintenance_MainSGA']
maint_p = res_maint.pvalues['Std_Maintenance_MainSGA']

def get_sig(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

print(f"{'Main Coefficient':<25} | {inv_coef:>8.4f}{get_sig(inv_p):<3} (t={inv_t:.2f}) | {maint_coef:>8.4f}{get_sig(maint_p):<3} (t={maint_t:.2f})")
print("-" * 80)
print("Hypothesis: Investment Coef > 0 (Significant), Maintenance Coef ~ 0 (Insignificant)")

