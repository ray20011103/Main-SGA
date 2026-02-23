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
df_merged = pd.merge(df_sga, df_capex[['Company_ID', 'Year', 'PPE_Capex', 'Intan_Capex', 'Total_Liabilities']], 
                     on=['Company_ID', 'Year'], how='inner')

df_merged = pd.merge(df_merged, df_industry[['Company_ID', 'Year', 'Net_Income', 'Current_Earnings_Growth', 'Total_Assets']], 
                     on=['Company_ID', 'Year'], how='left')

df_merged = pd.merge(df_merged, df_mv_year, on=['Company_ID', 'Year'], how='left')

# 4. 變數標準化 (Scaling by Assets)
df_merged['Scaled_RD'] = df_merged['研究發展費'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_PPE_Capex'] = df_merged['PPE_Capex'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_Intan_Capex'] = df_merged['Intan_Capex'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_Earnings'] = df_merged['Net_Income'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_Intan_Capex'] = df_merged['Scaled_Intan_Capex'].fillna(0)

# 5. 建構依變數: Future Earnings Volatility (t to t+3)
# 計算滾動 4 年的標準差 (包含當年)
# 需要先確保資料依年份排序
df_merged = df_merged.sort_values(by=['Company_ID', 'Year'])

# 使用 rolling window 計算標準差
# rolling(4) 會包含當前行與前3行。但論文定義是 t 到 t+3。
# 技巧：使用 shift 把未來的資料移到當前行，或者使用反向 rolling (pandas 比較難直接做)。
# 替代方法：Shift t+1, t+2, t+3 到當前行，然後橫向計算 std。

df_merged['Earnings_t0'] = df_merged['Scaled_Earnings']
df_merged['Earnings_t1'] = df_merged.groupby('Company_ID')['Scaled_Earnings'].shift(-1)
df_merged['Earnings_t2'] = df_merged.groupby('Company_ID')['Scaled_Earnings'].shift(-2)
df_merged['Earnings_t3'] = df_merged.groupby('Company_ID')['Scaled_Earnings'].shift(-3)

# 計算橫向標準差 (需至少 3 年有資料才計算，避免誤差太大)
earnings_cols = ['Earnings_t0', 'Earnings_t1', 'Earnings_t2', 'Earnings_t3']
df_merged['Future_Earnings_Vol'] = df_merged[earnings_cols].std(axis=1, ddof=1) # ddof=1 for sample std

# 6. 建構控制變數
df_merged['Size'] = np.log(df_merged['MV'])
df_merged['Leverage'] = df_merged['Total_Liabilities'] / df_merged['Total_Assets']
df_merged['Loss_Dummy'] = (df_merged['Net_Income'] < 0).astype(int)
df_merged['Current_Earnings_Growth'] = df_merged['Current_Earnings_Growth'].fillna(0)

# 7. 準備迴歸資料
cols_to_use = ['Future_Earnings_Vol', 'Scaled_RD', 'Investment_MainSGA', 
               'Scaled_PPE_Capex', 'Scaled_Intan_Capex', 
               'Size', 'Current_Earnings_Growth', 'Loss_Dummy', 'Leverage', 
               'Company_ID', 'Year', 'Industry_Group']

df_reg = df_merged.dropna(subset=cols_to_use).copy()

# === Step A: Winsorization (1% & 99%) ===
print("正在進行 Winsorization (1%, 99%):..")
vars_to_winsorize = ['Future_Earnings_Vol', 'Scaled_RD', 'Investment_MainSGA', 
                     'Scaled_PPE_Capex', 'Scaled_Intan_Capex', 
                     'Size', 'Current_Earnings_Growth', 'Leverage']

for col in vars_to_winsorize:
    df_reg[col] = winsorize(df_reg[col], limits=[0.01, 0.01])

# === Step B: Industry-Year Standardization (Z-Score) ===
print("正在進行 Industry-Year Standardization...")
invest_vars = ['Scaled_RD', 'Investment_MainSGA', 'Scaled_PPE_Capex', 'Scaled_Intan_Capex']

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

# Model A1: Future Earnings Volatility (Raw Values)
print("\n=== Equation 6 (A1): Future Earnings Volatility (Raw Values) ===")
formula_a1 = 'Future_Earnings_Vol ~ Scaled_RD + Investment_MainSGA + Scaled_PPE_Capex + Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage'
model_a1 = smf.ols(formula_a1, data=df_reg)
res_a1 = model_a1.fit(cov_type='cluster', cov_kwds={'groups': groups})
print(res_a1.summary())

# Model A2: Future Earnings Volatility (Standardized Values)
print("\n=== Equation 6 (A2): Future Earnings Volatility (Standardized Inv) ===")
formula_a2 = 'Future_Earnings_Vol ~ Std_Scaled_RD + Std_Investment_MainSGA + Std_Scaled_PPE_Capex + Std_Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage'
model_a2 = smf.ols(formula_a2, data=df_reg)
res_a2 = model_a2.fit(cov_type='cluster', cov_kwds={'groups': groups})
print(res_a2.summary())

# 整理係數比較 (Table 8 Style)
print("\n" + "="*80)
print("【Table 8 Replication】: Uncertainty of Future Benefits")
print("Dependent Variable: Future Earnings Volatility (Std Dev t~t+3)")
print("="*80)
print(f"{ '':<20} | { 'Raw Coef':<15} | { 'Std Coef':<15} | { 't-stat (Std)':<10}")
print("-" * 80)

rows = [
    ('R&D', 'Scaled_RD', 'Std_Scaled_RD'),
    ('MainSG&A Inv', 'Investment_MainSGA', 'Std_Investment_MainSGA'),
    ('PPE Capex', 'Scaled_PPE_Capex', 'Std_Scaled_PPE_Capex'),
    ('Intan Capex', 'Scaled_Intan_Capex', 'Std_Scaled_Intan_Capex'),
    ('Leverage', 'Leverage', 'Leverage')
]

for label, raw_var, std_var in rows:
    # Raw Model
    raw_coef = res_a1.params[raw_var]
    
    # Std Model
    var_a2 = std_var if std_var in res_a2.params else raw_var
    std_coef = res_a2.params[var_a2]
    t_stat = res_a2.tvalues[var_a2]
    
    # Sig
    sig = "***" if res_a2.pvalues[var_a2] < 0.01 else "**" if res_a2.pvalues[var_a2] < 0.05 else "*" if res_a2.pvalues[var_a2] < 0.1 else ""

    print(f"{label:<20} | {raw_coef:>8.4f}        | {std_coef:>8.4f}{sig:<3} | {t_stat:>6.2f}")

print("-" * 80)
print("Interpretation: Positive coef = Higher Risk (More Volatility)")
print("Hypothesis: R&D > MainSG&A > Capex")
