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
# 新增: 處理負債總額
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
# 取每家公司每年的平均市值或年底市值
df_mv_year = df_mv.sort_values('Date').groupby(['Company_ID', 'Year'])['MV'].last().reset_index()

# 3. 合併所有資料
# 先合併 Capex (含負債總額)
df_merged = pd.merge(df_sga, df_capex[['Company_ID', 'Year', 'PPE_Capex', 'Intan_Capex', 'Tobins_Q', 'Total_Liabilities']], 
                     on=['Company_ID', 'Year'], how='inner')

# 合併 Industry (獲利成長, 資產總額等) - Industry_Group 已在 df_sga 中，無需重複合併
df_merged = pd.merge(df_merged, df_industry[['Company_ID', 'Year', 'Net_Income', 'Current_Earnings_Growth', 'Total_Assets']], 
                     on=['Company_ID', 'Year'], how='left')
print("df_merged columns after merge:", df_merged.columns)

# 合併市值 (MV)
df_merged = pd.merge(df_merged, df_mv_year, on=['Company_ID', 'Year'], how='left')

# 4. 變數標準化 (Scaling by Assets)
df_merged['Scaled_RD'] = df_merged['研究發展費'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_PPE_Capex'] = df_merged['PPE_Capex'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_Intan_Capex'] = df_merged['Intan_Capex'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_Earnings'] = df_merged['Net_Income'] / df_merged['Avg_Total_Assets']
df_merged['Scaled_Intan_Capex'] = df_merged['Scaled_Intan_Capex'].fillna(0)

# 5. 建構控制變數
df_merged['Size'] = np.log(df_merged['MV'])
df_merged['Leverage'] = df_merged['Total_Liabilities'] / df_merged['Total_Assets']
df_merged['Loss_Dummy'] = (df_merged['Net_Income'] < 0).astype(int)
df_merged['Current_Earnings_Growth'] = df_merged['Current_Earnings_Growth'].fillna(0)

# 6. 建構應變數: Future Earnings Change
df_merged = df_merged.sort_values(by=['Company_ID', 'Year'])
for i in [1, 2, 3]:
    df_merged[f'Earnings_t{i}'] = df_merged.groupby('Company_ID')['Scaled_Earnings'].shift(-i)

df_merged['Avg_Future_Earnings'] = df_merged[[f'Earnings_t{i}' for i in [1, 2, 3]]].mean(axis=1)
df_merged['Change_In_Earnings'] = df_merged['Avg_Future_Earnings'] - df_merged['Scaled_Earnings']

# 7. 準備迴歸資料
cols_to_use = ['Change_In_Earnings', 'Tobins_Q', 'Scaled_RD', 'Investment_MainSGA', 
               'Scaled_PPE_Capex', 'Scaled_Intan_Capex', 
               'Size', 'Current_Earnings_Growth', 'Loss_Dummy', 'Leverage', 'Company_ID', 'Year', 'Industry_Group']

df_reg = df_merged.dropna(subset=cols_to_use).copy()

# === Step A: Winsorization (1% & 99%) on RAW values ===
print("正在進行 Winsorization (1%, 99%)...")
vars_to_winsorize = ['Change_In_Earnings', 'Tobins_Q', 'Scaled_RD', 'Investment_MainSGA', 
                     'Scaled_PPE_Capex', 'Scaled_Intan_Capex', 
                     'Size', 'Current_Earnings_Growth', 'Leverage']

for col in vars_to_winsorize:
    df_reg[col] = winsorize(df_reg[col], limits=[0.01, 0.01])

# === Step B: Industry-Year Standardization (Z-Score) ===
print("正在進行 Industry-Year Standardization...")
# 定義需要標準化的投資變數
invest_vars = ['Scaled_RD', 'Investment_MainSGA', 'Scaled_PPE_Capex', 'Scaled_Intan_Capex']

for col in invest_vars:
    new_col = f'Std_{col}'
    # 分組計算 Z-Score: (x - mean) / std
    # 使用 transform 以保持原 index
    grouped = df_reg.groupby(['Industry_Group', 'Year'])[col]
    mean = grouped.transform('mean')
    std = grouped.transform('std')
    
    df_reg[new_col] = (df_reg[col] - mean) / std
    
    # 處理 std 為 0 或 NaN 的情況 (若該產業該年只有一家公司或數值都一樣)
    df_reg[new_col] = df_reg[new_col].fillna(0)

# 檢查一下標準化後的統計量
print("\n標準化變數統計量 (前5筆):")
print(df_reg[['Std_Scaled_RD', 'Std_Investment_MainSGA', 'Std_Scaled_PPE_Capex']].describe())

# === 關鍵修正：將 Company_ID 轉為數值編碼 ===
df_reg['Company_Code'] = df_reg['Company_ID'].astype('category').cat.codes
df_reg['Year_Code'] = df_reg['Year'].astype(int)

# 建立群聚變數矩陣 (N x 2)
groups = df_reg[['Company_Code', 'Year_Code']].values

print(f"\n分析樣本數: {len(df_reg)}")

# Model A1: Future Earnings (Raw Values)
print("\n=== Equation 5 (A1): Future Earnings Change (Raw Values) ===")
formula_a1 = 'Change_In_Earnings ~ Scaled_RD + Investment_MainSGA + Scaled_PPE_Capex + Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage'
model_a1 = smf.ols(formula_a1, data=df_reg)
res_a1 = model_a1.fit(cov_type='cluster', cov_kwds={'groups': groups})
print(res_a1.summary())

# Model A2: Future Earnings (Standardized Values)
print("\n=== Equation 5 (A2): Future Earnings Change (Standardized Inv) ===")
formula_a2 = 'Change_In_Earnings ~ Std_Scaled_RD + Std_Investment_MainSGA + Std_Scaled_PPE_Capex + Std_Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage'
model_a2 = smf.ols(formula_a2, data=df_reg)
res_a2 = model_a2.fit(cov_type='cluster', cov_kwds={'groups': groups})
print(res_a2.summary())

# Model B1: Tobin's Q (Raw Values)
print("\n=== Equation 5 (B1): Tobin\'s Q (Raw Values) ===")
formula_b1 = 'Tobins_Q ~ Scaled_RD + Investment_MainSGA + Scaled_PPE_Capex + Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage'
model_b1 = smf.ols(formula_b1, data=df_reg)
res_b1 = model_b1.fit(cov_type='cluster', cov_kwds={'groups': groups})
print(res_b1.summary())

# Model B2: Tobin's Q (Standardized Values)
print("\n=== Equation 5 (B2): Tobin\'s Q (Standardized Inv) ===")
formula_b2 = 'Tobins_Q ~ Std_Scaled_RD + Std_Investment_MainSGA + Std_Scaled_PPE_Capex + Std_Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage'
model_b2 = smf.ols(formula_b2, data=df_reg)
res_b2 = model_b2.fit(cov_type='cluster', cov_kwds={'groups': groups})
print(res_b2.summary())

# 整理係數比較 (Table 7 Style)
print("\n" + "="*100)
print("【Table 7 Replication】: Future Benefits of Investment Outlays")
print("="*100)
print(f"{'':<25} | {'Future Earnings (Raw)':<22} | {'Future Earnings (Std)':<22} | {'Tobin Q (Raw)':<20} | {'Tobin Q (Std)':<20}")
print("-" * 115)

rows = [
    ('R&D', 'Scaled_RD', 'Std_Scaled_RD'),
    ('MainSG&A Inv', 'Investment_MainSGA', 'Std_Investment_MainSGA'),
    ('PPE Capex', 'Scaled_PPE_Capex', 'Std_Scaled_PPE_Capex'),
    ('Intan Capex', 'Scaled_Intan_Capex', 'Std_Scaled_Intan_Capex'),
    ('Leverage', 'Leverage', 'Leverage') # Leverage is typically raw in both or standardized in both? Paper standardizes only investment variables. We keep Leverage raw in both or standardize? Let's check raw leverage coefficient in standardized model. Wait, formula_a2 uses 'Leverage' (raw). That's fine.
]

for label, raw_var, std_var in rows:
    # 判斷變數是否存在於模型中 (Leverage 在 Std 模型中是 Raw Leverage)
    coef_a1 = res_a1.params[raw_var]
    t_a1 = res_a1.tvalues[raw_var]
    
    # 對於標準化模型，Leverage 變數名稱還是 'Leverage'
    var_a2 = std_var if std_var in res_a2.params else raw_var
    coef_a2 = res_a2.params[var_a2]
    t_a2 = res_a2.tvalues[var_a2]
    
    coef_b1 = res_b1.params[raw_var]
    t_b1 = res_b1.tvalues[raw_var]
    
    var_b2 = std_var if std_var in res_b2.params else raw_var
    coef_b2 = res_b2.params[var_b2]
    t_b2 = res_b2.tvalues[var_b2]
    
    # 標記顯著性
    def get_sig(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        
    sig_a1 = get_sig(res_a1.pvalues[raw_var])
    sig_a2 = get_sig(res_a2.pvalues[var_a2])
    sig_b1 = get_sig(res_b1.pvalues[raw_var])
    sig_b2 = get_sig(res_b2.pvalues[var_b2])

    print(f"{label:<25} | {coef_a1:>8.4f}{sig_a1:<3} (t={t_a1:.2f}) | {coef_a2:>8.4f}{sig_a2:<3} (t={t_a2:.2f}) | {coef_b1:>8.4f}{sig_b1:<3} (t={t_b1:.2f}) | {coef_b2:>8.4f}{sig_b2:<3} (t={t_b2:.2f})")

print("-" * 115)
print("註: 括號內為 t-value (Cluster-Robust)。 *** p<0.01, ** p<0.05, * p<0.1")