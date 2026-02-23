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
# 主要差異: 讀取 OC Stock 結果
df_oc = pd.read_csv(os.path.join(data_processed_dir, 'org_capital_results.csv'))
df_trad_oc = pd.read_csv(os.path.join(data_processed_dir, 'traditional_oc_results.csv')) # 新增讀取
df_sga = pd.read_csv(os.path.join(data_processed_dir, 'main_sga_v2_result.csv'))
df_capex = pd.read_csv(os.path.join(data_raw_dir, 'capex.csv'))
df_industry = pd.read_csv(os.path.join(data_raw_dir, 'industry.csv'))
df_mv = pd.read_csv(os.path.join(data_raw_dir, 'industry&return.csv'), 
                    dtype={'年月': str},
                    usecols=['證券代碼', '年月', '市值(百萬元)'])

# 2. 資料清洗與合併準備
df_oc['Company_ID'] = df_oc['Company_ID'].astype(str)
df_trad_oc['Company_ID'] = df_trad_oc['Company_ID'].astype(str) # 新增
df_sga['Company_ID'] = df_sga['Company_ID'].astype(str)

df_capex.columns = df_capex.columns.str.strip()
df_capex['Company_ID'] = df_capex['公司'].astype(str).str.split().str[0]
df_capex['Year'] = pd.to_datetime(df_capex['年月']).dt.year

# 處理數值欄位
df_capex['PPE_Capex'] = pd.to_numeric(df_capex['購置不動產廠房設備（含預付）－CFI'].astype(str).str.replace(',', ''), errors='coerce').abs()
df_capex['Intan_Capex'] = pd.to_numeric(df_capex['無形資產（增加）減少－CFI'].astype(str).str.replace(',', ''), errors='coerce').abs()
df_capex['Tobins_Q'] = pd.to_numeric(df_capex['Tobins Q'], errors='coerce')
if '負債總額' in df_capex.columns:
    df_capex['Total_Liabilities'] = pd.to_numeric(df_capex['負債總額'].astype(str).str.replace(',', ''), errors='coerce')
else:
    df_capex['Total_Liabilities'] = np.nan

df_industry.columns = df_industry.columns.str.strip()
df_industry['Company_ID'] = df_industry['公司'].astype(str).str.split().str[0]
df_industry['Year'] = pd.to_datetime(df_industry['年月']).dt.year
df_industry['Net_Income'] = pd.to_numeric(df_industry['歸屬母公司淨利（損）'].astype(str).str.replace(',', ''), errors='coerce')
df_industry['Total_Assets'] = pd.to_numeric(df_industry['資產總額'].astype(str).str.replace(',', ''), errors='coerce')
df_industry['Industry_Group'] = df_industry['TEJ產業_代碼'].astype(str).str[:3]

df_industry = df_industry.sort_values(['Company_ID', 'Year'])
df_industry['Lag_Net_Income'] = df_industry.groupby('Company_ID')['Net_Income'].shift(1)
df_industry['Current_Earnings_Growth'] = (df_industry['Net_Income'] - df_industry['Lag_Net_Income']) / df_industry['Lag_Net_Income'].abs()
df_industry['Current_Earnings_Growth'] = df_industry['Current_Earnings_Growth'].replace([np.inf, -np.inf], np.nan)

df_mv['Company_ID'] = df_mv['證券代碼'].astype(str).str.split().str[0]
df_mv['Date'] = pd.to_datetime(df_mv['年月'], format='%Y%m')
df_mv['Year'] = df_mv['Date'].dt.year
df_mv['MV'] = pd.to_numeric(df_mv['市值(百萬元)'].astype(str).str.replace(',', ''), errors='coerce')
df_mv_year = df_mv.sort_values('Date').groupby(['Company_ID', 'Year'])['MV'].last().reset_index()

# 3. 合併所有資料
# 先合併 OC 與 SGA
df_merged = pd.merge(df_sga, df_oc[['Company_ID', 'Year', 'OC_Intensity']], on=['Company_ID', 'Year'], how='inner')
# 合併 Traditional OC
df_merged = pd.merge(df_merged, df_trad_oc[['Company_ID', 'Year', 'Traditional_OC_Intensity']], on=['Company_ID', 'Year'], how='inner')

df_merged = pd.merge(df_merged, df_capex[['Company_ID', 'Year', 'PPE_Capex', 'Intan_Capex', 'Tobins_Q', 'Total_Liabilities']], 
                     on=['Company_ID', 'Year'], how='inner')

df_merged = pd.merge(df_merged, df_industry[['Company_ID', 'Year', 'Net_Income', 'Current_Earnings_Growth', 'Total_Assets']], 
                     on=['Company_ID', 'Year'], how='left')

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

# 新增: Lagged Tobin's Q (控制持續性)
df_merged = df_merged.sort_values(by=['Company_ID', 'Year'])
df_merged['Lag_Tobins_Q'] = df_merged.groupby('Company_ID')['Tobins_Q'].shift(1)

# 分組變數 (Size Median Split)
size_median = df_merged['Size'].median()
df_merged['Is_Large'] = df_merged['Size'] > size_median

# 6. 準備迴歸資料
cols_to_use = ['Tobins_Q', 'Lag_Tobins_Q', 'OC_Intensity', 'Traditional_OC_Intensity', 'Investment_MainSGA',
               'Scaled_RD', 'Scaled_PPE_Capex', 'Scaled_Intan_Capex', 
               'Size', 'Current_Earnings_Growth', 'Loss_Dummy', 'Leverage',
               'Company_ID', 'Year', 'Industry_Group', 'Is_Large']

df_reg = df_merged.dropna(subset=cols_to_use).copy()

# === Step A: Winsorization (1% & 99%) ===
print("正在進行 Winsorization (1%, 99%):..")
vars_to_winsorize = ['Tobins_Q', 'Lag_Tobins_Q', 'OC_Intensity', 'Traditional_OC_Intensity', 'Investment_MainSGA',
                     'Scaled_RD', 'Scaled_PPE_Capex', 'Scaled_Intan_Capex',
                     'Size', 'Current_Earnings_Growth', 'Leverage']

for col in vars_to_winsorize:
    df_reg[col] = winsorize(df_reg[col], limits=[0.01, 0.01])

# === Step B: Industry-Year Standardization (Z-Score) ===
print("正在進行 Industry-Year Standardization...")
invest_vars = ['OC_Intensity', 'Traditional_OC_Intensity', 'Investment_MainSGA', 'Scaled_RD', 'Scaled_PPE_Capex', 'Scaled_Intan_Capex']

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
groups = df_reg[['Company_Code', 'Year_Code']].values

print(f"\n分析樣本數: {len(df_reg)}")

# 定義迴歸函數
def run_regression_suite(data, label, use_stock=True, use_trad=False):
    if use_trad:
        main_var = 'Std_Traditional_OC_Intensity'
    else:
        main_var = 'Std_OC_Intensity' if use_stock else 'Std_Investment_MainSGA'
        
    print(f"\n=== {label} (Main Var: {main_var}) ===")
    
    # Model 1: Basic OLS
    f1 = f'Tobins_Q ~ {main_var} + Std_Scaled_RD + Std_Scaled_PPE_Capex + Std_Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage'
    
    # Model 2: With Fixed Effects (Industry + Year)
    # 使用 C() 語法加入虛擬變數
    f2 = f'Tobins_Q ~ {main_var} + Std_Scaled_RD + Std_Scaled_PPE_Capex + Std_Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage + C(Industry_Group) + C(Year)'
    
    # Model 3: Dynamic (Lag Q) + Fixed Effects
    f3 = f'Tobins_Q ~ Lag_Tobins_Q + {main_var} + Std_Scaled_RD + Std_Scaled_PPE_Capex + Std_Scaled_Intan_Capex + Size + Current_Earnings_Growth + Loss_Dummy + Leverage + C(Industry_Group) + C(Year)'
    
    models = [('Basic OLS', f1), ('Fixed Effects', f2), ('Dynamic + FE', f3)]
    
    for name, formula in models:
        model = smf.ols(formula, data=data)
        try:
            # Cluster SE by Firm & Year
            res = model.fit(cov_type='cluster', cov_kwds={'groups': data[['Company_Code', 'Year_Code']].values})
        except:
            res = model.fit(cov_type='HC1')
            
        coef = res.params[main_var]
        t_stat = res.tvalues[main_var]
        p_val = res.pvalues[main_var]
        adj_r2 = res.rsquared_adj
        
        # 如果有 Lag Q，也印出它的係數
        lag_info = ""
        if 'Lag_Tobins_Q' in res.params:
            lag_coef = res.params['Lag_Tobins_Q']
            lag_t = res.tvalues['Lag_Tobins_Q']
            lag_info = f" | Lag Q: {lag_coef:.3f} (t={lag_t:.2f})"
            
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"{name:<15} | Coef: {coef:.4f}{sig:<3} (t={t_stat:.2f}) | Adj R2: {adj_r2:.3f}{lag_info}")

print("\n" + "="*80)
print("【Tobin's Q Robustness Test: Dynamic Panel & Fixed Effects】")
print("="*80)

# 1. Full Sample
run_regression_suite(df_reg, "Full Sample", use_stock=True)

# 2. Sub-sample: Small vs. Large
run_regression_suite(df_reg[df_reg['Is_Large'] == False], "Small Firms", use_stock=True)
run_regression_suite(df_reg[df_reg['Is_Large'] == True], "Big Firms", use_stock=True)

print("\n" + "="*80)
print("【Comparison: Main SG&A vs. Traditional OC (Dynamic Model)】")
print("="*80)
# 只跑 Dynamic Model 比較
run_regression_suite(df_reg, "Main SG&A (New)", use_stock=True, use_trad=False)
run_regression_suite(df_reg, "Total SG&A (Trad)", use_stock=True, use_trad=True)

print("\n" + "="*80)
print("Hypothesis: OC Stock Coef remains positive and significant after controlling for Lag Q and FE.")