import pandas as pd
import numpy as np
import os

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
data_raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
data_processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

print("正在讀取資料以生成統計表格...")

# 讀取資料
df_sga = pd.read_csv(os.path.join(data_processed_dir, 'main_sga_v2_result.csv'))
df_oc = pd.read_csv(os.path.join(data_processed_dir, 'org_capital_results.csv'))
df_capex = pd.read_csv(os.path.join(data_raw_dir, 'capex.csv'))
df_industry = pd.read_csv(os.path.join(data_raw_dir, 'industry.csv'))
df_mv = pd.read_csv(os.path.join(data_raw_dir, 'industry&return.csv'), 
                    dtype={'年月': str}, usecols=['證券代碼', '年月', '市值(百萬元)'])

# 資料預處理
df_sga['Company_ID'] = df_sga['Company_ID'].astype(str)
df_oc['Company_ID'] = df_oc['Company_ID'].astype(str)

df_capex.columns = df_capex.columns.str.strip()
df_capex['Company_ID'] = df_capex['公司'].astype(str).str.split().str[0]
df_capex['Year'] = pd.to_datetime(df_capex['年月']).dt.year
df_capex['Tobins_Q'] = pd.to_numeric(df_capex['Tobins Q'], errors='coerce')
if '負債總額' in df_capex.columns:
    df_capex['Total_Liabilities'] = pd.to_numeric(df_capex['負債總額'].astype(str).str.replace(',', ''), errors='coerce')
else:
    df_capex['Total_Liabilities'] = np.nan

df_industry.columns = df_industry.columns.str.strip()
df_industry['Company_ID'] = df_industry['公司'].astype(str).str.split().str[0]
df_industry['Year'] = pd.to_datetime(df_industry['年月']).dt.year
df_industry['Total_Assets'] = pd.to_numeric(df_industry['資產總額'].astype(str).str.replace(',', ''), errors='coerce')

df_mv['Company_ID'] = df_mv['證券代碼'].astype(str).str.split().str[0]
df_mv['Date'] = pd.to_datetime(df_mv['年月'], format='%Y%m')
df_mv['Year'] = df_mv['Date'].dt.year
df_mv['MV'] = pd.to_numeric(df_mv['市值(百萬元)'].astype(str).str.replace(',', ''), errors='coerce')
df_mv_year = df_mv.sort_values('Date').groupby(['Company_ID', 'Year'])['MV'].last().reset_index()

# 合併主資料表
df_merged = pd.merge(df_sga, df_oc[['Company_ID', 'Year', 'OC_Intensity']], on=['Company_ID', 'Year'], how='inner')
df_merged = pd.merge(df_merged, df_capex[['Company_ID', 'Year', 'Tobins_Q', 'Total_Liabilities']], 
                     on=['Company_ID', 'Year'], how='inner')
df_merged = pd.merge(df_merged, df_industry[['Company_ID', 'Year', 'Total_Assets']], 
                     on=['Company_ID', 'Year'], how='left')
df_merged = pd.merge(df_merged, df_mv_year, on=['Company_ID', 'Year'], how='left')

# 建構變數
df_merged['Size'] = np.log(df_merged['MV'])
df_merged['Leverage'] = df_merged['Total_Liabilities'] / df_merged['Total_Assets']
df_merged['SG&A/Assets'] = df_merged['Scaled_MainSGA'] # 這其實已經是 Scaled 了

# ==========================================
# Table 1: Industry Distribution & R2
# ==========================================
print("\nGenerating Table 1: Industry Distribution...")
# 使用 main_sga_v2_result.csv 裡的 Regression_R2
# 每個產業-年度只有一個 R2，我們取平均
if 'Regression_R2' in df_merged.columns:
    table1 = df_merged.groupby('Industry_Group').agg(
        Firms=('Company_ID', 'nunique'),
        Obs=('Company_ID', 'count'),
        Avg_R2=('Regression_R2', 'mean')
    ).reset_index()
    
    table1['Firms (%)'] = table1['Firms'] / table1['Firms'].sum() * 100
    table1 = table1.sort_values('Firms', ascending=False)
    
    output_path = os.path.join(data_processed_dir, 'table1_industry.csv')
    table1.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(table1.head())

# ==========================================
# Table 2: Descriptive Statistics
# ==========================================
print("\nGenerating Table 2: Descriptive Statistics...")
vars_desc = ['OC_Intensity', 'Tobins_Q', 'Size', 'Leverage', 'SG&A/Assets']
# 先進行 Winsorization (1%, 99%)
from scipy.stats.mstats import winsorize
df_clean = df_merged.dropna(subset=vars_desc).copy()
for col in vars_desc:
    df_clean[col] = winsorize(df_clean[col], limits=[0.01, 0.01])

desc_stats = df_clean[vars_desc].describe().T
desc_stats = desc_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
output_path = os.path.join(data_processed_dir, 'table2_descriptive.csv')
desc_stats.to_csv(output_path)
print(f"Saved: {output_path}")
print(desc_stats)

# ==========================================
# Table 3: Correlation Matrix
# ==========================================
print("\nGenerating Table 3: Correlation Matrix...")
corr_matrix = df_clean[vars_desc].corr()
output_path = os.path.join(data_processed_dir, 'table3_correlation.csv')
corr_matrix.to_csv(output_path)
print(f"Saved: {output_path}")
print(corr_matrix)
