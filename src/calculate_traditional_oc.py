import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import warnings

warnings.filterwarnings("ignore")

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
data_raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
data_processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

print("Calculating Traditional OC (Total SG&A)...")

# ==========================================
# 1. 讀取與前處理 (與 calculate_mainsga_v2.py 相同)
# ==========================================
df = pd.read_csv(os.path.join(data_raw_dir, 'industry.csv'))
df_cpi = pd.read_csv(os.path.join(data_raw_dir, 'CPI.csv'))

# Preprocess CPI
df_cpi['Year'] = df_cpi['年月'].astype(str).str[:4].astype(int)
annual_cpi = df_cpi.groupby('Year')['數值'].mean().reset_index()
annual_cpi.columns = ['Year', 'CPI']
annual_cpi['CPI_Deflator'] = annual_cpi['CPI'] / 100.0

# Clean Industry Data
df.columns = df.columns.str.strip()
numeric_cols = ['資產總額', '營業收入淨額', '營業費用', '歸屬母公司淨利（損）']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        if col not in ['資產總額', '營業收入淨額']: df[col] = df[col].fillna(0)

df = df.dropna(subset=['資產總額', '營業收入淨額'])
df = df[df['資產總額'] > 0]
df['Company_ID'] = df['公司'].astype(str).str.split().str[0]
df['Date'] = pd.to_datetime(df['年月'])
df['Year'] = df['Date'].dt.year
df = df.sort_values(by=['Company_ID', 'Year'])
df['Industry_Group'] = df['TEJ產業_代碼'].astype(str).str[:3]

# ==========================================
# 2. 建構變數 (Total SG&A)
# ==========================================
# 傳統方法: 直接使用 "營業費用" (Total SG&A)
df['Raw_TotalSGA'] = df['營業費用']

# 計算 Lag Assets, Lag Revenue
df['Lag_Total_Assets'] = df.groupby('Company_ID')['資產總額'].shift(1)
df['Lag_Revenue'] = df.groupby('Company_ID')['營業收入淨額'].shift(1)
df_reg = df.dropna(subset=['Lag_Total_Assets', 'Lag_Revenue']).copy()

df_reg['Avg_Total_Assets'] = (df_reg['資產總額'] + df_reg['Lag_Total_Assets']) / 2
df_reg['Scaled_TotalSGA'] = df_reg['Raw_TotalSGA'] / df_reg['Avg_Total_Assets']
df_reg['Scaled_Revenue'] = df_reg['營業收入淨額'] / df_reg['Avg_Total_Assets']
df_reg['Dummy_Rev_Dec'] = (df_reg['營業收入淨額'] < df_reg['Lag_Revenue']).astype(int)
df_reg['Dummy_Loss'] = (df_reg['歸屬母公司淨利（損）'] < 0).astype(int)

# ==========================================
# 3. 執行迴歸拆解
# ==========================================
print("Decomposing Total SG&A...")
results = []
grouped = df_reg.groupby(['Industry_Group', 'Year'])

for (industry, year), group in grouped:
    if len(group) < 8: continue
    try:
        model = smf.ols('Scaled_TotalSGA ~ Scaled_Revenue + Dummy_Rev_Dec + Dummy_Loss', data=group)
        res = model.fit()
        
        beta1 = res.params['Scaled_Revenue']
        group = group.copy()
        group['Maintenance_SGA'] = beta1 * group['Scaled_Revenue']
        # Investment = Total - Maintenance
        group['Investment_TotalSGA'] = group['Scaled_TotalSGA'] - group['Maintenance_SGA']
        results.append(group)
    except:
        continue

if not results:
    raise ValueError("No results generated.")

df_res = pd.concat(results)

# ==========================================
# 4. PIM 存量計算 (Traditional OC)
# ==========================================
print("Calculating Traditional OC Stock (PIM)...")

# Merge CPI
df_res = pd.merge(df_res, annual_cpi[['Year', 'CPI_Deflator']], on='Year', how='left')
df_res['CPI_Deflator'] = df_res['CPI_Deflator'].fillna(1.0)

# Recover Nominal Investment
df_res['Nominal_Investment'] = df_res['Investment_TotalSGA'] * df_res['Avg_Total_Assets']
df_res['Real_Investment'] = df_res['Nominal_Investment'] / df_res['CPI_Deflator']

# PIM Logic
delta = 0.15
g = 0.2964 # Match calculate_oc_stock.py parameter

def pim_logic(group):
    group = group.sort_values('Year')
    oc_stocks = []
    if len(group) == 0: return group
    
    # Init
    first_flow = group['Real_Investment'].iloc[0]
    # g=0.2964 to match main OC calculation
    current_stock = first_flow / (g + delta) 
    oc_stocks.append(current_stock)
    
    for i in range(1, len(group)):
        flow = group['Real_Investment'].iloc[i]
        current_stock = current_stock * (1 - delta) + flow
        oc_stocks.append(current_stock)
    
    group['Real_Stock'] = oc_stocks
    return group

df_pim = df_res.groupby('Company_ID', group_keys=False).apply(pim_logic)

# Final Scale
df_pim['Nominal_Stock'] = df_pim['Real_Stock'] * df_pim['CPI_Deflator']
df_pim['Traditional_OC_Intensity'] = df_pim['Nominal_Stock'] / df_pim['Avg_Total_Assets']

# Save
output_path = os.path.join(data_processed_dir, 'traditional_oc_results.csv')
cols_to_save = ['Company_ID', 'Year', 'Industry_Group', 'Traditional_OC_Intensity', 'Avg_Total_Assets']
df_pim[cols_to_save].to_csv(output_path, index=False)
print(f"Traditional OC results saved to: {output_path}")