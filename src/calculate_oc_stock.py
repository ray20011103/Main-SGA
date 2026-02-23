import pandas as pd
import numpy as np
import os

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
data_raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
data_processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

# 1. 讀取 CPI 資料
print("正在讀取 CPI 資料...")
cpi_file = os.path.join(data_raw_dir, 'CPI.csv')
df_cpi = pd.read_csv(cpi_file)
df_cpi['Year'] = df_cpi['年月'].astype(str).str[:4].astype(int)
# 計算年度平均 CPI
annual_cpi = df_cpi.groupby('Year')['數值'].mean().reset_index()
annual_cpi.columns = ['Year', 'CPI']
# 基期調整 (2021=100) -> 轉為比例
annual_cpi['CPI_Deflator'] = annual_cpi['CPI'] / 100.0

# 2. 讀取先前拆解好的結果
input_file = os.path.join(data_processed_dir, 'main_sga_v2_result.csv')
print(f"正在讀取 {input_file}...")
df = pd.read_csv(input_file)

# 合併 CPI
df = pd.merge(df, annual_cpi[['Year', 'CPI_Deflator']], on='Year', how='left')
# 如果找不到 CPI (例如太早期的年份)，用最近的替代或設為 1 (假設不平減)
# 這裡簡單 fillna(1) 避免報錯，但應注意資料範圍
df['CPI_Deflator'] = df['CPI_Deflator'].fillna(1.0)

# 3. 計算實質金額 (Real Investment)
# 名目投資 = Investment_MainSGA * Avg_Total_Assets
# 實質投資 = 名目投資 / CPI_Deflator
df['Nominal_Investment'] = df['Investment_MainSGA'] * df['Avg_Total_Assets']
df['Real_Investment'] = df['Nominal_Investment'] / df['CPI_Deflator']

# 4. 設定 PIM 參數 (根據 Eisfeldt & Papanikolaou 2013)
delta = 0.15  # 折舊率 15%
g = 0.2964  

# 5. 對每家公司進行 PIM 計算
def calculate_pim(group):
    group = group.sort_values('Year')
    oc_stocks = []
    
    # 取得第一年的流量作為初始化的基礎
    # 使用 Real Investment
    first_flow = group['Real_Investment'].iloc[0]
    
    # 初始存量公式: O0 = F1 / (g + delta)
    current_stock = first_flow / (g + delta)
    oc_stocks.append(current_stock)
    
    # 遞迴計算後續年份: Ot = Ot-1 * (1 - delta) + Ft
    for i in range(1, len(group)):
        flow = group['Real_Investment'].iloc[i]
        current_stock = current_stock * (1 - delta) + flow
        oc_stocks.append(current_stock)
    
    group['Org_Capital_Stock_Real'] = oc_stocks
    return group

print("正在使用永續盤存法 (PIM) 計算組織資本存量 (Real Terms)...")
# 按公司群組並套用 PIM
df_result = df.groupby('Company_ID', group_keys=False).apply(calculate_pim)

# 6. 計算 OC/Assets (組織資本密集度)
# 分子是 Real OC Stock, 分母是 Nominal Assets?
# E-P (2013) 使用 Real OC / Real Assets 或 Nominal OC / Nominal Assets?
# 通常為了 Ratio 意義，分子分母要是同一單位。
# 既然我們算出了 Real OC，我們應該把它乘回 CPI 轉回 Nominal OC，再除以 Nominal Assets
# 或者將 Assets 也平減。
# E-P (2013): "stock of organizational capital, K_O, scaled by total assets, A"
# 為了避免混淆，將 Real OC 轉回 Nominal OC 比較直觀，因為股價和市值都是 Nominal 的。
# Nominal_OC_t = Real_OC_t * CPI_t
df_result['Org_Capital_Stock_Nominal'] = df_result['Org_Capital_Stock_Real'] * df_result['CPI_Deflator']
df_result['OC_Intensity'] = df_result['Org_Capital_Stock_Nominal'] / df_result['Avg_Total_Assets']

# 7. 儲存結果
output_file = os.path.join(data_processed_dir, 'org_capital_results.csv')
df_result.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"計算完成！結果已存至: {output_file}")
print("\n組織資本存量統計摘要:")
print(df_result[['Org_Capital_Stock_Nominal', 'OC_Intensity']].describe())

# 顯示前幾筆資料看看
print("\n前 5 筆計算結果:")
print(df_result[['Company_ID', 'Year', 'Real_Investment', 'Org_Capital_Stock_Nominal', 'OC_Intensity']].head())
