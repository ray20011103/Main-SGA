import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import os

warnings.filterwarnings("ignore")

# ==========================================
# 1. 讀取與前處理資料
# ==========================================

print("1. 讀取資料中...")

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
data_raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
data_processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

# (1) 讀取組織資本資料 (年頻率)
df_oc = pd.read_csv(os.path.join(data_processed_dir, 'org_capital_results.csv'))
df_oc['Company_ID'] = df_oc['Company_ID'].astype(str)

# (2) 讀取個股報酬率資料 (月頻率)
df_ret = pd.read_csv(os.path.join(data_raw_dir, 'industry&return.csv'))
df_ret['Company_ID'] = df_ret['證券代碼'].astype(str).str.split().str[0]
df_ret['Date'] = pd.to_datetime(df_ret['年月'], format='%Y%m')
df_ret['Year'] = df_ret['Date'].dt.year
df_ret['Month'] = df_ret['Date'].dt.month
# 報酬率與市值
df_ret['Return'] = pd.to_numeric(df_ret['報酬率_月'], errors='coerce') * 100
df_ret['Market_Cap'] = pd.to_numeric(df_ret['市值(百萬元)'], errors='coerce')

# (3) 讀取因子資料
try:
    df_factors = pd.read_csv(os.path.join(data_raw_dir, '8factors.csv'), sep='\t', encoding='utf-8')
except UnicodeDecodeError:
    df_factors = pd.read_csv(os.path.join(data_raw_dir, '8factors.csv'), sep='\t', encoding='cp950')

df_factors['Date'] = pd.to_datetime(df_factors['年月'], format='%Y%m')
rename_map = {
    '市場風險溢酬': 'Mkt_RF', '規模溢酬 (5因子)': 'SMB', '淨值市價比溢酬': 'HML',
    '無風險利率': 'RF', '盈利能力因子': 'RMW', '投資因子': 'CMA'
}
df_factors = df_factors.rename(columns=rename_map)
for col in ['Mkt_RF', 'SMB', 'HML', 'RF', 'RMW', 'CMA']:
    if col in df_factors.columns:
        df_factors[col] = pd.to_numeric(df_factors[col], errors='coerce')

# ==========================================
# 2. 資料合併與預處理
# ==========================================
print("2. 資料合併與預處理...")

# 邏輯: 每年 5 月底進行 Rebalance (假設 3/31 年報公布，給予 1 個月緩衝)
df_ret['Financial_Report_Year'] = np.where(df_ret['Month'] >= 5, df_ret['Year'] - 1, df_ret['Year'] - 2)

df_merged = pd.merge(
    df_ret, 
    df_oc[['Company_ID', 'Year', 'OC_Intensity']], 
    left_on=['Company_ID', 'Financial_Report_Year'], 
    right_on=['Company_ID', 'Year'], 
    how='inner'
)

df_merged = df_merged.dropna(subset=['OC_Intensity', 'Return', 'Market_Cap', 'TEJ產業_代碼'])

# ==========================================
# 3. 單變數排序 (Single Sort: OC Intensity) - Table 7
# ==========================================
print("\n=== Generating Table 7: Single Sorts ===")

def assign_quintiles(series):
    try:
        return pd.qcut(series, 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        try:
             return pd.qcut(series, 3, labels=[2, 3, 4]) 
        except ValueError:
            return np.nan

# 產業內排序
df_merged['OC_Rank'] = df_merged.groupby(['Date', 'TEJ產業_代碼'])['OC_Intensity'].transform(assign_quintiles)

# 計算 VW 報酬
port_ret = df_merged.groupby(['Date', 'OC_Rank'])[['Return', 'Market_Cap']].apply(
    lambda x: np.average(x['Return'], weights=x['Market_Cap'])
).reset_index(name='Portfolio_Return')

port_ret_pivot = port_ret.pivot(index='Date', columns='OC_Rank', values='Portfolio_Return')
port_ret_pivot['High-Low'] = port_ret_pivot[5] - port_ret_pivot[1]

# 整理 Table 7 數據
table7_data = []
for col in [1, 2, 3, 4, 5, 'High-Low']:
    series = port_ret_pivot[col]
    mean = series.mean()
    # Newey-West t-stat
    model = sm.OLS(series, sm.add_constant(np.ones(len(series)))).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    t_stat = model.tvalues[0]
    table7_data.append({'Rank': col, 'Mean': mean, 't-stat': t_stat})

pd.DataFrame(table7_data).to_csv(os.path.join(data_processed_dir, 'Table7_SingleSort.csv'), index=False)
print("Table 7 saved.")

# ==========================================
# 4. 雙重排序 (Double Sorts: Size vs OC) - Table 8
# ==========================================
print("\n=== Generating Table 8: Double Sorts ===")

# (1) 獨立排序
df_merged['Size_Group'] = df_merged.groupby('Date')['Market_Cap'].transform(
    lambda x: pd.qcut(x, 2, labels=['Small', 'Big'])
)

# (2) 計算 2x5 = 10 個投資組合的報酬
double_sort_ret = df_merged.groupby(['Date', 'Size_Group', 'OC_Rank'])[['Return', 'Market_Cap']].apply(
    lambda x: np.average(x['Return'], weights=x['Market_Cap'])
).reset_index(name='Ret')

ds_pivot = double_sort_ret.pivot_table(index='Date', columns=['Size_Group', 'OC_Rank'], values='Ret')

# 整理 Table 8 數據
table8_rows = []
for size_grp in ['Small', 'Big']:
    row_means = {}
    row_tstats = {}
    
    # Q1-Q5
    for q in [1, 2, 3, 4, 5]:
        series = ds_pivot[(size_grp, q)]
        row_means[f'Q{q}'] = series.mean()
        row_tstats[f'Q{q}_t'] = sm.OLS(series, sm.add_constant(np.ones(len(series)))).fit(cov_type='HAC', cov_kwds={'maxlags': 6}).tvalues[0]
    
    # High-Low
    hl_series = ds_pivot[(size_grp, 5)] - ds_pivot[(size_grp, 1)]
    row_means['High-Low'] = hl_series.mean()
    row_tstats['High-Low_t'] = sm.OLS(hl_series, sm.add_constant(np.ones(len(hl_series)))).fit(cov_type='HAC', cov_kwds={'maxlags': 6}).tvalues[0]
    
    # 存檔格式
    table8_rows.append({'Size': size_grp, 'Type': 'Mean', **row_means})
    table8_rows.append({'Size': size_grp, 'Type': 't-stat', **row_tstats})

pd.DataFrame(table8_rows).to_csv(os.path.join(data_processed_dir, 'Table8_DoubleSort.csv'), index=False)
print("Table 8 saved.")

# 準備 Figure 1 數據 (Small Firm High-Low Cumulative Return)
small_hl_series = ds_pivot[('Small', 5)] - ds_pivot[('Small', 1)]
big_hl_series = ds_pivot[('Big', 5)] - ds_pivot[('Big', 1)]

fig1_data = pd.DataFrame({
    'Date': small_hl_series.index,
    'Small_Firm_HL': (1 + small_hl_series/100).cumprod(),
    'Big_Firm_HL': (1 + big_hl_series/100).cumprod()
})
fig1_data.to_csv(os.path.join(data_processed_dir, 'Figure1_CumulativeReturn.csv'), index=False)
print("Figure 1 data saved.")

# ==========================================
# 5. 因子迴歸 (Factor Regressions) - Table 9
# ==========================================
print("\n=== Generating Table 9: Factor Regressions (Small Firms) ===")

# 只針對 Small Firm High-Low 跑因子回歸
y_small = small_hl_series.reset_index(drop=True)
# 需要確保日期對齊
regression_data = pd.merge(small_hl_series.rename('High-Low'), df_factors, on='Date', how='inner')

models = {
    'CAPM': ['Mkt_RF'],
    'FF3': ['Mkt_RF', 'SMB', 'HML'],
    'FF5': ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']
}

table9_data = []
y = regression_data['High-Low'] # High-Low is already excess return (zero-cost)

for name, factors in models.items():
    X = sm.add_constant(regression_data[factors])
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    
    res = {
        'Model': name,
        'Alpha': model.params['const'],
        'Alpha_t': model.tvalues['const'],
        'R2': model.rsquared
    }
    # Add factor loadings
    for f in ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']:
        if f in factors:
            res[f'{f}'] = model.params[f]
            res[f'{f}_t'] = model.tvalues[f]
    
    table9_data.append(res)

pd.DataFrame(table9_data).to_csv(os.path.join(data_processed_dir, 'Table9_FactorRegressions.csv'), index=False)
print("Table 9 saved.")
print("\nAll Asset Pricing Tables Generated Successfully.")
