import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
import os

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
data_raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
data_processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

# 1. 讀取資料
print("正在讀取資料...")

# (A) 財報資料 (含 Investment_MainSGA)
# 這裡多讀取 Industry_Group 以便檢查產業分布
df_sga = pd.read_csv(os.path.join(data_processed_dir, 'main_sga_v2_result.csv'))
df_sga['Company_ID'] = df_sga['Company_ID'].astype(str)
# 只需要這幾個欄位
df_rank = df_sga[['Company_ID', 'Year', 'Investment_MainSGA', 'Industry_Group']].dropna()

# (B) 報酬率資料
# 由於檔案較大，我們指定 dtype 以節省記憶體
df_ret = pd.read_csv(os.path.join(data_raw_dir, 'industry&return.csv'), 
                     dtype={'年月': str}, # 年月先讀成字串比較好處理
                     usecols=['證券代碼', '年月', '報酬率_月', '市值(百萬元)'])

# 清洗報酬率資料
df_ret['Company_ID'] = df_ret['證券代碼'].astype(str).str.split().str[0]
df_ret['Return'] = pd.to_numeric(df_ret['報酬率_月'], errors='coerce')
# 清洗市值資料 (移除可能的逗號並轉數值)
df_ret['MV'] = pd.to_numeric(df_ret['市值(百萬元)'].astype(str).str.replace(',', ''), errors='coerce')

# 轉換日期
df_ret['Date'] = pd.to_datetime(df_ret['年月'], format='%Y%m')
df_ret['Ret_Year'] = df_ret['Date'].dt.year
df_ret['Ret_Month'] = df_ret['Date'].dt.month

# 移除沒有報酬率或市值的資料
df_ret = df_ret.dropna(subset=['Return', 'MV'])

# === 新增：對報酬率進行 Winsorization (避免極端值干擾) ===
print("正在對報酬率進行 Winsorization (1%, 99%)...")
df_ret['Return'] = winsorize(df_ret['Return'], limits=[0.01, 0.01])

print(f"報酬率資料筆數: {len(df_ret)}")

# 2. 建立投資組合 (Portfolio Formation)
print("正在建構投資組合...")

# 儲存每家公司在每個年度的排名分組
# 對應邏輯: Financial Year T -> 影響 Return Period: (T+1).05 ~ (T+2).04
portfolio_assignments = []

years = sorted(df_rank['Year'].unique())
for year in years:
    # 取得該年度所有公司的數據
    current_year_data = df_rank[df_rank['Year'] == year].copy()
    
    # 根據 Investment_MainSGA 進行分組 (Q1=Low, Q5=High)
    # 修改為: 產業內分組 (Within-Industry Ranking)
    
    def industry_rank(group):
        # 如果該產業公司數太少，無法有效分組，則回傳空值或略過
        if len(group) < 5: 
            return None
            
        try:
            return pd.qcut(group['Investment_MainSGA'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except:
            # 處理重複值問題
            pct = group['Investment_MainSGA'].rank(pct=True)
            return pd.cut(pct, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # 對每個產業分別進行排名
    current_year_data['Rank'] = current_year_data.groupby('Industry_Group', group_keys=False).apply(industry_rank)
    
    # 移除無法分組的產業 (Rank 為 NaN)
    current_year_data = current_year_data.dropna(subset=['Rank'])
    
    current_year_data['Portfolio_Year'] = year # 這是財報年度
    portfolio_assignments.append(current_year_data[['Company_ID', 'Portfolio_Year', 'Rank', 'Industry_Group']])

df_assignments = pd.concat(portfolio_assignments)

# === 新增：檢查分組統計 ===
print("\n" + "="*50)
print("【投資組合分組檢查】")
print(f"總財報年度數: {len(years)} ({years[0]} - {years[-1]})")

# 檢查 1: 每年每組的平均公司數
grp_counts = df_assignments.groupby(['Portfolio_Year', 'Rank'], observed=True).size().unstack()
print(f"\n1. 每組平均公司數 (各年度平均):")
print(grp_counts.mean().round(1))

# 檢查 2: Q5 (High Investment) 的產業分布是否過度集中？
# 統計 Q5 中各產業出現的總次數
q5_firms = df_assignments[df_assignments['Rank'] == 'Q5']
top_industries_q5 = q5_firms['Industry_Group'].value_counts().head(5)
total_q5 = len(q5_firms)

print(f"\n2. 高投資組 (Q5) 的前五大產業分布:")
for ind, count in top_industries_q5.items():
    print(f"   - {ind}: {count} 家次 ({count/total_q5*100:.1f}%)")

# 檢查 3: Q1 (Low Investment) 的產業分布
q1_firms = df_assignments[df_assignments['Rank'] == 'Q1']
top_industries_q1 = q1_firms['Industry_Group'].value_counts().head(5)
total_q1 = len(q1_firms)

print(f"\n3. 低投資組 (Q1) 的前五大產業分布:")
for ind, count in top_industries_q1.items():
    print(f"   - {ind}: {count} 家次 ({count/total_q1*100:.1f}%)")

print("="*50 + "\n")
# ========================

# 3. 合併報酬率
# 這是一個 Many-to-Many 的對應，我們需要一個有效率的方法
# 方法: 為 Return 資料加上 "對應的財報年度" (Portfolio_Year)
# 規則: 如果是 2015/05 ~ 2016/04，對應的財報年度是 2014
# 公式: Portfolio_Year = Ret_Year - 1 (如果月份 >= 5) else Ret_Year - 2

conditions = [
    df_ret['Ret_Month'] >= 5,
    df_ret['Ret_Month'] < 5
]
choices = [
    df_ret['Ret_Year'] - 1,
    df_ret['Ret_Year'] - 2
]
df_ret['Portfolio_Year'] = np.select(conditions, choices)

# 現在可以 Merge 了
print("正在合併財報與報酬率...")
df_final = pd.merge(df_ret, df_assignments, on=['Company_ID', 'Portfolio_Year'], how='inner')

# 4. 計算投資組合績效 (Value-Weighted Returns)
print("計算績效中 (Value-Weighted)...")

# 計算權重 (Market Value)
# 論文 Appendix B: Value-weighted returns (VWRet) are calculated for each portfolio-month
def weighted_average(x):
    if x['MV'].sum() == 0:
        return np.nan
    return np.average(x['Return'], weights=x['MV'])

# 計算每個月、每個 Rank 的加權平均報酬
monthly_portfolio_ret = df_final.groupby(['Date', 'Rank'], observed=True).apply(weighted_average, include_groups=False).reset_index(name='Return')

# 整理成 Pivot Table: Index=Date, Columns=Rank
pivot_ret = monthly_portfolio_ret.pivot(index='Date', columns='Rank', values='Return')
pivot_ret.columns = ['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High']

# 計算多空策略 (Long-Short)
pivot_ret['Long_Short'] = pivot_ret['Q5_High'] - pivot_ret['Q1_Low']

# 5. 輸出統計結果
print("\n" + "="*50)
print("【投資組合策略績效分析 (Value-Weighted + Winsorized)】")
print("策略: 每年 5 月買入高 MainSG&A 投資公司 (Q5)，放空低投資公司 (Q1)")
print("分組方式: 產業內分組 (Within-Industry Quintiles)")
print("樣本期間: ", pivot_ret.index.min().strftime('%Y-%m'), "至", pivot_ret.index.max().strftime('%Y-%m'))
print("="*50)

# 計算年化報酬與檢定 (Newey-West t-stat)
# 將月報酬 * 100 轉為百分比方便閱讀
stats_df = pivot_ret * 100

results = []
for col in stats_df.columns:
    series = stats_df[col].dropna()
    mean_ret = series.mean()
    std_dev = series.std()
    
    # 使用 statsmodels 跑截距項回歸來計算 Newey-West t-stat
    X = sm.add_constant(np.ones(len(series)))
    model = sm.OLS(series, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    t_stat = model.tvalues[0]
    p_val = model.pvalues[0]
    
    results.append({
        'Rank': col,
        'Mean Return (%)': mean_ret,
        't-stat (NW)': t_stat,
        'p-value': p_val
    })

summary = pd.DataFrame(results).set_index('Rank')

print(summary[['Mean Return (%)', 't-stat (NW)', 'p-value']].round(3))


# 輸出結果到 CSV
output_file = os.path.join(data_processed_dir, 'hedge_portfolio_performance.csv')
pivot_ret.to_csv(output_file)
print(f"\n詳細月報酬數據已儲存至: {output_file}")