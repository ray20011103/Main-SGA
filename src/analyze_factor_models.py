import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

def analyze_factors():
    print("正在執行因子分析 (Factor Analysis)...")

    # 設定路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
    data_processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

    # 1. 讀取投資組合報酬率 (Long-Short)
    # 這是先前 analyze_hedge_portfolio.py 產出的檔案 (小數格式)
    try:
        df_port = pd.read_csv(os.path.join(data_processed_dir, 'hedge_portfolio_performance.csv'))
    except FileNotFoundError:
        print("錯誤: 找不到 '../data/processed/hedge_portfolio_performance.csv'，請先執行 analyze_hedge_portfolio.py")
        return

    # 處理日期格式
    df_port['Date'] = pd.to_datetime(df_port['Date'])
    df_port['YearMonth'] = df_port['Date'].dt.strftime('%Y%m')
    
    # 將報酬率從小數轉為百分比 (因為因子資料是百分比)
    # 我們主要分析 Long_Short 欄位
    if 'Long_Short' not in df_port.columns:
        print("錯誤: 投資組合資料中缺少 'Long_Short' 欄位")
        return
        
    df_port['Long_Short_Pct'] = df_port['Long_Short'] * 100

    # 2. 讀取 8factors.csv
    try:
        # 假設該檔案在當前目錄
        df_factors = pd.read_csv(os.path.join(data_raw_dir, '8factors.csv'))
    except FileNotFoundError:
        print("錯誤: 找不到 '../data/raw/8factors.csv'")
        return

    # 處理因子資料
    # 欄位: 證券代碼,年月,市場風險溢酬,規模溢酬 (5因子),淨值市價比溢酬,盈利能力因子,投資因子,無風險利率,動能因子,短期反轉因子
    df_factors['YearMonth'] = df_factors['年月'].astype(str)
    
    # 重新命名欄位以便操作
    rename_dict = {
        '市場風險溢酬': 'Mkt_RF',
        '規模溢酬 (5因子)': 'SMB',
        '淨值市價比溢酬': 'HML',
        '盈利能力因子': 'RMW',
        '投資因子': 'CMA',
        '動能因子': 'MOM',
        '短期反轉因子': 'STR',
        '無風險利率': 'RF'
    }
    df_factors = df_factors.rename(columns=rename_dict)

    # 3. 合併資料
    df_merged = pd.merge(df_port, df_factors, on='YearMonth', how='inner')
    
    if len(df_merged) == 0:
        print("警告: 合併後無資料，請檢查日期格式是否一致")
        return
    
    print(f"合併後樣本期間: {df_merged['YearMonth'].min()} 至 {df_merged['YearMonth'].max()} (共 {len(df_merged)} 個月)")

    # 4. 定義迴歸模型
    # 依變數: Long_Short_Pct
    # 自變數: 各種因子組合
    
    # 因子清單 (不含 RF)
    factors_all = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'STR']
    
    models = {
        'CAPM (1 Factor)': ['Mkt_RF'],
        'FF3 (3 Factors)': ['Mkt_RF', 'SMB', 'HML'],
        'Carhart (4 Factors)': ['Mkt_RF', 'SMB', 'HML', 'MOM'],
        'FF5 (5 Factors)': ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA'],
        'FF5 + MOM (6 Factors)': ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM'],
        'All (7 Factors)': ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'STR']
    }

    results_summary = []

    y = df_merged['Long_Short_Pct']

    print("\n" + "="*80)
    print(f"{ 'Model':<25} | {'Alpha (%)':<10} | {'t-stat':<8} | {'p-value':<8} | {'Adj. R2':<8}")
    print("-" * 80)

    for model_name, factor_list in models.items():
        X = df_merged[factor_list]
        X = sm.add_constant(X) # 加上截距項 (Alpha) 
        
        model = sm.OLS(y, X).fit()
        
        alpha = model.params['const']
        t_stat = model.tvalues['const']
        p_val = model.pvalues['const']
        adj_r2 = model.rsquared_adj
        
        # 標記顯著性
        sig = ""
        if p_val < 0.01: sig = "***"
        elif p_val < 0.05: sig = "**"
        elif p_val < 0.1: sig = "*"
        
        print(f"{model_name:<25} | {alpha:7.3f} {sig:<3} | {t_stat:7.3f}  | {p_val:7.3f}  | {adj_r2:7.3f}")
        
        results_summary.append({
            'Model': model_name,
            'Alpha': alpha,
            't-stat': t_stat,
            'p-value': p_val,
            'Adj_R2': adj_r2,
            'Factors': ", ".join(factor_list)
        })

    print("-" * 80)
    print("註: Alpha 為月報酬，*** p<0.01, ** p<0.05, * p<0.1")
    
    # 5. 詳細輸出最後一個模型 (All Factors) 的係數
    print("\n【完整模型 (7 Factors) 係數分析】")
    final_model_factors = models['All (7 Factors)']
    X_final = sm.add_constant(df_merged[final_model_factors])
    final_model = sm.OLS(y, X_final).fit()
    print(final_model.summary())

if __name__ == "__main__":
    analyze_factors()
