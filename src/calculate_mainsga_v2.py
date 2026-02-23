import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
data_raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
data_processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

# 1. 讀取資料
print("正在讀取資料 (data/raw/industry.csv)...")
df = pd.read_csv(os.path.join(data_raw_dir, 'industry.csv'))

# 2. 資料清洗與型別轉換
df.columns = df.columns.str.strip()

# 找出所有數值欄位 (包含可能帶有逗號的)
numeric_cols = ['資產總額', '營業收入淨額', '營業費用', 
                '推銷費用', '管理費用', '研究發展費', '其他費用', 
                '歸屬母公司淨利（損）']

for col in numeric_cols:
    if col in df.columns:
        # 轉字串 -> 移除逗號 -> 轉數字 (錯誤變 NaN)
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        # 將 NaN 補 0 (針對費用類欄位，資產/營收除外)
        if col not in ['資產總額', '營業收入淨額']:
             df[col] = df[col].fillna(0)

# 移除關鍵數據缺失的列 (資產或營收為 0 或 NaN 的都不行)
df = df.dropna(subset=['資產總額', '營業收入淨額'])
df = df[df['資產總額'] > 0] # 避免除以 0

# 處理公司代碼與日期
df['Company_ID'] = df['公司'].astype(str).str.split().str[0]
df['Date'] = pd.to_datetime(df['年月'])
df['Year'] = df['Date'].dt.year
df = df.sort_values(by=['Company_ID', 'Year'])

# 建立廣義產業代碼 (前3碼)
df['Industry_Group'] = df['TEJ產業_代碼'].astype(str).str[:3]
print(f"資料筆數: {len(df)}")

# 3. 建構變數
print("正在計算 MainSG&A...")

# 根據論文定義: MainSG&A = SG&A - R&D - Advertising
# 因為我們沒有廣告費，所以 MainSG&A = SG&A - R&D
df['Raw_MainSGA'] = df['營業費用'] - df['研究發展費']

# 計算落後變數 (t-1)
df['Lag_Total_Assets'] = df.groupby('Company_ID')['資產總額'].shift(1)
df['Lag_Revenue'] = df.groupby('Company_ID')['營業收入淨額'].shift(1)

# 移除沒有上一期資料的 (2013年會被濾掉)
df_reg = df.dropna(subset=['Lag_Total_Assets', 'Lag_Revenue']).copy()

# 計算平均資產
df_reg['Avg_Total_Assets'] = (df_reg['資產總額'] + df_reg['Lag_Total_Assets']) / 2

# 標準化 (論文核心步驟)
df_reg['Scaled_MainSGA'] = df_reg['Raw_MainSGA'] / df_reg['Avg_Total_Assets']
df_reg['Scaled_Revenue'] = df_reg['營業收入淨額'] / df_reg['Avg_Total_Assets']

# 虛擬變數
df_reg['Dummy_Rev_Dec'] = (df_reg['營業收入淨額'] < df_reg['Lag_Revenue']).astype(int)
df_reg['Dummy_Loss'] = (df_reg['歸屬母公司淨利（損）'] < 0).astype(int)

# 4. 執行迴歸 (分 Industry_Group 和 Year)
print("正在執行迴歸拆解...")

results = []
r2_stats = [] # Store R2 statistics
grouped = df_reg.groupby(['Industry_Group', 'Year'])

processed_count = 0
skipped_count = 0

for (industry, year), group in grouped:
    # 樣本數 < 8 則跳過
    if len(group) < 8:
        skipped_count += len(group)
        continue
        
    try:
        # 迴歸模型: MainSGA ~ Revenue + Rev_Dec + Loss
        model = smf.ols('Scaled_MainSGA ~ Scaled_Revenue + Dummy_Rev_Dec + Dummy_Loss', data=group)
        res = model.fit()
        
        # 紀錄 R2
        r2_val = res.rsquared
        r2_stats.append({'Industry': industry, 'Year': year, 'R2': r2_val, 'N': len(group)})
        
        # 取得係數 beta1 (Revenue 的係數)
        beta1 = res.params['Scaled_Revenue']
        
        # 計算維護部分 (Maintenance) = beta1 * Revenue
        group = group.copy()
        group['Maintenance_MainSGA'] = beta1 * group['Scaled_Revenue']
        
        # 計算投資部分 (Investment) = MainSG&A - Maintenance
        group['Investment_MainSGA'] = group['Scaled_MainSGA'] - group['Maintenance_MainSGA']
        
        # 將 R2 加入資料中 (方便後續檢查)
        group['Regression_R2'] = r2_val
        
        results.append(group)
        processed_count += len(group)
        
    except Exception as e:
        print(f"Error in {industry} {year}: {e}")
        skipped_count += len(group)
        continue

# 5. 輸出結果
if results:
    final_df = pd.concat(results)
    
    # 輸出 R2 統計
    r2_df = pd.DataFrame(r2_stats)
    print("\n" + "="*50)
    print("【SG&A 拆解模型 R-squared 診斷】")
    print(r2_df['R2'].describe().round(4))
    print(f"R2 > 0.9 的群組比例: {(r2_df['R2'] > 0.9).mean()*100:.1f}%")
    print(f"R2 < 0.1 的群組比例: {(r2_df['R2'] < 0.1).mean()*100:.1f}%")
    
    print("\nR2 最低的前 5 個產業-年度:")
    print(r2_df.nsmallest(5, 'R2'))
    print("\nR2 最高的前 5 個產業-年度:")
    print(r2_df.nlargest(5, 'R2'))
    print("="*50 + "\n")

    output_cols = [
        'Company_ID', '公司', 'Year', 'Industry_Group',
        'Scaled_MainSGA', 'Maintenance_MainSGA', 'Investment_MainSGA',
        'Raw_MainSGA', 'Avg_Total_Assets', 
        '營業費用', '研究發展費', '推銷費用', '管理費用',
        'Regression_R2'
    ]
    # 只保留存在的欄位
    output_cols = [c for c in output_cols if c in final_df.columns]
    
    final_df = final_df[output_cols]
    
    output_filename = os.path.join(data_processed_dir, 'main_sga_v2_result.csv')
    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"處理完成！有效樣本: {processed_count} (跳過: {skipped_count})")
    print(f"結果已存至: {output_filename}")
    print(final_df[['Scaled_MainSGA', 'Maintenance_MainSGA', 'Investment_MainSGA']].describe())
else:
    print("沒有產生任何結果，請檢查資料。")
