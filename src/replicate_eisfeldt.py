import pandas as pd
import numpy as np
import os

def load_and_process_data():
    """
    Loads and processes raw data to construct HML and HML_INT factors.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw_dir = os.path.join(base_dir, 'data', 'raw')
    data_processed_dir = os.path.join(base_dir, 'data', 'processed')

    print("Loading data...")

    # 1. Load Financial Data (Annual)
    # Assets
    df_assets = pd.read_csv(os.path.join(data_raw_dir, 'industry.csv'))
    df_assets['Company_ID'] = df_assets['公司'].astype(str).str.split().str[0]
    df_assets['Year'] = pd.to_datetime(df_assets['年月']).dt.year
    df_assets['Total_Assets'] = pd.to_numeric(df_assets['資產總額'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Liabilities
    df_liabs = pd.read_csv(os.path.join(data_raw_dir, 'capex.csv'))
    df_liabs['Company_ID'] = df_liabs['公司'].astype(str).str.split().str[0]
    df_liabs['Year'] = pd.to_datetime(df_liabs['年月']).dt.year
    if '負債總額' in df_liabs.columns:
        df_liabs['Total_Liabilities'] = pd.to_numeric(df_liabs['負債總額'].astype(str).str.replace(',', ''), errors='coerce')
    else:
        print("Warning: '負債總額' column not found in capex.csv")
        df_liabs['Total_Liabilities'] = np.nan

    # OC Stock (from processed data)
    df_oc = pd.read_csv(os.path.join(data_processed_dir, 'org_capital_results.csv'))
    df_oc['Company_ID'] = df_oc['Company_ID'].astype(str)
    # Ensure OC Stock is numeric
    df_oc['Org_Capital_Stock_Nominal'] = pd.to_numeric(df_oc['Org_Capital_Stock_Nominal'], errors='coerce')

    # Merge Annual Data
    df_annual = pd.merge(df_assets[['Company_ID', 'Year', 'Total_Assets']], 
                         df_liabs[['Company_ID', 'Year', 'Total_Liabilities']], 
                         on=['Company_ID', 'Year'], how='inner')
    
    df_annual = pd.merge(df_annual, df_oc[['Company_ID', 'Year', 'Org_Capital_Stock_Nominal']], 
                         on=['Company_ID', 'Year'], how='left')
    
    # Fill OC with 0 if missing (conservative approach for replication)
    df_annual['Org_Capital_Stock_Nominal'] = df_annual['Org_Capital_Stock_Nominal'].fillna(0)

    # Calculate Book Equities
    df_annual['Book_Equity'] = df_annual['Total_Assets'] - df_annual['Total_Liabilities']
    df_annual['Book_Equity_INT'] = df_annual['Book_Equity'] + df_annual['Org_Capital_Stock_Nominal']

    # 2. Load Monthly Return & Market Value Data
    # We need to read this carefully as it's large. 
    # Using 'industry&return.csv'
    df_ret = pd.read_csv(os.path.join(data_raw_dir, 'industry&return.csv'), 
                         dtype={'年月': str})
    
    df_ret['Company_ID'] = df_ret['證券代碼'].astype(str).str.split().str[0]
    df_ret['Date'] = pd.to_datetime(df_ret['年月'], format='%Y%m')
    df_ret['Month_Ret'] = pd.to_numeric(df_ret['報酬率_月'], errors='coerce') # Assuming data is already in decimal based on sample check
    # Sample check earlier showed 0.18, 0.09. This looks like 18%, 9%. Usually standard raw data is %. 
    # Let's assume it's %, so divide by 100. *Correction*: Sample output was 0.182095. If that's 18%, then it's already decimal? 
    # Wait, usually daily limit is 7-10%. 0.18 is 18%. 
    # Let's check max value later. If max > 10, it's %, if max < 1, it's decimal. 
    # Actually, let's just use it as is for sorting, but for return calculation we need to be sure.
    # Looking at sample: 0.182095. If this is 18%, it's decimal representation. If it's 0.18%, it's too small. 
    # So it's likely decimal. 
    
    df_ret['MV'] = pd.to_numeric(df_ret['市值(百萬元)'].astype(str).str.replace(',', ''), errors='coerce')
    df_ret['Industry'] = df_ret['TEJ產業_代碼']

    # Filter out missing returns or MV
    df_ret = df_ret.dropna(subset=['Month_Ret', 'MV', 'Industry'])

    print("Data loaded. Processing portfolios...")
    return df_annual, df_ret

def construct_factors(df_annual, df_ret):
    """
    Constructs Traditional HML and Intangible HML factors.
    """
    # Create a mapping for annual data to monthly data
    # Fama-French methodology:
    # At June of Year t, use Book Equity from Year t-1 and Market Equity from Dec Year t-1.
    
    # 1. Prepare Sorting Variables
    # We need ME from December for the denominator of B/M
    df_ret['Year'] = df_ret['Date'].dt.year
    df_ret['Month'] = df_ret['Date'].dt.month
    
    df_dec_me = df_ret[df_ret['Month'] == 12][['Company_ID', 'Year', 'MV']].rename(columns={'MV': 'ME_Dec'})
    
    # Merge ME_Dec to Annual Data (Year t-1)
    # We want BE(t-1) and ME(Dec, t-1) to form portfolios for June(t) to May(t+1)
    df_sorting = pd.merge(df_annual, df_dec_me, on=['Company_ID', 'Year'], how='inner')
    
    # Calculate Ratios
    df_sorting['BM_Ratio'] = df_sorting['Book_Equity'] / df_sorting['ME_Dec']
    df_sorting['BM_INT_Ratio'] = df_sorting['Book_Equity_INT'] / df_sorting['ME_Dec']
    
    # 2. Merge Sorting Variables to Monthly Returns
    # We need to shift the sorting year forward by 1. 
    # i.e., Data from Year 2000 (t-1) is used for portfolios starting July 2001 (t).
    # Wait, FF standard: 
    # June of year t: Sort on BMI(t-1). Returns are July t to June t+1.
    # So, we merge Annual(Year y) to Monthly(Year y+1, Month 7-12) and Monthly(Year y+2, Month 1-6).
    
    df_sorting['Portfolio_Formation_Year'] = df_sorting['Year'] + 1
    
    # Create a linkage
    # For each month in df_ret, determine the Portfolio_Formation_Year
    # If month >= 7, PFY = Year. If month < 7, PFY = Year - 1.
    df_ret['Portfolio_Formation_Year'] = np.where(df_ret['Month'] >= 7, df_ret['Year'], df_ret['Year'] - 1)
    
    df_merged = pd.merge(df_ret, df_sorting[['Company_ID', 'Portfolio_Formation_Year', 'BM_Ratio', 'BM_INT_Ratio']], 
                         on=['Company_ID', 'Portfolio_Formation_Year'], how='inner')
    
    # 3. Form Portfolios (Within Industry)
    # We need to compute breakpoints for each Industry-Month? 
    # No, Breakpoints are computed once per year (in June), but here we can just compute them monthly for simplicity 
    # or carry forward the June breakpoints. 
    # Eisfeldt: "We sort firms into tercile buckets by B_INT/M every period within each industry."
    # "Every period" usually means rebalanced annually in June, but held for the year.
    # For simplicity and robustness, let's calculate breakpoints based on the cross-section available in that month 
    # (or strictly following FF, calculate in June and hold). 
    # Let's do the "Hold" method properly.
    
    # Group by [Portfolio_Formation_Year, Industry]
    # We only care about the sorting variable, which is constant for the PFY.
    
    # Define a function to assign portfolios
    def assign_bucket(group, col_name):
        if len(group) < 3: # Need at least a few firms to sort
            return pd.Series(index=group.index, data=np.nan)
        try:
            # 30%, 70% breakpoints
            p30 = group[col_name].quantile(0.3)
            p70 = group[col_name].quantile(0.7)
            
            buckets = pd.Series(index=group.index, data='Neutral')
            buckets[group[col_name] <= p30] = 'Low' # Growth
            buckets[group[col_name] >= p70] = 'High' # Value
            return buckets
        except:
             return pd.Series(index=group.index, data=np.nan)

    print("Sorting portfolios within industries...")
    # Unique firms per year for sorting
    df_unique_sort = df_merged[['Portfolio_Formation_Year', 'Industry', 'Company_ID', 'BM_Ratio', 'BM_INT_Ratio']].drop_duplicates()
    
    # Apply sorting
    df_unique_sort['Port_Trad'] = (df_unique_sort.groupby(['Portfolio_Formation_Year', 'Industry'])
                                    .apply(lambda x: assign_bucket(x, 'BM_Ratio')).reset_index(level=[0,1], drop=True))
    
    df_unique_sort['Port_Int'] = (df_unique_sort.groupby(['Portfolio_Formation_Year', 'Industry'])
                                   .apply(lambda x: assign_bucket(x, 'BM_INT_Ratio')).reset_index(level=[0,1], drop=True))

    # Merge back to monthly data
    df_final = pd.merge(df_merged, df_unique_sort[['Portfolio_Formation_Year', 'Company_ID', 'Port_Trad', 'Port_Int']],
                        on=['Portfolio_Formation_Year', 'Company_ID'], how='left')
    
    # 4. Calculate Returns
    # Value-weighted returns
    # Weight should be Market Cap from previous month (Lagged MV).
    # Since we don't have lagged MV easily, we can use current MV as approximation or shift it.
    # Let's shift MV.
    df_final.sort_values(['Company_ID', 'Date'], inplace=True)
    df_final['Lag_MV'] = df_final.groupby('Company_ID')['MV'].shift(1)
    
    # Drop rows with no weight
    df_final = df_final.dropna(subset=['Lag_MV', 'Month_Ret', 'Port_Trad', 'Port_Int'])

    # Function to calculate VW return
    def calc_vw_ret(df, port_col):
        # Group by Date and Portfolio
        stats = df.groupby(['Date', port_col]).apply(
            lambda x: np.average(x['Month_Ret'], weights=x['Lag_MV'])
        ).reset_index(name='Ret')
        return stats

    print("Calculating factor returns...")
    trad_rets = calc_vw_ret(df_final, 'Port_Trad')
    int_rets = calc_vw_ret(df_final, 'Port_Int')
    
    # Pivot to get High and Low columns
    trad_pivot = trad_rets.pivot(index='Date', columns='Port_Trad', values='Ret')
    int_pivot = int_rets.pivot(index='Date', columns='Port_Int', values='Ret')
    
    trad_pivot['HML_Trad'] = trad_pivot['High'] - trad_pivot['Low']
    int_pivot['HML_Int'] = int_pivot['High'] - int_pivot['Low']
    
    return trad_pivot, int_pivot

def analyze_performance(trad, int_fac):
    """
    Analyzes and saves performance metrics.
    """
    # Combine
    combined = pd.DataFrame({
        'HML_Traditional': trad['HML_Trad'],
        'HML_Intangible': int_fac['HML_Int']
    }).dropna()
    
    # Cumulative Returns
    combined['Cum_Trad'] = (1 + combined['HML_Traditional']).cumprod()
    combined['Cum_Int'] = (1 + combined['HML_Intangible']).cumprod()
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'data', 'processed', 'eisfeldt_replication.csv')
    combined.to_csv(output_path)
    
    # Stats
    # Annualized Return
    ann_ret = combined[['HML_Traditional', 'HML_Intangible']].mean() * 12
    # Annualized Volatility
    ann_vol = combined[['HML_Traditional', 'HML_Intangible']].std() * np.sqrt(12)
    # Sharpe Ratio (Assuming RF=0 for simplicity comparison)
    sharpe = ann_ret / ann_vol
    
    print("\n" + "="*40)
    print("Replication Results (Eisfeldt et al., 2021)")
    print("="*40)
    print(f"Sample Period: {combined.index.min().date()} to {combined.index.max().date()}")
    print("-" * 40)
    print("Annualized Return:")
    print(ann_ret)
    print("-" * 40)
    print("Annualized Volatility:")
    print(ann_vol)
    print("-" * 40)
    print("Sharpe Ratio:")
    print(sharpe)
    print("="*40)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    df_annual, df_ret = load_and_process_data()
    trad_pivot, int_pivot = construct_factors(df_annual, df_ret)
    analyze_performance(trad_pivot, int_pivot)
