# Organizational Capital and Asset Prices in Taiwan Stock Market
## 專案概述 (Project Overview)

本研究旨在探討台灣股市中，組織資本 (Organizational Capital, OC) 是否為被定價的風險因子。不同於傳統使用整體 SG&A 費用作為代理變數，本研究採用 **Main SG&A (Selling, General, and Administrative expenses excluding R&D)**，並進一步將其拆解為「維護支出 (Maintenance)」與「投資支出 (Investment)」兩部分。

研究發現，Main SG&A 中的投資成分具有顯著的資產定價意涵，高組織資本投資的公司未來具有較高的預期報酬，且此溢酬無法被傳統因子模型完全解釋。

---

## 研究方法 (Methodology)

### 1. SG&A 拆解 (Decomposition)
參考 Enache and Srivastava (2018) 與 Eisfeldt and Papanikolaou (2013) 的方法，將 Main SG&A 迴歸於當期營收及虛擬變數，分離出：
- **Maintenance Component**: 維持現有運營所需的費用。
- **Investment Component**: 創造未來長期價值的組織資本投資。

### 2. 組織資本存量建構 (Capital Stock Accumulation)
使用永續盤存法 (Perpetual Inventory Method, PIM) 將每年的投資流量累積為存量：
$$ OC_{i,t} = (1 - \delta_O) OC_{i,t-1} + \frac{Investment_{i,t}}{CPI_t} $$
其中折舊率 $\delta_O$ 設定為 15%。

### 3. 投資組合分析 (Portfolio Sorts)
- **排序變數**: 組織資本投資強度 (Investment / Assets) 或 組織資本存量強度 (OC Stock / Assets)。
- **分組方式**: 產業內五分位分組 (Within-Industry Quintile Sorts) 以控制產業特徵差異。
- **績效衡量**: 計算 Value-Weighted Monthly Returns，並使用 CAPM, Fama-French 3-Factor, 5-Factor 模型檢驗 Alpha。

---

## 實證結果摘要 (Preliminary Results)

1.  **投資效益 (Future Benefits)**:
    - 迴歸分析顯示，Main SG&A Investment 與未來盈餘成長 (Future Earnings Change) 顯著正相關。
    - 該關聯性強於 Capex (實體資本投資)，顯示無形資產在現代企業價值創造中的核心地位。

2.  **價值相關性與穩健性 (Value Relevance & Robustness - Tobin's Q)**:
    - **基本模型**: 全樣本下組織資本存量 (OC Stock) 與 Tobin's Q 顯著正相關 (Coef = 0.161, t=6.60)。
    - **固定效果 (Fixed Effects)**: 加入產業與年度固定效果後，關聯性依然顯著 (Coef = 0.166, t=6.89)，排除產業特徵與總體趨勢的影響。
    - **動態面板模型 (Dynamic Panel)**: 加入 **Tobin's Q 落後期 ($Q_{t-1}$)** 以控制估值持續性。
        - **全樣本**: OC 係數仍顯著為正 (Coef = 0.064, t=4.51)，證明 OC 提供超出過去估值的增量價值。
        - **規模差異**: **小型股 (t=7.48)** 的 OC 價值敏感度顯著高於 **大型股 (t=2.43)**。
    - **解讀**: 對於大型股，OC 價值已被市場正確反映在價格中；對於小型股，市場雖然認可其價值，但反應仍未完全 (Under-reaction)，這解釋了為何小型股未來仍有顯著超額報酬。

3.  **風險屬性 (Risk Implications)**:
    - Main SG&A Investment 與未來盈餘波動度 (Earnings Volatility) 正相關，支持其作為風險因子的論點。

4.  **資產定價 (Asset Pricing) - 基於 OC 存量 (Stock)**:
    - **方法更新**: 納入 CPI 平減以計算實質投資，並將投資組合重組時間提前至 **5 月底** (反映 3/31 財報公布)。
    - **全樣本分析**: 
        - **產業內排序 (Within-Industry)**: High-Low 策略平均月報酬為 0.039% (t=0.10)，統計上不顯著。
        - **全市場排序 (Universe Sort)**: High-Low 策略平均月報酬為 -0.203% (t=-0.54)，亦不顯著。
    - **雙重排序 (Double Sorts)**: 發現 OC 溢酬具有高度的 **規模效應 (Size Effect)**。
        - **小型股 (Small Firms)**: High-Low 策略產生顯著的平均月報酬 **0.499%** (t-stat = **3.47**)。
        - **大型股 (Big Firms)**: High-Low 策略平均月報酬僅 0.112% (t-stat = 0.28)，效果消失。
    - **因子解釋**: 在全樣本下，CAPM Alpha 為 0.56% (t=1.84)，但加入 Size 與 Value 因子後 (FF3, FF5)，Alpha 變得不顯著，這與 Double Sort 的結果一致。
    - **穩健性測試 (Robustness)**: 將投資組合重組時間提前至 **4 月底**，小型股的 OC 溢酬依然顯著 (平均月報酬 0.469%, t-stat = 3.25)。

---

## 論文實證架構建議 (Thesis Empirical Outline)

若要將此研究發展為碩士論文，建議依據以下架構整理實證結果表格與圖表：

### 1. 資料描述 (Data Description)
*   **Table 1: 樣本篩選與產業分布 (Sample Selection & Industry Distribution)**
    *   列出各產業代碼 (TEJ) 的公司家數占比。
    *   展示 SG&A 拆解模型在各產業的平均 $R^2$，證明拆解方法的適用性。
*   **Table 2: 敘述性統計 (Descriptive Statistics)**
    *   主要變數 (Tobin's Q, OC/Assets, SG&A/Assets, Size, BM, Leverage) 的平均值、標準差、分位數。
*   **Table 3: 相關係數矩陣 (Correlation Matrix)**
    *   檢視 OC 指標與其他控制變數 (如 R&D, Size) 的相關性，確認無嚴重共線性。

### 2. 組織資本指標驗證 (Validation of OC Measure)
*   **Table 4: 組織資本與未來盈餘 (Future Earnings)**
    *   迴歸分析：`Future Earnings Change ~ OC Investment + Controls`。
    *   目的：證明 Main SG&A 具有投資性質，能帶來未來效益。
*   **Table 5: 組織資本與風險 (Risk Implications)**
    *   迴歸分析：`Earnings Volatility ~ OC Investment + Controls`。
    *   目的：證明 OC 投資伴隨著較高的經營風險。
*   **Table 6: 價值相關性 (Value Relevance - Tobin's Q)**
    *   **Panel A**: OLS 迴歸結果 (Full Sample)。
    *   **Panel B**: 動態面板模型與固定效果 (Dynamic Panel & Fixed Effects)。
    *   **Panel C**: 規模分組 (Small vs. Big)。
    *   目的：證明 OC 存量能解釋企業估值差異，且結果具穩健性。

### 3. 資產定價檢定 (Asset Pricing Tests)
*   **Table 7: 單變數排序投資組合 (Single Sort Portfolios)**
    *   報告 Q1~Q5 各組的平均報酬、High-Low 策略報酬與 t-stat。
    *   包含「產業內排序」與「全市場排序」的比較，證明單純排序效果不顯著。
*   **Table 8: 雙重排序投資組合 (Double Sorts: Size x OC)**
    *   **關鍵表格**：展示在 Small Firms 與 Big Firms 下，High-Low OC 策略的績效差異。
    *   突顯 OC 溢酬集中於小型股的發現。
*   **Table 9: 因子模型迴歸 (Factor Regressions)**
    *   針對 Small Firm High-Low 策略，跑 CAPM, FF3, FF5 模型。
    *   檢驗 Alpha 是否顯著，以及是否被 Size (SMB) 或 Value (HML) 因子解釋。
*   **Figure 1: 累積報酬率走勢圖 (Cumulative Returns)**
    *   繪製 Small-High OC, Small-Low OC, Big-High OC, Big-Low OC 四條線的累積財富曲線。
    *   直觀展示小型高 OC 股的長期優勢。

### 4. 穩健性測試 (Robustness Checks)
*   **Table 10: 不同重組時間的敏感度 (Rebalancing Timing)**
    *   比較 4 月、5 月、6 月重組下的 High-Low 績效，證明結果不依賴特定月份。
*   **Table 11: 不同折舊率的敏感度 (Depreciation Rate)**
    *   嘗試使用 $\delta=0.10$ 或 $\delta=0.20$ 計算 OC 存量，檢驗結果是否穩健 (可選)。

## 檔案結構 (Structure)

- `src/`: Python 分析程式碼。
- `data/`: 原始與處理後數據。
- `notebooks/`: 簡報與視覺化圖表。
- `paper/`: 論文 LaTeX 原始碼。
