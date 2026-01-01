import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ==========================================
# 1. 頁面設定與模型載入
# ==========================================
st.set_page_config(page_title="AI 交易戰情室", layout="wide", page_icon="📈")

# 自訂 CSS
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stTextArea textarea { font-size: 16px; font-family: 'Consolas', monospace; }
    .stButton button { height: 60px; font-size: 20px; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        m_long = joblib.load("model_long.pkl")
        m_short = joblib.load("model_short.pkl")
        return m_long, m_short
    except FileNotFoundError:
        st.error("❌ 找不到模型檔案！請確認 model_long.pkl 與 model_short.pkl 是否已上傳。")
        return None, None

model_long, model_short = load_models()

# 初始化歷史紀錄
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame()
# 用來記錄已經處理過的「時間」，防止重複貼上
if 'processed_times' not in st.session_state:
    st.session_state.processed_times = set()

# ==========================================
# 2. 側邊控制欄
# ==========================================
st.title("📈 AI 交易決策系統 (時間同步版)")

with st.container():
    st.markdown("### 📋 數據輸入中心")
    st.info("請複製 Excel 整列數據 (含A欄時間)。系統會依據「時間」自動過濾重複資料。")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        raw_text = st.text_area(
            "在此貼上數據...", 
            height=120,
            placeholder="例如: 08:45  17600  17550 ... (第1欄必須是時間)"
        )
    
    with col2:
        st.write("") 
        st.write("") 
        run_btn = st.button("🚀 開始判讀", type="primary", use_container_width=True)
        clear_btn = st.button("🗑️ 清空紀錄", use_container_width=True)

# ==========================================
# 3. 核心運算邏輯
# ==========================================
# 這是第 2 欄到第 16 欄的特徵名稱 (共 15 個)
feature_names = [
    "BB_Upper", "BB_MA20", "BB_Lower", "MA_Slope", "BB_Width_Delta",
    "Vol_Rel", "K", "D", "Close_Pos", "Volatility", 
    "K_Rel_Strength", "Body_Ratio", "Week", "Settlement_Day", "Time_Period"
]

if clear_btn:
    st.session_state.history = pd.DataFrame()
    st.session_state.processed_times = set()
    st.rerun()

if run_btn and raw_text:
    if model_long is None:
        st.error("⚠️ 模型未載入，無法執行。")
    else:
        rows = raw_text.strip().split('\n')
        new_records = []
        duplicate_count = 0
        
        for i, row_str in enumerate(rows):
            try:
                # 切割數據 (支援 Tab 或 逗號)
                vals_str = re.split(r'[\t,]+', row_str.strip())
                # 過濾掉空字串
                vals_str = [v.strip() for v in vals_str if v.strip()]
                
                # 檢查欄位數量：至少要有 1 (時間) + 15 (特徵) = 16 欄
                if len(vals_str) < 16:
                    continue 
                
                # 第 1 欄 (Index 0) 是時間
                k_time = vals_str[0]
                
                # 第 2~16 欄 (Index 1~15) 是數值特徵
                try:
                    feature_vals = [float(v) for v in vals_str[1:16]]
                except ValueError:
                    st.warning(f"第 {i+1} 行數據格式錯誤，請確認從 B 欄開始都是數字。")
                    continue

                # --- 防呆機制 (依時間去重) ---
                # 如果這個時間點已經處理過，就跳過
                if k_time in st.session_state.processed_times:
                    duplicate_count += 1
                    continue
                
                # 標記此時間已處理
                st.session_state.processed_times.add(k_time)
                
                # --- 建立輸入資料表 ---
                row_dict = dict(zip(feature_names, feature_vals))
                df_input = pd.DataFrame([row_dict])
                
                # --- AI 預測 ---
                p_long = model_long.predict_proba(df_input)[0][1] * 100
                p_short = model_short.predict_proba(df_input)[0][1] * 100
                settlement_day = int(row_dict.get('Settlement_Day', 0))
                
                # --- 決策邏輯 ---
                signal = "觀望 ✋"
                conf = 0.0
                action = "暫無建議"
                bg_color = "#f0f2f6"
                
                if p_long > 70:
                    signal = "做多 (LONG) 🔥"
                    conf = p_long
                    action = "停損 65 點"
                    bg_color = "#fadbd8"
                elif p_short > 70:
                    prefix = "做空 (SHORT) ⚡"
                    if p_short > 80: prefix = "重倉空 (STRONG) ⚡⚡"
                    signal = prefix
                    conf = p_short
                    action = "停損 50 點"
                    bg_color = "#d5f5e3"
                
                if settlement_day == 2:
                    action += " | ⚠️ 月結算日"
                    if conf < 80 and "觀望" not in signal:
                        action += " (小心洗盤)"

                # --- 紀錄結果 ---
                record = {
                    "K棒時間": k_time,  # 這是 Excel A 欄的時間
                    "AI 訊號": signal,
                    "信心度": f"{conf:.1f}%",
                    "操作建議": action,
                    "Color": bg_color
                }
                new_records.append(record)
                
            except Exception:
                pass 

        # --- 更新歷史紀錄 ---
        if new_records:
            new_df = pd.DataFrame(new_records)
            # 將新資料合併到最上方
            st.session_state.history = pd.concat([new_df, st.session_state.history], ignore_index=True)
            
            msg = f"✅ 成功新增 {len(new_records)} 筆 K 棒資料！"
            if duplicate_count > 0:
                msg += f" (已自動過濾 {duplicate_count} 筆舊資料)"
            st.success(msg)
        elif duplicate_count > 0:
            st.warning(f"⚠️ 資料未更新：貼上的 {duplicate_count} 筆資料都已經存在。")

# ==========================================
# 4. 顯示結果儀表板
# ==========================================
if not st.session_state.history.empty:
    st.markdown("---")
    st.subheader("📊 即時判讀日誌")
    
    # 顯示最新一筆
    latest = st.session_state.history.iloc[0]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("K棒時間", latest['K棒時間'])
    col_b.metric("最新訊號", latest['AI 訊號'])
    col_c.metric("信心度", latest['信心度'])
    col_d.metric("建議", latest['操作建議'])
    
    st.write("")
    
    # 詳細表格
    def color_rows(row):
        return [f'background-color: {row["Color"]}; color: black; font-weight: bold' for _ in row]

    display_df = st.session_state.history[['K棒時間', 'AI 訊號', '信心度', '操作建議', 'Color']]
    st.dataframe(
        display_df.style.apply(color_rows, axis=1),
        use_container_width=True,
        hide_index=True,
        column_config={"Color": None}
    )
    
else:
    st.info("👋 等待數據中... 請從 Excel 複製包含「時間 (A欄)」的整列資料貼上。")

st.markdown("---")
