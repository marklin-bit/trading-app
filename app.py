import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime

# ==========================================
# 1. 網頁設定與模型載入
# ==========================================
st.set_page_config(page_title="AI 交易戰情室", layout="wide", page_icon="📈")

# 自訂 CSS 讓介面更專業
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stTextArea textarea { font-size: 16px; }
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

# 初始化歷史紀錄 (Session State)
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame()

# ==========================================
# 2. 側邊控制欄 & 數據輸入區
# ==========================================
st.title("📈 AI 交易決策系統 (分享版)")

with st.container():
    st.markdown("### 📋 數據輸入中心")
    st.info("請直接從 Excel 複製整列數據 (Ctrl+C)，點擊下方文字框貼上 (Ctrl+V)。支援一次貼上多筆。")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        raw_text = st.text_area(
            "在此貼上數據...", 
            height=120,
            placeholder="例如: 17600  17550  17500  1  0.05  1.5  0.8 ... (Tab 分隔)"
        )
    
    with col2:
        st.write("") # 排版用空格
        st.write("") 
        run_btn = st.button("🚀 開始判讀", type="primary", use_container_width=True)
        clear_btn = st.button("🗑️ 清空紀錄", use_container_width=True)

# ==========================================
# 3. 核心運算邏輯
# ==========================================
feature_names = [
    "BB_Upper", "BB_MA20", "BB_Lower", "MA_Slope", "BB_Width_Delta",
    "Vol_Rel", "K", "D", "Close_Pos", "Volatility", 
    "K_Rel_Strength", "Body_Ratio", "Week", "Settlement_Day", "Time_Period"
]

if clear_btn:
    st.session_state.history = pd.DataFrame()
    st.rerun()

if run_btn and raw_text:
    if model_long is None:
        st.error("⚠️ 模型未載入，無法執行。")
    else:
        # 逐行解析
        rows = raw_text.strip().split('\n')
        new_records = []
        
        for i, row_str in enumerate(rows):
            try:
                # 切割數據 (支援 Tab 或 逗號)
                vals = re.split(r'[\t,]+', row_str.strip())
                vals = [float(v) for v in vals if v.strip()]
                
                if len(vals) < 15:
                    st.warning(f"⚠️ 第 {i+1} 行數據不足 (只有 {len(vals)} 欄，需要 15 欄)，已跳過。")
                    continue
                
                # 建立輸入資料表
                row_dict = dict(zip(feature_names, vals[:15]))
                df_input = pd.DataFrame([row_dict])
                
                # AI 預測
                p_long = model_long.predict_proba(df_input)[0][1] * 100
                p_short = model_short.predict_proba(df_input)[0][1] * 100
                settlement_day = int(row_dict.get('Settlement_Day', 0))
                
                # 決策邏輯
                signal = "觀望 ✋"
                conf = 0.0
                action = "暫無建議"
                bg_color = "#f0f2f6" # 灰色
                
                if p_long > 70:
                    signal = "做多 (LONG) 🔥"
                    conf = p_long
                    action = "停損 65 點"
                    bg_color = "#fadbd8" # 淺紅
                elif p_short > 70:
                    prefix = "做空 (SHORT) ⚡"
                    if p_short > 80: prefix = "重倉空 (STRONG) ⚡⚡"
                    signal = prefix
                    conf = p_short
                    action = "停損 50 點"
                    bg_color = "#d5f5e3" # 淺綠
                
                # 月結算日邏輯
                if settlement_day == 2:
                    action += " | ⚠️ 月結算日"
                    if conf < 80 and "觀望" not in signal:
                        action += " (小心洗盤)"

                # 紀錄結果
                record = {
                    "時間": datetime.now().strftime("%H:%M:%S"),
                    "AI 訊號": signal,
                    "信心度": f"{conf:.1f}%",
                    "操作建議": action,
                    "Color": bg_color
                }
                new_records.append(record)
                
            except ValueError:
                st.error(f"❌ 第 {i+1} 行含有非數字內容，請檢查。")
            except Exception as e:
                st.error(f"❌ 第 {i+1} 行發生錯誤: {e}")

        # 更新歷史紀錄 (最新的在最上面)
        if new_records:
            new_df = pd.DataFrame(new_records)
            st.session_state.history = pd.concat([new_df, st.session_state.history], ignore_index=True)

# ==========================================
# 4. 顯示結果儀表板
# ==========================================
if not st.session_state.history.empty:
    st.markdown("---")
    st.subheader("📊 即時判讀日誌")
    
    # 顯示最新一筆的大看板
    latest = st.session_state.history.iloc[0]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("最新訊號", latest['AI 訊號'])
    col_b.metric("信心度", latest['信心度'])
    col_c.metric("建議", latest['操作建議'])
    
    st.write("")
    
    # 顯示詳細表格
    # 使用 Pandas Styler 進行條件上色
    def color_rows(row):
        return [f'background-color: {row["Color"]}; color: black; font-weight: bold' for _ in row]

    display_df = st.session_state.history[['時間', 'AI 訊號', '信心度', '操作建議', 'Color']]
    st.dataframe(
        display_df.style.apply(color_rows, axis=1),
        use_container_width=True,
        hide_index=True,
        column_config={"Color": None} # 隱藏顏色代碼欄位
    )
    
else:
    st.info("👋 目前尚無資料。請從上方貼上 Excel 數據以開始分析。")

# 頁尾說明
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>AI Model V1.0 | Powered by Streamlit & Scikit-Learn</div>", unsafe_allow_html=True)