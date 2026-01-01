import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ==========================================
# 1. 頁面設定與模型載入
# ==========================================
st.set_page_config(page_title="AI 交易戰情室 (智能倉位版)", layout="wide", page_icon="📈")

# 自訂 CSS
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stTextArea textarea { font-size: 16px; font-family: 'Consolas', monospace; }
    .stButton button { height: 50px; font-size: 18px; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
    th { text-align: center !important; }
    
    /* 狀態按鈕樣式 */
    div[data-testid="stHorizontalBlock"] button { border-radius: 20px; }
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

# 初始化 Session State
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame()
if 'processed_times' not in st.session_state:
    st.session_state.processed_times = set()
if 'position' not in st.session_state:
    st.session_state.position = "None" # None, Long, Short

# 定義清除輸入框的函數
def clear_text_area():
    st.session_state["input_area"] = ""

# ==========================================
# 2. 側邊控制欄 & 倉位回報區
# ==========================================
st.title("📈 AI 交易決策系統 (智能倉位版)")

# --- 倉位狀態控制區 (新功能) ---
st.markdown("### 🚦 您的目前倉位狀態 (請手動更新)")
col_p1, col_p2, col_p3 = st.columns(3)

def set_pos(pos):
    st.session_state.position = pos

# 根據目前狀態顯示不同顏色
btn_none_type = "primary" if st.session_state.position == "None" else "secondary"
btn_long_type = "primary" if st.session_state.position == "Long" else "secondary"
btn_short_type = "primary" if st.session_state.position == "Short" else "secondary"

with col_p1:
    st.button("⚪ 我目前空手", type=btn_none_type, use_container_width=True, on_click=set_pos, args=("None",))
with col_p2:
    st.button("🔴 我持有多單", type=btn_long_type, use_container_width=True, on_click=set_pos, args=("Long",))
with col_p3:
    st.button("🟢 我持有空單", type=btn_short_type, use_container_width=True, on_click=set_pos, args=("Short",))

st.divider()

with st.container():
    col_input, col_btns = st.columns([3, 1])
    
    with col_input:
        raw_text = st.text_area(
            "在此貼上 Excel 數據...", 
            height=120,
            placeholder="例如: 08:45  17600  17550 ... (第1欄必須是時間)",
            key="input_area" 
        )
    
    with col_btns:
        st.write("")
        run_btn = st.button("🚀 開始判讀", type="primary", use_container_width=True)
        clear_input_btn = st.button("🧹 清除輸入", on_click=clear_text_area, use_container_width=True)
        clear_hist_btn = st.button("🗑️ 清空歷史", use_container_width=True)

# ==========================================
# 3. 核心運算邏輯
# ==========================================
feature_names = [
    "BB_Upper", "BB_MA20", "BB_Lower", "MA_Slope", "BB_Width_Delta",
    "Vol_Rel", "K", "D", "Close_Pos", "Volatility", 
    "K_Rel_Strength", "Body_Ratio", "Week", "Settlement_Day", "Time_Period"
]

if clear_hist_btn:
    st.session_state.history = pd.DataFrame()
    st.session_state.processed_times = set()
    st.rerun()

if run_btn and raw_text:
    if model_long is None:
        st.error("⚠️ 模型未載入。")
    else:
        rows = raw_text.strip().split('\n')
        new_records = []
        duplicate_count = 0
        
        for i, row_str in enumerate(rows):
            try:
                vals_str = re.split(r'[\t,]+', row_str.strip())
                vals_str = [v.strip() for v in vals_str if v.strip()]
                
                if len(vals_str) < 16: continue 
                
                k_time = vals_str[0]
                if k_time in st.session_state.processed_times:
                    duplicate_count += 1
                    continue
                st.session_state.processed_times.add(k_time)
                
                try:
                    feature_vals = [float(v) for v in vals_str[1:16]]
                except ValueError: continue

                # AI 預測
                row_dict = dict(zip(feature_names, feature_vals))
                df_input = pd.DataFrame([row_dict])
                
                p_long = model_long.predict_proba(df_input)[0][1] * 100
                p_short = model_short.predict_proba(df_input)[0][1] * 100
                settlement_day = int(row_dict.get('Settlement_Day', 0))
                
                # --- 智能決策邏輯 (結合倉位狀態) ---
                current_pos = st.session_state.position
                
                signal = "觀望 ✋"
                conf = 0.0
                action = "暫無建議"
                bg_color = "#f0f2f6"
                
                # 1. 判斷多空訊號
                is_long_signal = p_long > 70
                is_short_signal = p_short > 70
                
                # 2. 根據目前倉位給建議
                if current_pos == "None": # 空手時
                    if is_long_signal:
                        signal = "做多 (LONG) 🔥"
                        conf = p_long
                        action = "進場！停損 65 點"
                        bg_color = "#fadbd8"
                    elif is_short_signal:
                        prefix = "做空 (SHORT) ⚡"
                        if p_short > 80: prefix = "重倉空 (STRONG) ⚡⚡"
                        signal = prefix
                        conf = p_short
                        action = "進場！停損 50 點"
                        bg_color = "#d5f5e3"
                    else:
                        action = "耐心等待訊號..."
                
                elif current_pos == "Long": # 持有多單時
                    if is_long_signal:
                        signal = "續抱多單 (HOLD) 🔒"
                        conf = p_long
                        action = "趨勢延續，請續抱"
                        bg_color = "#fadbd8"
                    elif is_short_signal: # 反轉訊號
                        signal = "反手做空 (REVERSE) 🔄"
                        conf = p_short
                        action = "多單出場，反手做空"
                        bg_color = "#f5b7b1" # 深紅警戒
                    else: # 訊號消失
                        signal = "多單出場 (EXIT) 🚪"
                        conf = max(p_long, p_short)
                        action = "動能減弱，建議獲利了結"
                        bg_color = "#eaecee"

                elif current_pos == "Short": # 持有空單時
                    if is_short_signal:
                        signal = "續抱空單 (HOLD) 🔒"
                        conf = p_short
                        action = "趨勢延續，請續抱"
                        bg_color = "#d5f5e3"
                    elif is_long_signal: # 反轉訊號
                        signal = "反手做多 (REVERSE) 🔄"
                        conf = p_long
                        action = "空單出場，反手做多"
                        bg_color = "#a9dfbf" # 深綠警戒
                    else: # 訊號消失
                        signal = "空單出場 (EXIT) 🚪"
                        conf = max(p_long, p_short)
                        action = "動能減弱，建議獲利了結"
                        bg_color = "#eaecee"

                # 月結算日邏輯
                if settlement_day == 2 and "進場" in action:
                     action += " (⚠️ 月結算日小心)"

                record = {
                    "K棒時間": k_time,
                    "AI 訊號": signal,
                    "信心度": f"{conf:.1f}%",
                    "操作建議": action,
                    "Color": bg_color,
                    "raw_time": k_time
                }
                new_records.append(record)
                
            except Exception: pass 

        if new_records:
            new_df = pd.DataFrame(new_records)
            st.session_state.history = pd.concat([st.session_state.history, new_df], ignore_index=True)
            
            try:
                st.session_state.history['sort_key'] = pd.to_datetime(
                    st.session_state.history['raw_time'], format='%H:%M', errors='coerce'
                )
                st.session_state.history = st.session_state.history.sort_values(
                    by=['sort_key', 'raw_time'], na_position='last'
                ).reset_index(drop=True)
            except:
                st.session_state.history = st.session_state.history.sort_values('raw_time').reset_index(drop=True)
            
            msg = f"✅ 成功新增 {len(new_records)} 筆資料！"
            if duplicate_count > 0: msg += f" (過濾 {duplicate_count} 筆重複)"
            st.success(msg)
        elif duplicate_count > 0:
            st.warning(f"⚠️ 資料未更新：貼上的資料都已存在。")

# ==========================================
# 4. 顯示結果
# ==========================================
if not st.session_state.history.empty:
    st.markdown("---")
    st.subheader("📊 即時判讀日誌")
    
    latest = st.session_state.history.iloc[-1]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("最新 K 棒時間", latest['K棒時間'])
    col_b.metric("AI 訊號", latest['AI 訊號'])
    col_c.metric("信心度", latest['信心度'])
    col_d.metric("建議", latest['操作建議'])
    
    def color_rows(row):
        return [f'background-color: {row["Color"]}; color: black; font-weight: bold' for _ in row]

    display_df = st.session_state.history[['K棒時間', 'AI 訊號', '信心度', '操作建議', 'Color']]
    st.dataframe(display_df.style.apply(color_rows, axis=1), use_container_width=True, hide_index=True)
    
else:
    st.info("👋 請先在上方選擇您的「目前倉位」，再貼上數據進行分析。")

st.markdown("---")
