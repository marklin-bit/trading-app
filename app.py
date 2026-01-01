import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.set_page_config(page_title="AI 交易戰情室 (V15 欄位鎖定版)", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px; font-family: 'Consolas', monospace; }
    .stButton button { height: 50px; font-size: 18px; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
    th { text-align: center !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        m_long = joblib.load("model_long.pkl")
        m_short = joblib.load("model_short.pkl")
        return m_long, m_short
    except: return None, None

model_long, model_short = load_models()

if 'history' not in st.session_state: st.session_state.history = pd.DataFrame()
if 'processed_times' not in st.session_state: st.session_state.processed_times = set()
if 'position' not in st.session_state: st.session_state.position = "None"

def clear_text_area(): st.session_state["input_area"] = ""

st.title("📈 AI 交易決策系統 (V15 欄位鎖定版)")

col1, col2, col3 = st.columns(3)
def set_pos(p): st.session_state.position = p
col1.button("⚪ 空手", on_click=set_pos, args=("None",), use_container_width=True, type="primary" if st.session_state.position=="None" else "secondary")
col2.button("🔴 多單", on_click=set_pos, args=("Long",), use_container_width=True, type="primary" if st.session_state.position=="Long" else "secondary")
col3.button("🟢 空單", on_click=set_pos, args=("Short",), use_container_width=True, type="primary" if st.session_state.position=="Short" else "secondary")

st.divider()

col_in, col_btn = st.columns([3, 1])
raw_text = col_in.text_area("貼上數據 (含A欄時間)...", height=120, key="input_area")
col_btn.write("")
run_btn = col_btn.button("🚀 開始判讀", type="primary", use_container_width=True)
col_btn.button("🧹 清除輸入", on_click=clear_text_area, use_container_width=True)
if col_btn.button("🗑️ 清空歷史", use_container_width=True):
    st.session_state.history = pd.DataFrame()
    st.session_state.processed_times = set()
    st.rerun()

# [關鍵] 這裡的順序必須跟 Colab V15 訓練時的一模一樣！
feature_names = [
    "BB_Upper", "BB_MA20", "BB_Lower", "MA_Slope", "BB_Width_Delta",
    "Vol_Rel", "K", "D", "Close_Pos", "Volatility", 
    "K_Rel_Strength", "Body_Ratio", "Week", "Settlement_Day", "Time_Period"
]

if run_btn and raw_text and model_long:
    rows = raw_text.strip().split('\n')
    new_records = []
    
    for row_str in rows:
        try:
            vals = [v.strip() for v in re.split(r'[\t,]+', row_str.strip()) if v.strip()]
            if len(vals) < 16: continue # 1(時間) + 15(特徵)
            
            k_time = vals[0]
            if k_time in st.session_state.processed_times: continue
            st.session_state.processed_times.add(k_time)
            
            feats = [float(v) for v in vals[1:16]]
            row_dict = dict(zip(feature_names, feats))
            
            # --- V15: 計算 Bias (第16個特徵) ---
            bb_h, bb_l, ma, pos = row_dict['BB_Upper'], row_dict['BB_Lower'], row_dict['BB_MA20'], row_dict['Close_Pos']
            approx_close = bb_l + (bb_h - bb_l) * pos
            bias = (approx_close - ma) / ma * 100 if ma != 0 else 0
            row_dict['Bias'] = bias
            
            # 斜率防呆
            if row_dict['MA_Slope'] == 2: row_dict['MA_Slope'] = -1
            
            # 建立輸入 DataFrame (確保16個欄位順序正確)
            df_input = pd.DataFrame([row_dict])[feature_names + ['Bias']]
            
            # 預測
            p_long = model_long.predict_proba(df_input)[0][1] * 100
            p_short = model_short.predict_proba(df_input)[0][1] * 100
            
            # 決策
            pos_now = st.session_state.position
            signal, action, bg_color = "觀望 ✋", "等待訊號", "#f0f2f6"
            
            # 門檻調回 50，因為這是平衡權重後的機率，不需要太高
            if p_long > 50:
                signal = "做多 (LONG) 🔥"
                action = "進場！" if pos_now == "None" else "多單續抱"
                bg_color = "#fadbd8"
            elif p_short > 50:
                signal = "做空 (SHORT) ⚡"
                action = "進場！" if pos_now == "None" else "空單續抱"
                bg_color = "#d5f5e3"
                
            new_records.append({
                "時間": k_time, "訊號": signal, "信心": f"{max(p_long, p_short):.1f}%",
                "乖離率": f"{bias:.2f}%", "建議": action, "Color": bg_color, "raw": k_time
            })
        except: pass

    if new_records:
        new_df = pd.DataFrame(new_records)
        st.session_state.history = pd.concat([st.session_state.history, new_df], ignore_index=True)
        try:
            st.session_state.history['sort'] = pd.to_datetime(st.session_state.history['raw'], format='%H:%M', errors='coerce')
            st.session_state.history = st.session_state.history.sort_values(by=['sort', 'raw']).reset_index(drop=True)
        except: pass
        st.success(f"已更新 {len(new_records)} 筆")

if not st.session_state.history.empty:
    st.markdown("---")
    def color_rows(row): return [f'background-color: {row["Color"]}; color: black' for _ in row]
    cols = ['時間', '訊號', '信心', '乖離率', '建議', 'Color']
    st.dataframe(st.session_state.history[cols].style.apply(color_rows, axis=1), use_container_width=True, hide_index=True, column_config={"Color": None})
else:
    st.info("V15 系統就緒。已校正欄位對應與多空權重。")
