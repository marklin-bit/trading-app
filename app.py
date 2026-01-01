# ==========================================
# V12 終極特徵工程版：自動生成反轉特徵
# ==========================================
!pip install scikit-learn pandas numpy

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib 
from google.colab import files

# 1. 讀取資料
file_path = '模型訓練.csv'
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except:
    try:
        df = pd.read_csv(file_path, encoding='cp950')
    except:
        df = pd.read_csv(file_path, encoding='big5')

# 欄位對應
column_mapping = {
    '收盤價': 'Close', '開盤價': 'Open', '最高價': 'High', '最低價': 'Low',
    '布林上通道\nUB2.00': 'BB_Upper', 'BBandMA20': 'BB_MA20', '布林下通道\nLB2.00': 'BB_Lower',
    'MA斜率\n0平/1上/2下': 'MA_Slope',  
    '布林帶寬度變化率': 'BB_Width_Delta', '相對成交量': 'Vol_Rel',
    'K(36,3)': 'K', 'D(36,3)': 'D', '收盤時\n通道位置': 'Close_Pos', '波動率': 'Volatility',
    'K 棒\n相對強度': 'K_Rel_Strength', '實體佔比': 'Body_Ratio', 'Week': 'Week',
    '結算日\n(0/1周結算/2月結算)': 'Settlement_Day', '時段\n(0盤初/1盤中/2盤尾)': 'Time_Period',
    '單別\n1多單/2空單': 'Type', '動作\n0無/1買進/2持單/3賣出': 'Action',
    '交易序號': 'Transaction_ID', '-1賠/0持平/1賺': 'Y_Outcome', '獲利強度\n0弱/1輕/2中/3強': 'Profit_Strength'
}
df.rename(columns=column_mapping, inplace=True)

# 2. [關鍵] 自動生成新特徵 (Feature Engineering)
# 乖離率 (Bias): 價格偏離均線的程度
df['Bias'] = (df['Close'] - df['BB_MA20']) / df['BB_MA20'] * 100

# K棒實體力度: (收-開)/開
df['Candle_Force'] = (df['Close'] - df['Open']) / df['Open'] * 100

# 上影線長度 (判斷壓力)
df['Upper_Shadow'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / df['Open'] * 100

# 下影線長度 (判斷支撐)
df['Lower_Shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / df['Open'] * 100

# MA斜率修正
if 'MA_Slope' in df.columns:
    df['MA_Slope'] = df['MA_Slope'].replace(2, -1)

for col in ['BB_Width_Delta', 'Vol_Rel', 'Body_Ratio']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)

# 3. 標記獲利單
sell_rows = df[df['Action'] == 3][['Transaction_ID', 'Y_Outcome', 'Profit_Strength']]
sell_rows.rename(columns={'Y_Outcome': 'Final_Outcome', 'Profit_Strength': 'Final_Strength'}, inplace=True)
sell_rows = sell_rows.drop_duplicates(subset=['Transaction_ID'], keep='last')
df = df.merge(sell_rows, on='Transaction_ID', how='left')
df['Final_Outcome'] = df['Final_Outcome'].fillna(0)
df['Final_Strength'] = df['Final_Strength'].fillna(0)

df['Target_Profit'] = 0
mask_profit = (df['Action'] == 1) & (df['Final_Outcome'] == 1) & (df['Final_Strength'] >= 1)
df.loc[mask_profit, 'Target_Profit'] = 1

# 4. 訓練包含新特徵的模型
def train_enhanced_model(data, filename):
    # 加入新特徵
    features = [
        'BB_Upper', 'BB_MA20', 'BB_Lower', 'MA_Slope', 'BB_Width_Delta',
        'Vol_Rel', 'K', 'D', 'Close_Pos', 'Volatility', 'K_Rel_Strength',
        'Body_Ratio', 'Week', 'Settlement_Day', 'Time_Period',
        'Bias', 'Candle_Force', 'Upper_Shadow', 'Lower_Shadow' # 新朋友
    ]
    
    X = data[features]
    y = data['Target_Profit']
    
    if len(y.unique()) < 2: return

    # 使用平衡權重
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y, sample_weight=sample_weights)
    
    joblib.dump(model, filename)
    files.download(filename)
    print(f"已下載強化版模型: {filename}")

train_enhanced_model(df[df['Type'] == 1].copy(), 'model_long.pkl')
train_enhanced_model(df[df['Type'] == 2].copy(), 'model_short.pkl')
