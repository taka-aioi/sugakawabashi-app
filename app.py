import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="菅川橋 水位予測システム V4", page_icon="🌊", layout="wide")
st.title("🌊 菅川橋 水位予測システム (時系列雨量×可変マージン V4)")
st.markdown("過去1〜2時間の雨量トレンドを考慮して時間差を読み解き、さらに現在の危険度（水位）に応じて安全マージンを自動調整する実戦特化型システムです。")

st.sidebar.header("📁 データ入力")
uploaded_file = st.sidebar.file_uploader("Excelファイルをドラッグ＆ドロップしてください", type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.header("🔮 これからの予測雨量 (mm)")
future_rain_1h = st.sidebar.number_input("これからの1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_2h = st.sidebar.number_input("2時間後の1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_3h = st.sidebar.number_input("3時間後の1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_4h = st.sidebar.number_input("4時間後の1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_5h = st.sidebar.number_input("5時間後の1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_6h = st.sidebar.number_input("6時間後の1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

@st.cache_resource
def load_all_models():
    m1, m3, m6 = xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor()
    m1.load_model("model_1h_v4.json")
    m3.load_model("model_3h_v4.json")
    m6.load_model("model_6h_v4.json")
    return m1, m3, m6

try:
    model_1h, model_3h, model_6h = load_all_models()
except Exception as e:
    st.error(f"AIモデル(V4)の読み込みに失敗しました。GitHubに新モデルが配置されているか確認してください。")
    st.stop()

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file, sheet_name='統合データ（メイン）', header=None)
        columns_raw = df_raw.iloc[4].tolist()
        df = df_raw.iloc[5:].copy()
        
        clean_columns = [str(c).replace('\n', '').replace('/', '').replace(' ', '').strip() for c in columns_raw]
        df.columns = clean_columns
        
        for col in ['water_level現況水位(m)', 'wl_change_1h1h前からの水位変化(m)', 'rainfall_1h1時間累積雨量(mm)', 'rainfall_3h3時間累積雨量(mm)', 'rainfall_6h6時間累積雨量(mm)', 'rainfall_24h24時間累積雨量(mm)']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['datetime（日時）', 'water_level現況水位(m)']).reset_index(drop=True)
        
        # 直近データと、時系列過去データの切り出し（10分刻みなので6行前、12行前）
        latest_row = df.iloc[-1]
        latest_time = pd.to_datetime(latest_row['datetime（日時）'])
        
        current_wl = float(latest_row['water_level現況水位(m)'])
        current_change = float(latest_row['wl_change_1h1h前からの水位変化(m)']) if not pd.isna(latest_row['wl_change_1h1h前からの水位変化(m)']) else 0.0
        current_r3 = float(latest_row['rainfall_3h3時間累積雨量(mm)'])
        current_r6 = float(latest_row['rainfall_6h6時間累積雨量(mm)'])
        current_r24 = float(latest_row['rainfall_24h24時間累積雨量(mm)'])
        
        # 過去エクセルから自動取得する時系列特徴量
        rain_1h_ago = float(df.iloc[-7]['rainfall_1h1時間累積雨量(mm)']) if len(df) >= 7 else float(latest_row['rainfall_1h1時間累積雨量(mm)'])
        rain_2h_ago = float(df.iloc[-13]['rainfall_1h1時間累積雨量(mm)']) if len(df) >= 13 else rain_1h_ago
        
        features_order = [
            'water_level現況水位(m)', 'wl_change_1h1h前からの水位変化(m)', 
            'rainfall_3h3時間累積雨量(mm)', 'rainfall_6h6時間累積雨量(mm)', 'rainfall_24h24時間累積雨量(mm)',
            'rain_1h_ago', 'rain_2h_ago'
        ]
        
        # --- 未来予測（過去のトレンドも乗せてAIに渡す） ---
        r3_1h = current_r3 + future_rain_1h
        r6_1h = current_r6 + future_rain_1h
        r24_1h = current_r24 + future_rain_1h
        X_1h = pd.DataFrame([[current_wl, current_change, r3_1h, r6_1h, r24_1h, rain_1h_ago, rain_2h_ago]], columns=features_order)
        pred_1h = current_wl + float(model_1h.predict(X_1h)[0])
        
        rain_to_3h = sum([future_rain_1h, future_rain_2h, future_rain_3h])
        r3_3h = current_r3 + rain_to_3h
        r6_3h = current_r6 + rain_to_3h
        r24_3h = current_r24 + rain_to_3h
        X_3h = pd.DataFrame([[current_wl, current_change, r3_3h, r6_3h, r24_3h, rain_1h_ago, rain_2h_ago]], columns=features_order)
        pred_3h = current_wl + float(model_3h.predict(X_3h)[0])
        
        total_future_rain = sum([future_rain_1h, future_rain_2h, future_rain_3h, future_rain_4h, future_rain_5h, future_rain_6h])
        r3_6h = current_r3 + total_future_rain
        r6_6h = current_r6 + total_future_rain
        r24_6h = current_r24 + total_future_rain
        X_6h = pd.DataFrame([[current_wl, current_change, r3_6h, r6_6h, r24_6h, rain_1h_ago, rain_2h_ago]], columns=features_order)
        pred_6h = current_wl + float(model_6h.predict(X_6h)[0])
        
        pred_wl_list = [pred_1h, pred_1h + (pred_3h - pred_1h) * 0.5, pred_3h, pred_3h + (pred_6h - pred_3h) * (1/3), pred_3h + (pred_6h - pred_3h) * (2/3), pred_6h]
        
        # 💡 【スマート可変マージン設計】
        # 水位が0.5m以下（安全）ならマージンは最小0.05m。0.9m（待機水位）に近づくほどマージンを最大0.20mまで自動で引き上げる
        worst_wl_list = []
        for v in pred_wl_list:
            if v <= 0.5:
                margin = 0.05
            elif v >= 0.9:
                margin = 0.20
            else:
                # 0.5m〜0.9mの間を線形補間
                margin = 0.05 + (0.20 - 0.05) * ((v - 0.5) / (0.9 - 0.5))
            worst_wl_list.append(v + margin)
            
        st.subheader("📊 未来の水位予測サマリー (V4)")
        alert_level = 0.90
        max_worst_pred = max(worst_wl_list)
        
        if max_worst_pred >= alert_level:
            st.error(f"🚨 【大雨警戒アラート】可変マージン予測により、水防団待機水位（{alert_level:.2f}m）を上回る予測（最大 {max_worst_pred:.2f}m）が出ました！")
        else:
            st.success(f"✅ 現在のところ、最悪シナリオでも待機水位（{alert_level:.2f}m）を超える予測はありません。")
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="現在 水位", value=f"{current_wl:.2f} m")
        col2.metric(label="🔮 AI基本予測 (6時間後)", value=f"{pred_6h:.2f} m")
        col3.metric(label="🛡️ 最悪シナリオ予測 (6時間後)", value=f"{max_worst_pred:.2f} m")
        col4.metric(label="⚠️ 水防団待機水位", value=f"{alert_level:.2f} m")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['datetime（日時）'], y=df['water_level現況水位(m)'], name='過去の実績水位', line=dict(color='blue', width=2)))
        pred_times = [latest_time + timedelta(hours=h) for h in range(1, 7)]
        fig.add_trace(go.Scatter(x=pred_times, y=pred_wl_list, name='AI基本予測 (V4)', mode='markers+lines', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=pred_times, y=worst_wl_list, name='⚠️ 最悪シナリオ予測 (可変マージン)', mode='markers+lines', line=dict(color='red', width=2)))
        fig.add_hline(y=alert_level, line_dash="dot", line_color="darkred", annotation_text=f"水防団待機水位 ({alert_level:.2f}m)")
        fig.update_layout(xaxis_title="日時", yaxis_title="水位 (m)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
else:
    st.info("💡 左側のサイドバーからデータをアップロードしてください。")
