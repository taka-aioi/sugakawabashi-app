import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="菅川橋 水位予測システム V3.1", page_icon="🌊", layout="wide")
st.title("🌊 菅川橋 水位予測システム (雨量特化×期間トリミング版)")
st.markdown("AIから水位の先入観を排除し、**純粋に雨の量と現在の勢いだけ**から未来の上昇量を計算する実戦仕様です。")

st.sidebar.header("📁 データ入力")
uploaded_file = st.sidebar.file_uploader("Excelファイルをドラッグ＆ドロップ", type=["xlsx"])

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
    m1.load_model("model_1h_v3.json")
    m3.load_model("model_3h_v3.json")
    m6.load_model("model_6h_v3.json")
    return m1, m3, m6

try:
    model_1h, model_3h, model_6h = load_all_models()
except Exception as e:
    st.error(f"AIモデルの読み込みに失敗しました。GitHubのファイルを確認してください。")
    st.stop()

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file, sheet_name='統合データ（メイン）', header=None)
        columns_raw = df_raw.iloc[4].tolist()
        df = df_raw.iloc[5:].copy()
        clean_columns = [str(c).replace('\n', '').replace('/', '').replace(' ', '').strip() for c in columns_raw]
        df.columns = clean_columns
        
        df['water_level現況水位(m)'] = pd.to_numeric(df['water_level現況水位(m)'], errors='coerce')
        df['wl_change_1h1h前からの水位変化(m)'] = pd.to_numeric(df['wl_change_1h1h前からの水位変化(m)'], errors='coerce')
        df['rainfall_3h3時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_3h3時間累積雨量(mm)'], errors='coerce')
        df['rainfall_6h6時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_6h6時間累積雨量(mm)'], errors='coerce')
        df['rainfall_24h24時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_24h24時間累積雨量(mm)'], errors='coerce')
        
        df = df.dropna(subset=['datetime（日時）', 'water_level現況水位(m)']).reset_index(drop=True)
        
        latest_row = df.iloc[-1]
        latest_time = pd.to_datetime(latest_row['datetime（日時）'])
        current_wl = float(latest_row['water_level現況水位(m)'])
        current_change = float(latest_row['wl_change_1h1h前からの水位変化(m)']) if not pd.isna(latest_row['wl_change_1h1h前からの水位変化(m)']) else 0.0
        current_r3 = float(latest_row['rainfall_3h3時間累積雨量(mm)'])
        current_r6 = float(latest_row['rainfall_6h6時間累積雨量(mm)'])
        current_r24 = float(latest_row['rainfall_24h24時間累積雨量(mm)'])
        
        # ★ 特徴量から現況水位を排除した4つの並び
        features_order = [
            'wl_change_1h1h前からの水位変化(m)', 
            'rainfall_3h3時間累積雨量(mm)', 
            'rainfall_6h6時間累積雨量(mm)',
            'rainfall_24h24時間累積雨量(mm)'
        ]
        
        # 1時間後
        X_1h = pd.DataFrame([[current_change, current_r3 + future_rain_1h, current_r6 + future_rain_1h, current_r24 + future_rain_1h]], columns=features_order)
        pred_1h = current_wl + float(model_1h.predict(X_1h)[0])
        
        # 3時間後
        rain_to_3h = sum([future_rain_1h, future_rain_2h, future_rain_3h])
        X_3h = pd.DataFrame([[current_change, current_r3 + rain_to_3h, current_r6 + rain_to_3h, current_r24 + rain_to_3h]], columns=features_order)
        pred_3h = current_wl + float(model_3h.predict(X_3h)[0])
        
        # 6時間後
        total_future_rain = sum([future_rain_1h, future_rain_2h, future_rain_3h, future_rain_4h, future_rain_5h, future_rain_6h])
        X_6h = pd.DataFrame([[current_change, current_r3 + total_future_rain, current_r6 + total_future_rain, current_r24 + total_future_rain]], columns=features_order)
        pred_6h = current_wl + float(model_6h.predict(X_6h)[0])
        
        pred_wl_list = [pred_1h, pred_1h + (pred_3h - pred_1h) * 0.5, pred_3h, pred_3h + (pred_6h - pred_3h) * (1/3), pred_3h + (pred_6h - pred_3h) * (2/3), pred_6h]
        worst_wl_list = [v + 0.20 for v in pred_wl_list]
        
        st.subheader("📊 未来の水位予測サマリー")
        alert_level = 0.90
        max_worst_pred = max(worst_wl_list)
        
        if max_worst_pred >= alert_level:
            st.error(f"🚨 【大雨警戒アラート】6時間以内に水防団待機水位を上回る予測（最大 {max_worst_pred:.2f}m）が検出されました！")
        else:
            st.success(f"✅ 現在のところ警戒水位を超える予測はありません。")
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="現在 水位", value=f"{current_wl:.2f} m")
        col2.metric(label="🔮 1時間後 (最悪)", value=f"{worst_wl_list[0]:.2f} m")
        col3.metric(label="🔮 3時間後 (最悪)", value=f"{worst_wl_list[2]:.2f} m")
        col4.metric(label="🔮 6時間後 (最悪)", value=f"{worst_wl_list[5]:.2f} m")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['datetime（日時）'], y=df['water_level現況水位(m)'], name='過去の実績水位', line=dict(color='blue', width=2)))
        pred_times = [latest_time + timedelta(hours=h) for h in range(1, 7)]
        fig.add_trace(go.Scatter(x=pred_times, y=pred_wl_list, name='AI基本予測', mode='markers+lines', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=pred_times, y=worst_wl_list, name='⚠️ 最悪シナリオ予測 (+0.20m)', mode='markers+lines', line=dict(color='red', width=2)))
        fig.add_hline(y=alert_level, line_dash="dot", line_color="darkred", annotation_text=f"水防団待機水位 ({alert_level:.2f}m)")
        fig.update_layout(xaxis_title="日時", yaxis_title="水位 (m)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"エラー: {e}")
else:
    st.info("💡 左側のサイドバーからデータをアップロードしてください。")
