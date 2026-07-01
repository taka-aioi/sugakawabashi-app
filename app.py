import streamlit as st
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

st.set_page_config(page_title="菅川橋 水位予測システム V6", page_icon="🌊", layout="wide")
st.title("🌊 菅川橋 水位予測システム (完全体 V6)")
st.markdown("時系列雨量による『大雨への追従力』と、ゆうくんの『0.1m横ばいガード』を融合させた決定版システムです。")

# --- サイドバー：入力エリア ---
st.sidebar.header("💧 現在の川の状況")
input_current_wl = st.sidebar.number_input("現在の実況水位 (m)", min_value=-1.0, max_value=5.0, value=0.69, step=0.01)
input_current_change = st.sidebar.number_input("1時間前からの水位変化量 (m)", min_value=-2.0, max_value=2.0, value=0.04, step=0.01)
input_rain_1h_ago = st.sidebar.number_input("直近1時間前〜現時点までに降った雨量 (mm)", min_value=0.0, max_value=200.0, value=3.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("🔮 これからの予測降水量 (mm)")
future_rain_1h = st.sidebar.number_input("これからの1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=3.0, step=1.0)
future_rain_3h_sum = st.sidebar.number_input("1〜3時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
future_rain_6h_sum = st.sidebar.number_input("3〜6時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=18.0, step=1.0)
future_rain_12h_sum = st.sidebar.number_input("6〜12時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=35.0, step=1.0)
future_rain_24h_sum = st.sidebar.number_input("12〜24時間後までの合計雨量 (mm)", min_value=0.0, max_value=1000.0, value=97.0, step=1.0)

# モデルの読み込み
@st.cache_resource
def load_all_models():
    m1, m3, m6, m12, m24 = xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor()
    m1.load_model("model_1h_v6.json")
    m3.load_model("model_3h_v6.json")
    m6.load_model("model_6h_v6.json")
    m12.load_model("model_12h_v6.json")
    m24.load_model("model_24h_v6.json")
    return m1, m3, m6, m12, m24

try:
    model_1h, model_3h, model_6h, model_12h, model_24h = load_all_models()
except Exception as e:
    st.error(f"AIモデル(V6)の読み込みに失敗しました。GitHubに5つの 'v6.json' ファイルがあるか確認してください。")
    st.stop()

# --- 未来予測の計算 (未来への雨量積み上げロジック) ---
try:
    features_order = [
        'water_level現況水位(m)', 'wl_change_1h1h前からの水位変化(m)', 
        'rainfall_1h1時間累積雨量(mm)', 'rainfall_3h3時間累積雨量(mm)', 
        'rainfall_6h6時間累積雨量(mm)', 'rainfall_24h24時間累積雨量(mm)',
        'rain_1h_ago1時間前の1時間雨量(mm)'
    ]

    # 1時間後予測の入力（直近1時間雨量を過去としてセット）
    X_1h = pd.DataFrame([[input_current_wl, input_current_change, future_rain_1h, future_rain_1h + input_rain_1h_ago, future_rain_1h + input_rain_1h_ago, future_rain_1h + input_rain_1h_ago, input_rain_1h_ago]], columns=features_order)
    raw_1h = input_current_wl + float(model_1h.predict(X_1h)[0])

    # 3時間後予測の入力（未来の雨量を累積に足し算していく）
    rain_3h = future_rain_3h_sum / 3.0
    X_3h = pd.DataFrame([[input_current_wl, input_current_change, rain_3h, future_rain_3h_sum, future_rain_3h_sum, future_rain_3h_sum, future_rain_1h]], columns=features_order)
    raw_3h = input_current_wl + float(model_3h.predict(X_3h)[0])

    # 6時間後予測
    rain_6h = future_rain_6h_sum / 3.0
    cum_6h = future_rain_3h_sum + future_rain_6h_sum
    X_6h = pd.DataFrame([[input_current_wl, input_current_change, rain_6h, cum_6h, cum_6h, cum_6h, rain_3h]], columns=features_order)
    raw_6h = input_current_wl + float(model_6h.predict(X_6h)[0])

    # 12時間後予測
    rain_12h = future_rain_12h_sum / 6.0
    cum_12h = cum_6h + future_rain_12h_sum
    X_12h = pd.DataFrame([[input_current_wl, input_current_change, rain_12h, rain_12h*3, cum_12h, cum_12h, rain_6h]], columns=features_order)
    raw_12h = input_current_wl + float(model_12h.predict(X_12h)[0])

    # 24時間後予測
    rain_24h = future_rain_24h_sum / 12.0
    cum_24h = cum_12h + future_rain_24h_sum
    X_24h = pd.DataFrame([[input_current_wl, input_current_change, rain_24h, rain_24h*3, rain_24h*6, cum_24h, rain_12h]], columns=features_order)
    raw_24h = input_current_wl + float(model_24h.predict(X_24h)[0])

    raw_wl_list = [raw_1h, raw_3h, raw_6h, raw_12h, raw_24h]
    pred_hours = [0, 1, 3, 6, 12, 24]
    
    # 💡 【ゆうくんの黄金ルール適用：0.1m以下なら横ばい、それ以上は引き波を活かす】
    pred_wl_list = [input_current_wl]
    for v in raw_wl_list:
        if v <= 0.10:
            pred_wl_list.append(max(0.00, min(input_current_wl, 0.10)))
        else:
            pred_wl_list.append(v)

    # 可変マージン
    worst_wl_list = [input_current_wl]
    for v in pred_wl_list[1:]:
        if v <= 0.5:
            margin = 0.05
        elif v >= 0.9:
            margin = 0.20
        else:
            margin = 0.05 + (0.20 - 0.05) * ((v - 0.5) / (0.9 - 0.5))
        worst_wl_list.append(v + margin)

    # --- 画面表示 ---
    st.subheader("📊 24時間未来予測サマリー")
    alert_level = 0.90
    max_worst_pred = max(worst_wl_list)

    if max_worst_pred >= alert_level:
        st.error(f"🚨 【大雨警戒アラート】24時間以内に水防団待機水位（{alert_level:.2f}m）を上回る予測（最大 {max_worst_pred:.2f}m）が出ました！")
    else:
        st.success(f"✅ 24時間先まで、最悪シナリオでも待機水位（{alert_level:.2f}m）を超える予測はありません。")
        
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="現在 水位", value=f"{input_current_wl:.2f} m")
    col2.metric(label="🔮 AI基本予測 (24時間後)", value=f"{pred_wl_list[-1]:.2f} m")
    col3.metric(label="🛡️ 最悪シナリオ最大値", value=f"{max_worst_pred:.2f} m")
    col4.metric(label="⚠️ 水防団待機水位", value=f"{alert_level:.2f} m")

    # グラフ化
    st.subheader("📈 これから24時間後までの水位予測カーブ")
    now_time = datetime.now()
    time_axis = [now_time + timedelta(hours=h) for h in pred_hours]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=pred_wl_list, name='AI基本予測 (V6)', mode='markers+lines', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=time_axis, y=worst_wl_list, name='⚠️ 最悪シナリオ予測 (可変マージン)', mode='markers+lines', line=dict(color='red', width=2, dash='dash')))
    fig.add_hline(y=alert_level, line_dash="dot", line_color="darkred", annotation_text=f"水防団待機水位 ({alert_level:.2f}m)")

    fig.update_layout(xaxis_title="これからの時間・日時", yaxis_title="水位 (m)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"予測計算中にエラーが発生しました: {e}")
