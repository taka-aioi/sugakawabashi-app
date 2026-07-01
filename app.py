import streamlit as st
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

st.set_page_config(page_title="菅川橋 水位予測システム V5.3", page_icon="🌊", layout="wide")
st.title("🌊 菅川橋 水位予測システム (黄金ルール搭載版 V5.3)")
st.markdown("エクセル不要。現在の川の状況とこれからの予測雨量を入れるだけで、24時間後までの水位を予測します。")

# --- サイドバー：入力エリア ---
st.sidebar.header("💧 現在の川の状況")
input_current_wl = st.sidebar.number_input("現在の実況水位 (m)", min_value=-1.0, max_value=5.0, value=0.60, step=0.01)
input_current_change = st.sidebar.number_input("1時間前からの水位変化量 (m)", min_value=-2.0, max_value=2.0, value=0.00, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("🔮 これからの予測降水量 (mm)")
future_rain_1h = st.sidebar.number_input("これからの1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_3h_sum = st.sidebar.number_input("1〜3時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)
future_rain_6h_sum = st.sidebar.number_input("3〜6時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)
future_rain_12h_sum = st.sidebar.number_input("6〜12時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)
future_rain_24h_sum = st.sidebar.number_input("12〜24時間後までの合計雨量 (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=10.0)

# モデルの読み込み
@st.cache_resource
def load_all_models():
    m1, m3, m6, m12, m24 = xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor()
    m1.load_model("model_1h_v5.json")
    m3.load_model("model_3h_v5.json")
    m6.load_model("model_6h_v5.json")
    m12.load_model("model_12h_v5.json")
    m24.load_model("model_24h_v5.json")
    return m1, m3, m6, m12, m24

try:
    model_1h, model_3h, model_6h, model_12h, model_24h = load_all_models()
except Exception as e:
    st.error(f"AIモデルの読み込みに失敗しました。GitHubに5つの 'v5.json' ファイルがあるか確認してください。")
    st.stop()

# --- 未来予測の計算 ---
try:
    features_order = ['water_level現況水位(m)', 'wl_change_1h1h前からの水位変化(m)', 'rainfall_1h1時間累積雨量(mm)']

    # AIの生予測を計算
    X_1h = pd.DataFrame([[input_current_wl, input_current_change, future_rain_1h]], columns=features_order)
    raw_1h = input_current_wl + float(model_1h.predict(X_1h)[0])

    X_3h = pd.DataFrame([[input_current_wl, input_current_change, future_rain_3h_sum / 3.0]], columns=features_order)
    raw_3h = input_current_wl + float(model_3h.predict(X_3h)[0])

    X_6h = pd.DataFrame([[input_current_wl, input_current_change, future_rain_6h_sum / 3.0]], columns=features_order)
    raw_6h = input_current_wl + float(model_6h.predict(X_6h)[0])

    X_12h = pd.DataFrame([[input_current_wl, input_current_change, future_rain_12h_sum / 6.0]], columns=features_order)
    raw_12h = input_current_wl + float(model_12h.predict(X_12h)[0])

    X_24h = pd.DataFrame([[input_current_wl, input_current_change, future_rain_24h_sum / 12.0]], columns=features_order)
    raw_24h = input_current_wl + float(model_24h.predict(X_24h)[0])

    raw_wl_list = [raw_1h, raw_3h, raw_6h, raw_12h, raw_24h]
    pred_hours = [0, 1, 3, 6, 12, 24]
    
    # 💡 【ゆうくんの黄金ルールを適用！】
    # 基本予測の組み立て
    pred_wl_list = [input_current_wl]
    for v in raw_wl_list:
        # 予測が0.1m以下になりそうなら、通常の0.00m付近（ここでは入力された現在値、または0.0m）で横ばいガード
        if v <= 0.10:
            # 現在の水位がすでに通常の範囲（0.1m以下）なら、そのまま横ばい。
            # もし高い水位から下がってきたなら、0.10mで底打ちさせる
            pred_wl_list.append(max(0.00, min(input_current_wl, 0.10)))
        else:
            pred_wl_list.append(v)

    # 最悪シナリオ（可変マージン）の組み立て
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
    fig.add_trace(go.Scatter(x=time_axis, y=pred_wl_list, name='AI基本予測 (V5.3)', mode='markers+lines', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=time_axis, y=worst_wl_list, name='⚠️ 最悪シナリオ予測 (可変マージン)', mode='markers+lines', line=dict(color='red', width=2, dash='dash')))
    fig.add_hline(y=alert_level, line_dash="dot", line_color="darkred", annotation_text=f"水防団待機水位 ({alert_level:.2f}m)")

    fig.update_layout(xaxis_title="これからの時間・日時", yaxis_title="水位 (m)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"予測計算中にエラーが発生しました: {e}")
