import streamlit as st
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="菅川橋 水位予測シミュレーター V5.1", page_icon="🌊", layout="wide")
st.title("🌊 菅川橋 水位予測シミュレーター (完全手入力・24時間対応版)")
st.markdown("エクセルファイルのアップロードは不要です。現在の状態とこれからの予測雨量を入力するだけで、AIが24時間後までの推移を瞬時に計算します。")

# 1. サイドバー：現在の状況を手入力
st.sidebar.header("💧 現在の川の状況")
current_wl = st.sidebar.number_input("現在の水位 (m)", min_value=-1.00, max_value=5.00, value=0.30, step=0.01)
current_change = st.sidebar.number_input("1時間前からの水位変化 (m)", min_value=-2.00, max_value=2.00, value=0.00, step=0.01)
current_rain_1h = st.sidebar.number_input("今まさに降っている1時間雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

st.sidebar.markdown("---")

# 2. サイドバー：これからの予測雨量
st.sidebar.header("🔮 これからの予測降水量 (mm)")
future_rain_1h = st.sidebar.number_input("これからの1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_3h_sum = st.sidebar.number_input("1〜3時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)
future_rain_6h_sum = st.sidebar.number_input("3〜6時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)
future_rain_12h_sum = st.sidebar.number_input("6〜12時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)
future_rain_24h_sum = st.sidebar.number_input("12〜24時間後までの合計雨量 (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=10.0)

# AIモデル(V5)の読み込み
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
    st.error(f"AIモデル(V5)の読み込みに失敗しました。GitHubに5つの 'v5.json' ファイルが正しく配置されているか確認してください。")
    st.stop()

# 3. メイン予測処理
try:
    # AIが学習した時の特徴量の並び順
    features_order = ['water_level現況水位(m)', 'wl_change_1h1h前からの水位変化(m)', 'rainfall_1h1時間累積雨量(mm)']
    
    # 各未来の予測（手入力された現在の値をベースに計算）
    X_1h = pd.DataFrame([[current_wl, current_change, future_rain_1h]], columns=features_order)
    pred_1h = current_wl + float(model_1h.predict(X_1h)[0])
    
    X_3h = pd.DataFrame([[current_wl, current_change, future_rain_3h_sum / 3.0]], columns=features_order)
    pred_3h = current_wl + float(model_3h.predict(X_3h)[0])
    
    X_6h = pd.DataFrame([[current_wl, current_change, future_rain_6h_sum / 3.0]], columns=features_order)
    pred_6h = current_wl + float(model_6h.predict(X_6h)[0])
    
    X_12h = pd.DataFrame([[current_wl, current_change, future_rain_12h_sum / 6.0]], columns=features_order)
    pred_12h = current_wl + float(model_12h.predict(X_12h)[0])
    
    X_24h = pd.DataFrame([[current_wl, current_change, future_rain_24h_sum / 12.0]], columns=features_order)
    pred_24h = current_wl + float(model_24h.predict(X_24h)[0])
    
    # 予測結果のまとめ
    pred_hours = [0, 1, 3, 6, 12, 24]
    pred_wl_list = [current_wl, pred_1h, pred_3h, pred_6h, pred_12h, pred_24h]
    
    # 可変マージンロジック
    worst_wl_list = []
    for v in pred_wl_list:
        if v <= 0.5:
            margin = 0.05
        elif v >= 0.9:
            margin = 0.20
        else:
            margin = 0.05 + (0.20 - 0.05) * ((v - 0.5) / (0.9 - 0.5))
        worst_wl_list.append(v + margin)
        
    # サマリー表示
    st.subheader("📊 シミュレーション結果サマリー")
    alert_level = 0.90
    max_worst_pred = max(worst_wl_list[1:]) # 現在地を除いた未来の最大値
    
    if max_worst_pred >= alert_level:
        st.error(f"🚨 【大雨警戒アラート】24時間以内に水防団待機水位（{alert_level:.2f}m）を上回る予測（最大 {max_worst_pred:.2f}m）が出ました！")
    else:
        st.success(f"✅ 24時間先まで、最悪シナリオでも待機水位（{alert_level:.2f}m）を超える予測はありません。")
        
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="入力された 現在水位", value=f"{current_wl:.2f} m")
    col2.metric(label="🔮 AI基本予測 (24時間後)", value=f"{pred_24h:.2f} m")
    col3.metric(label="🛡️ 最悪シナリオ最大値", value=f"{max_worst_pred:.2f} m")
    col4.metric(label="⚠️ 水防団待機水位", value=f"{alert_level:.2f} m")
    
    # グラフ化（過去データがないので、現在時刻からの経過時間でプロット）
    st.subheader("📈 24時間後までの予測タイムライン")
    
    now = datetime.now()
    time_axis = [now + timedelta(hours=h) for h in pred_hours]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=pred_wl_list, name='AI基本予測 (V5.1)', mode='markers+lines', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=time_axis, y=worst_wl_list, name='⚠️ 最悪シナリオ予測 (可変マージン)', mode='markers+lines', line=dict(color='red', width=2, dash='dash')))
    
    fig.add_hline(y=alert_level, line_dash="dot", line_color="darkred", annotation_text=f"水防団待機水位 ({alert_level:.2f}m)")
    fig.update_layout(xaxis_title="時間（現在からの予測推移）", yaxis_title="水位 (m)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"予測計算中にエラーが発生しました: {e}")
