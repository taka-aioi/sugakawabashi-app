import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 画面全体のデザイン設定
st.set_page_config(page_title="菅川橋 水位予測システム V3", page_icon="🌊", layout="wide")

st.title("🌊 菅川橋 水位予測システム (24時間雨量連動・現況水位あり版)")
st.markdown("AIに**長時間の雨の土台（24時間累積雨量）**と**現況水位**を両方記憶させた改良版システムです。予測誤差マージン（+0.20m）を考慮した最悪シナリオも同時に表示します。")

# 1. サイドバー：ファイルのアップロードと未来の雨量入力
st.sidebar.header("📁 データ入力")
uploaded_file = st.sidebar.file_uploader(
    "Excelファイルをここにドラッグ＆ドロップしてください", 
    type=["xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.header("🔮 これからの予測雨量 (mm)")
st.sidebar.markdown("これからの気象予報データなどを基に、各時間帯の**1時間あたりの雨量**を入力してください。")

future_rain_1h = st.sidebar.number_input("これからの1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_2h = st.sidebar.number_input("2時間後の1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_3h = st.sidebar.number_input("3時間後の1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_4h = st.sidebar.number_input("4時間後の1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_5h = st.sidebar.number_input("5時間後の1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_6h = st.sidebar.number_input("6時間後の1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

# 3つの専用AIモデル(V3)をまとめて読み込む
@st.cache_resource
def load_all_models():
    m1 = xgb.XGBRegressor()
    m1.load_model("model_1h_v3.json")
    m3 = xgb.XGBRegressor()
    m3.load_model("model_3h_v3.json")
    m6 = xgb.XGBRegressor()
    m6.load_model("model_6h_v3.json")
    return m1, m3, m6

try:
    model_1h, model_3h, model_6h = load_all_models()
except Exception as e:
    st.error(f"AIモデル(V3)の読み込みに失敗しました。GitHubに 'model_1h_v3.json'、'model_3h_v3.json'、'model_6h_v3.json' が配置されているか確認してください。")
    st.stop()

# 2. メイン処理
if uploaded_file is not None:
    try:
        # データの読み込み
        df_raw = pd.read_excel(uploaded_file, sheet_name='統合データ（メイン）', header=None)
        columns_raw = df_raw.iloc[4].tolist()
        df = df_raw.iloc[5:].copy()
        
        # 項目名のお掃除
        clean_columns = [str(c).replace('\n', '').replace('/', '').replace(' ', '').strip() for c in columns_raw]
        df.columns = clean_columns
        
        # 必要な列を数値化
        df['water_level現況水位(m)'] = pd.to_numeric(df['water_level現況水位(m)'], errors='coerce')
        df['wl_change_1h1h前からの水位変化(m)'] = pd.to_numeric(df['wl_change_1h1h前からの水位変化(m)'], errors='coerce')
        df['rainfall_3h3時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_3h3時間累積雨量(mm)'], errors='coerce')
        df['rainfall_6h6時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_6h6時間累積雨量(mm)'], errors='coerce')
        df['rainfall_24h24時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_24h24時間累積雨量(mm)'], errors='coerce')
        
        df = df.dropna(subset=['datetime（日時）', 'water_level現況水位(m)']).reset_index(drop=True)
        
        # 直近の（最後の）行を取得
        latest_row = df.iloc[-1]
        latest_time = pd.to_datetime(latest_row['datetime（日時）'])
        
        current_wl = float(latest_row['water_level現況水位(m)'])
        current_change = float(latest_row['wl_change_1h1h前からの水位変化(m)']) if not pd.isna(latest_row['wl_change_1h1h前からの水位変化(m)']) else 0.0
        current_r3 = float(latest_row['rainfall_3h3時間累積雨量(mm)'])
        current_r6 = float(latest_row['rainfall_6h6時間累積雨量(mm)'])
        current_r24 = float(latest_row['rainfall_24h24時間累積雨量(mm)'])
        
        # AIに渡す特徴量の順番（現況水位あり版）
        features_order = [
            'water_level現況水位(m)', 
            'wl_change_1h1h前からの水位変化(m)', 
            'rainfall_3h3時間累積雨量(mm)', 
            'rainfall_6h6時間累積雨量(mm)',
            'rainfall_24h24時間累積雨量(mm)'
        ]
        
        # --- 未来の予測計算 ---
        # 1時間後
        r3_1h = current_r3 + future_rain_1h
        r6_1h = current_r6 + future_rain_1h
        r24_1h = current_r24 + future_rain_1h
        X_1h = pd.DataFrame([[current_wl, current_change, r3_1h, r6_1h, r24_1h]], columns=features_order)
        rise_1h = float(model_1h.predict(X_1h)[0])
        pred_1h = current_wl + rise_1h
        
        # 3時間後
        rain_to_3h = sum([future_rain_1h, future_rain_2h, future_rain_3h])
        r3_3h = current_r3 + rain_to_3h
        r6_3h = current_r6 + rain_to_3h
        r24_3h = current_r24 + rain_to_3h
        X_3h = pd.DataFrame([[current_wl, current_change, r3_3h, r6_3h, r24_3h]], columns=features_order)
        rise_3h = float(model_3h.predict(X_3h)[0])
        pred_3h = current_wl + rise_3h
        
        # 6時間後
        total_future_rain = sum([future_rain_1h, future_rain_2h, future_rain_3h, future_rain_4h, future_rain_5h, future_rain_6h])
        r3_6h = current_r3 + total_future_rain
        r6_6h = current_r6 + total_future_rain
        r24_6h = current_r24 + total_future_rain
        X_6h = pd.DataFrame([[current_wl, current_change, r3_6h, r6_6h, r24_6h]], columns=features_order)
        rise_6h = float(model_6h.predict(X_6h)[0])
        pred_6h = current_wl + rise_6h
        
        # 時間補間処理
        pred_wl_list = [pred_1h, pred_1h + (pred_3h - pred_1h) * 0.5, pred_3h, pred_3h + (pred_6h - pred_3h) * (1/3), pred_3h + (pred_6h - pred_3h) * (2/3), pred_6h]
        
        # 安全マージン（ゆうくん指定の0.20m）
        margin_value = 0.20
        worst_wl_list = [v + margin_value for v in pred_wl_list]
        
        # 3. サマリー表示
        st.subheader("📊 未来の水位予測サマリー")
        alert_level = 0.90
        max_worst_pred = max(worst_wl_list)
        
        if max_worst_pred >= alert_level:
            st.error(f"🚨 【大雨警戒アラート】最悪シナリオ予測において、6時間以内に水防団待機水位（{alert_level:.2f}m）を上回る予測（最大 {max_worst_pred:.2f}m）が出ました！")
        else:
            st.success(f"✅ 現在のところ、最悪シナリオでも待機水位（{alert_level:.2f}m）を超える予測はありません。")
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="現在 水位", value=f"{current_wl:.2f} m")
        col2.metric(label="🔮 AI基本予測 (6時間後)", value=f"{pred_6h:.2f} m")
        col3.metric(label="🛡️ 最悪シナリオ予測 (6時間後)", value=f"{max_worst_pred:.2f} m")
        col4.metric(label="⚠️ 水防団待機水位", value=f"{alert_level:.2f} m")
        
        # 4. グラフの作成
        st.subheader("📈 水位の推移と予測値")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['datetime（日時）'], y=df['water_level現況水位(m)'], name='過去の実績水位', line=dict(color='blue', width=2)))
        
        pred_times = [latest_time + timedelta(hours=h) for h in range(1, 7)]
        fig.add_trace(go.Scatter(x=pred_times, y=pred_wl_list, name='AI基本予測 (V3モデル)', mode='markers+lines', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=pred_times, y=worst_wl_list, name='⚠️ 最悪シナリオ予測 (+0.20mマージン)', mode='markers+lines', line=dict(color='red', width=2)))
        
        fig.add_hline(y=alert_level, line_dash="dot", line_color="darkred", annotation_text=f"水防団待機水位 ({alert_level:.2f}m)")
        fig.update_layout(xaxis_title="日時", yaxis_title="水位 (m)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
else:
    st.info("💡 左側のサイドバーからデータをアップロードしてください。")
