import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 画面全体のデザイン設定
st.set_page_config(page_title="菅川橋 水位予測システム", page_icon="🌊", layout="wide")

st.title("🌊 菅川橋 水位予測システム (トリプルAIモデル搭載・完全版)")
st.markdown("広島県防災Webのデータを基に、1h後・3h後・6h後の「3つの専用AIモデル」をトリプル駆動させ、未来の水位の変化カーブを最もリアルにシミュレーションします。")

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

# 3つの専用AIモデルをまとめて読み込む
@st.cache_resource
def load_all_models():
    m1 = xgb.XGBRegressor()
    m1.load_model("model_1h_v2.json")
    
    m3 = xgb.XGBRegressor()
    m3.load_model("model_3h_v2.json") # 新メンバー！
    
    m6 = xgb.XGBRegressor()
    m6.load_model("model_6h_v2.json")
    return m1, m3, m6

try:
    model_1h, model_3h, model_6h = load_all_models()
except Exception as e:
    st.error(f"AIモデルの読み込みに失敗しました。'model_1h_v2.json'、'model_3h_v2.json'、'model_6h_v2.json' の3つのファイルがすべてGitHubにアップロードされているか確認してください。")
    st.stop()

# 2. メイン処理：ファイルがアップロードされたら動く
if uploaded_file is not None:
    try:
        # データの読み込み
        df_raw = pd.read_excel(uploaded_file, sheet_name='統合データ（メイン）', header=None)
        columns_raw = df_raw.iloc[4].tolist()
        df = df_raw.iloc[5:].copy()
        
        # 項目名のお掃除（改行、スペース、スラッシュを消して統一）
        clean_columns = [str(c).replace('\n', '').replace('/', '').replace(' ', '').strip() for c in columns_raw]
        df.columns = clean_columns
        
        # 必要な列を数値化
        df['water_level現況水位(m)'] = pd.to_numeric(df['water_level現況水位(m)'], errors='coerce')
        df['wl_change_1h1h前からの水位変化(m)'] = pd.to_numeric(df['wl_change_1h1h前からの水位変化(m)'], errors='coerce')
        df['rainfall_3h3時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_3h3時間累積雨量(mm)'], errors='coerce')
        df['rainfall_6h6時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_6h6時間累積雨量(mm)'], errors='coerce')
        
        # 不要な空行を削除
        df = df.dropna(subset=['datetime（日時）', 'water_level現況水位(m)']).reset_index(drop=True)
        
        if len(df) == 0:
            st.warning("⚠️ 有効なデータ行が見つかりませんでした。6行目以降にデータが入っているか確認してください。")
            st.stop()
            
        # 直近の（最後の）行を取得
        latest_row = df.iloc[-1]
        latest_time = pd.to_datetime(latest_row['datetime（日時）'])
        
        current_wl = float(latest_row['water_level現況水位(m)'])
        current_change = float(latest_row['wl_change_1h1h前からの水位変化(m)']) if not pd.isna(latest_row['wl_change_1h1h前からの水位変化(m)']) else 0.0
        current_r3 = float(latest_row['rainfall_3h3時間累積雨量(mm)'])
        current_r6 = float(latest_row['rainfall_6h6時間累積雨量(mm)'])
        
        features_order = [
            'water_level現況水位(m)', 
            'wl_change_1h1h前からの水位変化(m)', 
            'rainfall_3h3時間累積雨量(mm)', 
            'rainfall_6h6時間累積雨量(mm)'
        ]
        
        # --- 3つの専用モデルで主要ポイントを一発予測！ ---
        
        # ① 1時間後の予測
        r3_1h = current_r3 + future_rain_1h
        r6_1h = current_r6 + future_rain_1h
        X_1h = pd.DataFrame([[current_wl, current_change, r3_1h, r6_1h]], columns=features_order)
        pred_1h = float(model_1h.predict(X_1h)[0])
        
        # ② 3時間後の予測（専用モデル）
        rain_to_3h = sum([future_rain_1h, future_rain_2h, future_rain_3h])
        r3_3h = current_r3 + rain_to_3h
        r6_3h = current_r6 + rain_to_3h
        X_3h = pd.DataFrame([[current_wl, current_change, r3_3h, r6_3h]], columns=features_order)
        pred_3h = float(model_3h.predict(X_3h)[0])
        
        # ③ 6時間後の予測（専用モデル）
        total_future_rain = sum([future_rain_1h, future_rain_2h, future_rain_3h, future_rain_4h, future_rain_5h, future_rain_6h])
        r3_6h = current_r3 + total_future_rain
        r6_6h = current_r6 + total_future_rain
        X_6h = pd.DataFrame([[current_wl, current_change, r3_6h, r6_6h]], columns=features_order)
        pred_6h = float(model_6h.predict(X_6h)[0])
        
        # --- 中間の時間をトリプル予測をベースに高精度に補間 ---
        pred_wl_list = []
        
        # 1時間後
        pred_wl_list.append(pred_1h)
        
        # 2時間後（現在・1時間後・3時間後の流れから自然に補間）
        pred_2h = pred_1h + (pred_3h - pred_1h) * 0.5
        pred_wl_list.append(pred_2h)
        
        # 3時間後
        pred_wl_list.append(pred_3h)
        
        # 4時間後・5時間後（3時間後から6時間後の間の変化カーブ）
        pred_4h = pred_3h + (pred_6h - pred_3h) * (1/3)
        pred_5h = pred_3h + (pred_6h - pred_3h) * (2/3)
        pred_wl_list.append(pred_4h)
        pred_wl_list.append(pred_5h)
        
        # 6時間後
        pred_wl_list.append(pred_6h)

        # 3. サマリー表示
        st.subheader("📊 未来の水位予測サマリー")
        
        alert_level = 0.90
        max_pred = max(pred_wl_list)
        if max_pred >= alert_level:
            st.error(f"🚨 【警告】6時間以内に水防団待機水位（{alert_level:.2f}m）を上回る予測（最大 {max_pred:.2f}m）が検出されました！迅速に警戒してください。")
        else:
            st.success(f"✅ 現在のところ、6時間以内に待機水位（{alert_level:.2f}m）を超える予測はありません。")
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="現在 水位", value=f"{current_wl:.2f} m")
        col2.metric(label="🔮 1時間後 予測", value=f"{pred_1h:.2f} m", delta=f"{pred_1h - current_wl:+.2f} m")
        col3.metric(label="🔮 3時間後 予測", value=f"{pred_3h:.2f} m", delta=f"{pred_3h - pred_1h:+.2f} m")
        col4.metric(label="🔮 6時間後 予測", value=f"{pred_6h:.2f} m", delta=f"{pred_6h - pred_4h:+.2f} m")
        
        # 4. インタラクティブグラフの作成
        st.subheader("📈 水位の推移と予測値（マウスを当てると数値が表示されます）")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['datetime（日時）'], y=df['water_level現況水位(m)'], name='過去の実績水位', line=dict(color='blue', width=2)))
        
        # 予測値の点（1〜6時間後まで）
        pred_times = [latest_time + timedelta(hours=h) for h in range(1, 7)]
        fig.add_trace(go.Scatter(x=pred_times, y=pred_wl_list, name='AIの未来予測 (トリプルモデル駆動)', mode='markers+lines', line=dict(color='orange', dash='dash'), marker=dict(size=10)))
        
        # 待機水位の横線（0.90m）
        fig.add_hline(y=alert_level, line_dash="dot", line_color="red", annotation_text=f"水防団待機水位 ({alert_level:.2f}m)", annotation_position="top left")
        
        fig.update_layout(xaxis_title="日時", yaxis_title="水位 (m)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"ファイルの解析中にエラーが発生しました。シート名や形式が正しいか確認してください。内容: {e}")
else:
    st.info("💡 左側のサイドバーから、菅川橋のExcelデータをアップロードすると予測が始まります！")
