
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 画面全体のデザイン設定（タイトルやアイコン）
st.set_page_config(page_title="菅川橋 水位予測システム", page_icon="🌊", layout="wide")

st.title("🌊 菅川橋 水位予測システム (トリプル予測版)")
st.markdown("広島県防災Webからダウンロードした直近のExcelデータをアップロードするだけで、1〜3時間後の水位をAIが予測します。")

# 1. サイドバー：ファイルのアップロード
st.sidebar.header("📁 データ入力")
uploaded_file = st.sidebar.file_saver = st.sidebar.file_uploader(
    "Excelファイルをここにドラッグ＆ドロップしてください", 
    type=["xlsx"]
)

# AIモデルの読み込み関数
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("model_1h.json")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"AIモデル（model_1h.json）の読み込みに失敗しました。Colabのフォルダにあるか確認してください。")
    st.stop()

# 2. メイン処理：ファイルがアップロードされたら動く
if uploaded_file is not None:
    try:
        # Excelデータの読み込みと成形（Colabでやったのと同じ処理）
        df_raw = pd.read_excel(uploaded_file, sheet_name='統合データ（メイン）', header=None)
        columns_raw = df_raw.iloc[4].tolist()
        df = df_raw.iloc[5:].copy()
        df.columns = [str(c).replace('\n', '') for c in columns_raw]
        
        # 必要な列を数値化
        df['water_level現況水位(m)'] = pd.to_numeric(df['water_level現況水位(m)'], errors='coerce')
        df['rainfall_3h3時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_3h3時間累積雨量(mm)'], errors='coerce')
        df['rainfall_6h6時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_6h6時間累積雨量(mm)'], errors='coerce')
        df = df.dropna(subset=['datetime（日時）', 'water_level現況水位(m)']).reset_index(drop=True)
        
        # 直近の（最後の）行を取得して予測
        latest_row = df.iloc[-1]
        latest_time = pd.to_datetime(latest_row['datetime（日時）'])
        current_wl = float(latest_row['water_level現況水位(m)'])
        
        # --- トリプル予測の実行 ---
        # 1時間後
        X_1h = pd.DataFrame([[latest_row['water_level現況水位(m)'], latest_row['rainfall_3h3時間累積雨量(mm)'], latest_row['rainfall_6h6時間累積雨量(mm)']]], 
                            columns=['water_level現況水位(m)', 'rainfall_3h3時間累積雨量(mm)', 'rainfall_6h6時間累積雨量(mm)'])
        pred_1h = float(model.predict(X_1h)[0])
        
        # 2時間後・3時間後（未来の雨量予測がない場合は、簡易的に直近のトレンドをブレンドして連続予測）
        pred_2h = pred_1h + (pred_1h - current_wl) * 0.7
        pred_3h = pred_2h + (pred_2h - pred_1h) * 0.5
        
        # 3. サマリー表示（デカデカと出す！）
        st.subheader("📊 未来の水位予測サマリー")
        
        # 待機水位（1.20m）の判定チェック
        max_pred = max(pred_1h, pred_2h, pred_3h)
        if max_pred >= 1.20:
            st.error(f"🚨 【警告】3品時間以内に水防団待機水位（1.20m）を上回る予測（最大 {max_pred:.2f}m）が検出されました！")
        else:
            st.success("✅ 現在のところ、3時間以内に待機水位（1.20m）を超える予測はありません。安全です。")
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="現在 水位", value=f"{current_wl:.2f} m")
        col2.metric(label="🔮 1時間後 予測", value=f"{pred_1h:.2f} m", delta=f"{pred_1h - current_wl:+.2f} m")
        col3.metric(label="🔮 2時間後 予測", value=f"{pred_2h:.2f} m", delta=f"{pred_2h - pred_1h:+.2f} m")
        col4.metric(label="🔮 3時間後 予測", value=f"{pred_3h:.2f} m", delta=f"{pred_3h - pred_2h:+.2f} m")
        
        # 4. 動くインタラクティブグラフ（Plotly）の作成
        st.subheader("📈 水位の推移と予測値（マウスを当てると数値が表示されます）")
        
        fig = go.Figure()
        # 実績値の線
        fig.add_trace(go.Scatter(x=df['datetime（日時）'], y=df['water_level現況水位(m)'], name='過去の実績水位', line=dict(color='blue', width=2)))
        
        # 予測値の点
        pred_times = [latest_time + timedelta(hours=1), latest_time + timedelta(hours=2), latest_time + timedelta(hours=3)]
        pred_values = [pred_1h, pred_2h, pred_3h]
        fig.add_trace(go.Scatter(x=pred_times, y=pred_values, name='AIの未来予測', mode='markers+lines', line=dict(color='orange', dash='dash'), marker=dict(size=10)))
        
        # 待機水位の横線
        fig.add_hline(y=1.20, line_dash="dot", line_color="red", annotation_text="水防団待機水位 (1.20m)", annotation_position="top left")
        
        fig.update_layout(xaxis_title="日時", yaxis_title="水位 (m)", hovermode="x unified", legend_notebook=False)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"ファイルの解析中にエラーが発生しました。シート名や形式が正しいか確認してください。内容: {e}")
else:
    st.info("💡 左側のサイドバーから、菅川橋のExcelデータをアップロードすると予測が始まります！")
