import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 画面全体のデザイン設定
st.set_page_config(page_title="菅川橋 水位予測システム", page_icon="🌊", layout="wide")

st.title("🌊 菅川橋 水位予測システム (6時間先シミュレーション版)")
st.markdown("広島県防災Webからダウンロードした直近のExcelデータをアップロードし、これからの予測雨量を入力することで、1〜6時間後の水位をAIがシミュレーションします。")

# 1. サイドバー：ファイルのアップロードと未来の雨量入力
st.sidebar.header("📁 データ入力")
uploaded_file = st.sidebar.file_uploader(
    "Excelファイルをここにドラッグ＆ドロップしてください", 
    type=["xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.header("🔮 これからの予測雨量 (mm)")
st.sidebar.markdown("これからの気象予報データなどを基に、各時間帯の**1時間あたりの雨量**を入力してください。")

# 6時間後まで入力ボックスを拡張
future_rain_1h = st.sidebar.number_input("これからの1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_2h = st.sidebar.number_input("2時間後の1分間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_3h = st.sidebar.number_input("3時間後の1分間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_4h = st.sidebar.number_input("4時間後の1分間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_5h = st.sidebar.number_input("5時間後の1分間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
future_rain_6h = st.sidebar.number_input("6時間後の1分間の雨量 (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

# AIモデルの読み込み関数
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("model_1h.json")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"AIモデル（model_1h.json）の読み込みに失敗しました。ファイルが同じ場所にアップロードされているか確認してください。")
    st.stop()

# 2. メイン処理：ファイルがアップロードされたら動く
if uploaded_file is not None:
    try:
        # ゆうくんのテンプレート（統合データ（メイン））を読み込む
        df_raw = pd.read_excel(uploaded_file, sheet_name='統合データ（メイン）', header=None)
        columns_raw = df_raw.iloc[4].tolist()
        df = df_raw.iloc[5:].copy()
        df.columns = [str(c).replace('\n', '').strip() for c in columns_raw]
        
        # 必要な列を数値化
        df['water_level/現況水位(m)'] = pd.to_numeric(df['water_level/現況水位(m)'], errors='coerce')
        df['rainfall_3h/3時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_3h/3時間累積雨量(mm)'], errors='coerce')
        df['rainfall_6h/6時間累積雨量(mm)'] = pd.to_numeric(df['rainfall_6h/6時間累積雨量(mm)'], errors='coerce')
        
        # 不要な空行を削除
        df = df.dropna(subset=['datetime（日時）', 'water_level/現況水位(m)']).reset_index(drop=True)
        
        if len(df) == 0:
            st.warning("⚠️ 有効なデータ行が見つかりませんでした。6行目以降にデータが入っているか確認してください。")
            st.stop()
            
        # 直近の（最後の）行を取得
        latest_row = df.iloc[-1]
        latest_time = pd.to_datetime(latest_row['datetime（日時）'])
        current_wl = float(latest_row['water_level/現況水位(m)'])
        
        # 現在の累積雨量
        current_r3 = float(latest_row['rainfall_3h/3時間累積雨量(mm)'])
        current_r6 = float(latest_row['rainfall_6h/6時間累積雨量(mm)'])
        
        # --- 6時間後までのシミュレーション計算 ---
        # 各時間の予測雨量をリスト化
        future_rains = [future_rain_1h, future_rain_2h, future_rain_3h, future_rain_4h, future_rain_5h, future_rain_6h]
        pred_wl_list = []
        
        temp_wl = current_wl
        temp_r3 = current_r3
        temp_r6 = current_r6
        
        # 1時間ずつループを回して6時間後まで連続予測！
        for i in range(6):
            rain = future_rains[i]
            # 3時間累積と6時間累積に新しい1時間分の雨量を上乗せ
            temp_r3 += rain
            temp_r6 += rain
            
            # AIモデルに入力して次の1時間後の水位を予測
            X = pd.DataFrame([[temp_wl, temp_r3, temp_r6]], 
                             columns=['water_level現況水位(m)', 'rainfall_3h3時間累積雨量(mm)', 'rainfall_6h6時間累積雨量(mm)'])
            next_pred = float(model.predict(X)[0])
            
            # 予測値をリストに保存し、次のループのベースにする
            pred_wl_list.append(next_pred)
            temp_wl = next_pred

        # 3. サマリー表示
        st.subheader("📊 未来の水位予測サマリー")
        
        # 水防団待機水位（0.90m）の判定チェック
        alert_level = 0.90
        max_pred = max(pred_wl_list)
        if max_pred >= alert_level:
            st.error(f"🚨 【警告】6時間以内に水防団待機水位（{alert_level:.2f}m）を上回る予測（最大 {max_pred:.2f}m）が検出されました！迅速に警戒してください。")
        else:
            st.success(f"✅ 現在のところ、6時間以内に待機水位（{alert_level:.2f}m）を超える予測はありません。")
            
        # メトリック表示（現在の状況 ➔ 1h ➔ 3h ➔ 6h の主要ポイントを抜粋して表示）
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="現在 水位", value=f"{current_wl:.2f} m")
        col2.metric(label="🔮 1時間後 予測", value=f"{pred_wl_list[0]:.2f} m", delta=f"{pred_wl_list[0] - current_wl:+.2f} m")
        col3.metric(label="🔮 3時間後 予測", value=f"{pred_wl_list[2]:.2f} m", delta=f"{pred_wl_list[2] - pred_wl_list[1]:+.2f} m")
        col4.metric(label="🔮 6時間後 予測", value=f"{pred_wl_list[5]:.2f} m", delta=f"{pred_wl_list[5] - pred_wl_list[4]:+.2f} m")
        
        # 4. 動くインタラクティブグラフの作成
        st.subheader("📈 水位の推移と予測値（マウスを当てると数値が表示されます）")
        
        fig = go.Figure()
        # 実績値の線
        fig.add_trace(go.Scatter(x=df['datetime（日時）'], y=df['water_level/現況水位(m)'], name='過去の実績水位', line=dict(color='blue', width=2)))
        
        # 予測値の点（1〜6時間後まで生成）
        pred_times = [latest_time + timedelta(hours=h) for h in range(1, 7)]
        fig.add_trace(go.Scatter(x=pred_times, y=pred_wl_list, name='AIの未来予測 (6時間シミュレーション)', mode='markers+lines', line=dict(color='orange', dash='dash'), marker=dict(size=10)))
        
        # 待機水位の横線（0.90m）
        fig.add_hline(y=alert_level, line_dash="dot", line_color="red", annotation_text=f"水防団待機水位 ({alert_level:.2f}m)", annotation_position="top left")
        
        fig.update_layout(xaxis_title="日時", yaxis_title="水位 (m)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"ファイルの解析中にエラーが発生しました。シート名や形式が正しいか確認してください。内容: {e}")
else:
    st.info("💡 左側のサイドバーから、菅川橋のExcelデータをアップロードすると予測が始まります！")
