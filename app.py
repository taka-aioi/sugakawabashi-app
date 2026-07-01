import streamlit as st
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="菅川橋 水位予測システム V7.0", page_icon="🌊", layout="wide")
st.title("🌊 菅川橋 水位予測システム (1時間刻み高解像度版 V7.0)")
st.markdown("ゆうくん設計の新スケジュール。直近6時間を1時間単位で入力することで、雨の『勢いの変化』を完璧に捉え、フライング出動を防ぎます。")

# --- ページ実行（更新）時の日本時間を1回だけ固定取得 ---
jst_zone = timezone(timedelta(hours=9))
if "calculated_time_v7" not in st.session_state:
    st.session_state.calculated_time_v7 = datetime.now(jst_zone)

def update_time():
    st.session_state.calculated_time_v7 = datetime.now(jst_zone)

# --- サイドバー：入力エリア ---
st.sidebar.header("💧 現在の川の状況")
input_current_wl = st.sidebar.number_input("現在の実況水位 (m)", min_value=-1.0, max_value=5.0, value=0.69, step=0.01, on_change=update_time)
input_current_change = st.sidebar.number_input("1時間前からの水位変化量 (m)", min_value=-2.0, max_value=2.0, value=0.04, step=0.01, on_change=update_time)
input_rain_1h_ago = st.sidebar.number_input("直近1時間前〜現時点までの雨量 (mm)", min_value=0.0, max_value=200.0, value=3.0, step=1.0, on_change=update_time)

st.sidebar.markdown("---")
st.sidebar.header("🔮 1時間刻みのこれからの予測雨量 (mm)")

# 💡 ゆうくん発案の超細密スケジュール入力枠！今回の検証用に初期値をあらかじめセットしてあるよ
f_rain_1h = st.sidebar.number_input("これからの「1時間目」の雨量 (mm)", min_value=0.0, max_value=200.0, value=3.0, step=1.0, on_change=update_time)
f_rain_2h = st.sidebar.number_input("これからの「2時間目」の雨量 (mm)", min_value=0.0, max_value=200.0, value=10.0, step=1.0, on_change=update_time)
f_rain_3h = st.sidebar.number_input("これからの「3時間目」の雨量 (mm)", min_value=0.0, max_value=200.0, value=10.0, step=1.0, on_change=update_time)
f_rain_4h = st.sidebar.number_input("これからの「4時間目」の雨量 (mm)", min_value=0.0, max_value=200.0, value=6.0, step=1.0, on_change=update_time)
f_rain_5h = st.sidebar.number_input("これからの「5時間目」の雨量 (mm)", min_value=0.0, max_value=200.0, value=6.0, step=1.0, on_change=update_time)
f_rain_6h = st.sidebar.number_input("これからの「6時間目」の雨量 (mm)", min_value=0.0, max_value=200.0, value=6.0, step=1.0, on_change=update_time)

st.sidebar.markdown("---")
st.sidebar.header("⏳ 中期予測雨量 (mm)")
f_rain_6_12h_sum = st.sidebar.number_input("6〜12時間後までの『6時間合計』雨量 (mm)", min_value=0.0, max_value=1000.0, value=35.0, step=1.0, on_change=update_time)

base_now_time = st.session_state.calculated_time_v7

# モデルの読み込み
@st.cache_resource
def load_all_models():
    m1, m3, m6, m12 = xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor()
    m1.load_model("model_1h_v7.json")
    m3.load_model("model_3h_v7.json")
    m6.load_model("model_6h_v7.json")
    m12.load_model("model_12h_v7.json")
    return m1, m3, m6, m12

try:
    model_1h, model_3h, model_6h, model_12h = load_all_models()
except Exception as e:
    st.error(f"AIモデル(V7)の読み込みに失敗しました。GitHubに4つの 'v7.json' ファイルがあるか確認してください。")
    st.stop()

# --- 未来予測の計算 (V7専用特徴量へのマッピング) ---
try:
    features_order = [
        'water_level現況水位(m)', 'wl_change_1h1h前からの水位変化(m)', 'rainfall_1h1時間累積雨量(mm)',
        'f_rain_1h', 'f_rain_2h', 'f_rain_3h', 'f_rain_4h', 'f_rain_5h', 'f_rain_6h', 'f_rain_6_12h_sum'
    ]

    # 各モデルに対して、入力された時系列雨量をそのまま真っ直ぐ届ける！小細工なし！
    X_input = pd.DataFrame([[
        input_current_wl, input_current_change, input_rain_1h_ago,
        f_rain_1h, f_rain_2h, f_rain_3h, f_rain_4h, f_rain_5h, f_rain_6h, f_rain_6_12h_sum
    ]], columns=features_order)

    # 12時間までの予測値をスマートに計算
    raw_1h = input_current_wl + float(model_1h.predict(X_input)[0])
    raw_3h = input_current_wl + float(model_3h.predict(X_input)[0])
    raw_6h = input_current_wl + float(model_6h.predict(X_input)[0])
    raw_12h = input_current_wl + float(model_12h.predict(X_input)[0])

    raw_wl_list = [raw_1h, raw_3h, raw_6h, raw_12h]
    pred_hours = [0, 1, 3, 6, 12] # 24時間を廃止して12時間仕様に！
    alert_level = 0.90
    
    # 💡 黄金ルール適用
    pred_wl_list = [input_current_wl]
    for v in raw_wl_list:
        if v <= 0.10:
            pred_wl_list.append(max(0.00, min(input_current_wl, 0.10)))
        else:
            pred_wl_list.append(v)

    # 最悪シナリオ予測
    worst_wl_list = [input_current_wl]
    for v in pred_wl_list[1:]:
        if v <= 0.5:
            margin = 0.05
        elif v >= 0.9:
            margin = 0.20
        else:
            margin = 0.05 + (0.20 - 0.05) * ((v - 0.5) / (0.91 - 0.5))
        worst_wl_list.append(v + margin)

    # 時間軸（JST）
    time_axis = [base_now_time + timedelta(hours=h) for h in pred_hours]

    # 和暦(令和)の文字列生成
    def get_wareki_str(dt):
        year = dt.year
        wareki_year = year - 2018
        year_str = "元" if wareki_year == 1 else str(wareki_year)
        return dt.strftime(f"令和{year_str}年%m月%d日 %H時%M分")

    now_wareki_str = get_wareki_str(base_now_time)

    # 突破時刻を逆算する関数
    def find_exact_cross_time(wl_list):
        for i in range(1, len(wl_list)):
            if wl_list[i-1] < alert_level <= wl_list[i]:
                val_diff = wl_list[i] - wl_list[i-1]
                if val_diff == 0:
                    return time_axis[i-1]
                ratio = (alert_level - wl_list[i-1]) / val_diff
                hours_to_cross = pred_hours[i-1] + (pred_hours[i] - pred_hours[i-1]) * ratio
                return base_now_time + timedelta(hours=hours_to_cross)
        return None

    cross_time_worst = find_exact_cross_time(worst_wl_list)
    cross_time_base = find_exact_cross_time(pred_wl_list)

    def get_duration_str(target_dt):
        if target_dt is None:
            return ""
        diff = target_dt - base_now_time
        total_seconds = int(diff.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"現在から{hours}時間{minutes}分後"

    # --- 画面表示 ---
    st.subheader(f"📊 12時間未来予測サマリー （{now_wareki_str} 時点）")
    
    if cross_time_worst is not None:
        worst_time_str = cross_time_worst.strftime("%d日 %H時%M分")
        worst_dur_str = get_duration_str(cross_time_worst)
        worst_display = f"**{worst_time_str}（{worst_dur_str}）**"
        
        if cross_time_base is not None:
            base_time_str = cross_time_base.strftime("%d日 %H時%M分")
            base_dur_str = get_duration_str(cross_time_base)
            base_display = f"**{base_time_str}（{base_dur_str}）**"
        else:
            base_display = "**12時間以内突破なし**"
            
        st.error(f"🚨 【大雨警戒アラート】水防団待機水位（0.90m）を超える予測時刻は、{worst_display} 〜 {base_display} となっています。この時間帯を目安に堤防点検を開始してください。")
    else:
        st.success(f"✅ 12時間先まで、最悪シナリオでも待機水位（{alert_level:.2f}m）を超える予測はありません。")
        
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="現在 水位", value=f"{input_current_wl:.2f} m")
    col2.metric(label="🔮 AI基本予測 (12時間後)", value=f"{pred_wl_list[-1]:.2f} m")
    col3.metric(label="🛡️ 最悪シナリオ最大値", value=f"{max(worst_wl_list):.2f} m")
    col4.metric(label="⚠️ 水防団待機水位", value=f"{alert_level:.2f} m")

    # グラフ化
    st.subheader("📈 これから12時間後までの水位予測カーブ (日本時間: JST)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=pred_wl_list, name='AI基本予測 (V7.0)', mode='markers+lines', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=time_axis, y=worst_wl_list, name='⚠️ 最悪シナリオ予測 (可変マージン)', mode='markers+lines', line=dict(color='red', width=2, dash='dash')))
    fig.add_hline(y=alert_level, line_dash="dot", line_color="darkred", annotation_text=f"水防団待機水位 ({alert_level:.2f}m)")

    fig.update_layout(
        xaxis=dict(tickformat="%d日 %H:%M", title="日時 (JST)"),
        yaxis_title="水位 (m)", 
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"予測計算中にエラーが発生しました: {e}")
