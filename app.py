import streamlit as st
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="菅川橋 水位予測システム V6.9", page_icon="🌊", layout="wide")
st.title("🌊 菅川橋 水位予測システム (現場指揮特化型 V6.9)")
st.markdown("0.90mを突破する『正確な日時』と『残り時間』をダブル表示。現場の出動司令に迷いを与えません。")

# --- 💡 ページが読み込まれた（またはサイドバーが動いた）「その瞬間」の日本時間をここで1回だけ固定取得！ ---
jst_zone = timezone(timedelta(hours=9))
if "calculated_time" not in st.session_state:
    st.session_state.calculated_time = datetime.now(jst_zone)

# サイドバーの入力が変更されたら、計算基準時刻をその瞬間の時間でカチッと上書き固定する
def update_time():
    st.session_state.calculated_time = datetime.now(jst_zone)

# --- サイドバー：入力エリア（変更されたら一瞬で時刻を固定更新） ---
st.sidebar.header("💧 現在の川の状況")
input_current_wl = st.sidebar.number_input("現在の実況水位 (m)", min_value=-1.0, max_value=5.0, value=0.69, step=0.01, on_change=update_time)
input_current_change = st.sidebar.number_input("1時間前からの水位変化量 (m)", min_value=-2.0, max_value=2.0, value=0.04, step=0.01, on_change=update_time)
input_rain_1h_ago = st.sidebar.number_input("直近1時間前〜現時点までに降った雨量 (mm)", min_value=0.0, max_value=200.0, value=3.0, step=1.0, on_change=update_time)

st.sidebar.markdown("---")
st.sidebar.header("🔮 これからの予測降水量 (mm)")
future_rain_1h = st.sidebar.number_input("これからの1時間の雨量 (mm)", min_value=0.0, max_value=200.0, value=3.0, step=1.0, on_change=update_time)
future_rain_3h_sum = st.sidebar.number_input("1〜3時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=20.0, step=1.0, on_change=update_time)
future_rain_6h_sum = st.sidebar.number_input("3〜6時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=18.0, step=1.0, on_change=update_time)
future_rain_12h_sum = st.sidebar.number_input("6〜12時間後までの合計雨量 (mm)", min_value=0.0, max_value=500.0, value=35.0, step=1.0, on_change=update_time)
future_rain_24h_sum = st.sidebar.number_input("12〜24時間後までの合計雨量 (mm)", min_value=0.0, max_value=1000.0, value=97.0, step=1.0, on_change=update_time)

# 基準となる固定時間を変数に代入
base_now_time = st.session_state.calculated_time

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
    st.error(f"AIモデル(V6)の読み込みに失敗しました。")
    st.stop()

# --- 未来予測の計算 ---
try:
    features_order = [
        'water_level現況水位(m)', 'wl_change_1h1h前からの水位変化(m)', 
        'rainfall_1h1時間累積雨量(mm)', 'rainfall_3h3時間累積雨量(mm)', 
        'rainfall_6h6時間累積雨量(mm)', 'rainfall_24h24時間累積雨量(mm)',
        'rain_1h_ago1時間前の1時間雨量(mm)'
    ]

    v_1h = future_rain_1h
    v_1_3h = future_rain_3h_sum / 2.0
    v_3_6h = future_rain_6h_sum / 3.0
    v_6_12h = future_rain_12h_sum / 6.0
    v_12_24h = future_rain_24h_sum / 12.0

    # 各未来時点の計算
    X_1h = pd.DataFrame([[input_current_wl, input_current_change, v_1h, v_1h + input_rain_1h_ago, v_1h + input_rain_1h_ago, v_1h + input_rain_1h_ago, input_rain_1h_ago]], columns=features_order)
    raw_1h = input_current_wl + float(model_1h.predict(X_1h)[0])

    cum_3h = v_1h + future_rain_3h_sum
    X_3h = pd.DataFrame([[input_current_wl, input_current_change, v_1_3h, cum_3h, cum_3h, cum_3h, v_1_3h]], columns=features_order)
    raw_3h = input_current_wl + float(model_3h.predict(X_3h)[0])

    cum_6h = cum_3h + future_rain_6h_sum
    X_6h = pd.DataFrame([[input_current_wl, input_current_change, v_3_6h, future_rain_6h_sum, cum_6h, cum_6h, v_3_6h]], columns=features_order)
    raw_6h = input_current_wl + float(model_6h.predict(X_6h)[0])

    cum_12h = cum_6h + future_rain_12h_sum
    X_12h = pd.DataFrame([[input_current_wl, input_current_change, v_6_12h, v_6_12h * 3, future_rain_12h_sum, cum_12h, v_6_12h]], columns=features_order)
    raw_12h = input_current_wl + float(model_12h.predict(X_12h)[0])

    cum_24h = cum_12h + future_rain_24h_sum
    X_24h = pd.DataFrame([[input_current_wl, input_current_change, v_12_24h, v_12_24h * 3, v_12_24h * 6, cum_24h, v_12_24h]], columns=features_order)
    raw_24h = input_current_wl + float(model_24h.predict(X_24h)[0])

    raw_wl_list = [raw_1h, raw_3h, raw_6h, raw_12h, raw_24h]
    pred_hours = [0, 1, 3, 6, 12, 24]
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

    # 固定された更新時刻をベースにグラフの時間軸（JST）を作成
    time_axis = [base_now_time + timedelta(hours=h) for h in pred_hours]

    # 💡 和暦(令和)の文字列を生成（更新した瞬間の時刻で完全に固定）
    def get_wareki_str(dt):
        year = dt.year
        wareki_year = year - 2018
        year_str = "元" if wareki_year == 1 else str(wareki_year)
        return dt.strftime(f"令和{year_str}年%m月%d日 %H時%M分")

    now_wareki_str = get_wareki_str(base_now_time)

    # 💡 突破する瞬間を逆算する関数
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

    # 💡 「あと何時間何分後か」を正確に計算する関数
    def get_duration_str(target_dt):
        if target_dt is None:
            return ""
        diff = target_dt - base_now_time
        total_seconds = int(diff.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"現在から{hours}時間{minutes}分後"

    # --- 画面表示 ---
    # 💡 タイトル横の時間は、枠内を触ってページが更新された時間でピタッと固定されます！
    st.subheader(f"📊 24時間未来予測サマリー （{now_wareki_str} 時点）")
    
    if cross_time_worst is not None:
        worst_time_str = cross_time_worst.strftime("%d日 %H時%M分")
        worst_dur_str = get_duration_str(cross_time_worst)
        
        # 💡 【ゆうくんの指定通りに大大修正！】
        # 最悪シナリオの「日時（〇〇分後）」の文字列を作成
        worst_display = f"**{worst_time_str}（{worst_dur_str}）**"
        
        if cross_time_base is not None:
            base_time_str = cross_time_base.strftime("%d日 %H時%M分")
            base_dur_str = get_duration_str(cross_time_base)
            # 基本予測の「日時（〇〇分後）」の文字列を作成
            base_display = f"**{base_time_str}（{base_dur_str}）**"
        else:
            base_display = "**24時間以内突破なし**"
            
        # ゆうくんの理想通りの「何日何時何分（〇〇分後） 〜 何日何時何分（〇〇分後）」の形に完全同期！
        st.error(f"🚨 【大雨警戒アラート】水防団待機水位（0.90m）を超える予測時刻は、{worst_display} 〜 {base_display} となっています。この時間帯を目安に堤防点検を開始してください。")
    else:
        st.success(f"✅ 24時間先まで、最悪シナリオでも待機水位（{alert_level:.2f}m）を超える予測はありません。")
        
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="現在 水位", value=f"{input_current_wl:.2f} m")
    col2.metric(label="🔮 AI基本予測 (24時間後)", value=f"{pred_wl_list[-1]:.2f} m")
    col3.metric(label="🛡️ 最悪シナリオ最大値", value=f"{max(worst_wl_list):.2f} m")
    col4.metric(label="⚠️ 水防団待機水位", value=f"{alert_level:.2f} m")

    # グラフ化
    st.subheader("📈 これから24時間後までの水位予測カーブ (日本時間: JST)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=pred_wl_list, name='AI基本予測 (V6.9)', mode='markers+lines', line=dict(color='orange', width=2)))
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
