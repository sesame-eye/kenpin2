import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="段数カウンター Ver.6", layout="centered")
st.title("🎯 カラー別・高精度段数カウンター")

# --- 色別の学習データの初期化 ---
if 'brain' not in st.session_state:
    # 色ごとの感度設定を保存
    st.session_state.brain = {"白": 40, "黄": 35, "緑": 45, "青": 50}

# --- サイドバー：学習と設定 ---
st.sidebar.header("🎨 カラーモード選択")
color_mode = st.sidebar.selectbox("測定するテープの色", ["白", "黄", "緑", "青"])
st.sidebar.write(f"現在の「{color_mode}」用感度: {st.session_state.brain[color_mode]}")

st.sidebar.write("---")
st.sidebar.header("🎓 AIの修正学習")
true_count = st.sidebar.number_input("正解の段数を入力", min_value=1, value=15)
if st.sidebar.button(f"{color_mode}の学習データを更新"):
    # 現在の画像に対して、正解に最も近い感度を逆算して保存する処理（後述のロジックを流用）
    st.session_state.is_learning_mode = True
else:
    st.session_state.is_learning_mode = False

# --- メイン：写真入力 ---
option = st.radio("写真入力", ("カメラ撮影", "ライブラリ参照"), horizontal=True)
uploaded_file = st.camera_input("撮影") if option == "カメラ撮影" else st.file_uploader("写真を選択", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 画像処理
    h, w = raw_img.shape[:2]
    proc_h = 1000
    img = cv2.resize(raw_img, (int(w * (proc_h/h)), proc_h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ノイズ除去とエッジ強調
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.absolute(sobely)
    sobely = np.uint8(sobely)

    # ターゲット感度
    target_sens = st.session_state.brain[color_mode]

    def count_layers(sens):
        # 画像中央の30%の範囲だけでスキャン（背景のノイズを避ける）
        pw = sobely.shape[1]
        scan_x = [int(pw*0.45), int(pw*0.5), int(pw*0.55)]
        counts = []
        for x in scan_x:
            line = sobely[:, x]
            c = 0
            active = False
            for val in line:
                if val > sens:
                    if not active:
                        c += 1
                        active = True
                else:
                    active = False
            counts.append(c)
        return counts

    # 学習ボタンが押された場合の逆算処理
    if st.session_state.is_learning_mode:
        best_s = 30
        min_diff = 999
        for s in range(10, 150, 1):
            res = count_layers(s)
            diff = abs(np.median(res) - true_count)
            if diff < min_diff:
                min_diff = diff
                best_s = s
        st.session_state.brain[color_mode] = best_s
        st.sidebar.success(f"{color_mode}の感度を{best_s}に最適化しました！")
        target_sens = best_s

    # 最終判定
    results = count_layers(target_sens)
    final_ans = int(np.median(results))
    
    # 自信度の計算（3本のラインのバラツキをみる）
    variation = np.std(results)
    conf_score = max(0, min(100, 100 - int(variation * 25)))

    # 表示
    st.image(raw_img, use_column_width=True)
    col1, col2 = st.columns(2)
    col1.metric("判定結果", f"{final_ans} 段")
    col2.metric("AI自信度", f"{conf_score} %")
    
    st.info(f"💡 現在【{color_mode}】専用モードで動作中です。他の色の場合は左で切り替えてください。")
