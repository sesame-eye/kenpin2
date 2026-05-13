import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="段数カウンター Ver.5", layout="centered")
st.title("🚀 プロフェッショナル段数AI")

# --- 学習データの初期化 ---
if 'color_profiles' not in st.session_state:
    st.session_state.color_profiles = [] # 特徴(色)
    st.session_state.learned_sens = []   # 対応する最適感度

# --- UI：常に表示する学習エリア ---
st.sidebar.header("🎓 AI学習パネル")
true_count = st.sidebar.number_input("正解の段数を入力", min_value=1, value=15)
learn_button = st.sidebar.button("この画像を正解として学習")

# --- メイン：写真入力 ---
option = st.radio("写真入力", ("カメラ撮影", "ライブラリ参照"), horizontal=True)
uploaded_file = st.camera_input("撮影") if option == "カメラ撮影" else st.file_uploader("写真を選択", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 1. プリセット（色解析）
    hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
    avg_color = [np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])]

    # AIが感度を予測
    target_sens = 30 # デフォルト
    if st.session_state.color_profiles:
        dists = [np.linalg.norm(np.array(avg_color) - np.array(p)) for p in st.session_state.color_profiles]
        nearest_idx = np.argmin(dists)
        target_sens = st.session_state.learned_sens[nearest_idx]

    # 2. 画像処理（エッジ強調）
    h, w = raw_img.shape[:2]
    proc_h = 1200
    img = cv2.resize(raw_img, (int(w * (proc_h/h)), proc_h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # ソーベルフィルタで横線を強力抽出
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.absolute(sobely)
    sobely = np.uint8(sobely)

    # 3. マルチラインスキャン (3箇所で計測して安定性をみる)
    def count_on_line(x_pos, sens):
        line = sobely[:, x_pos]
        c = 0
        active = False
        for val in line:
            if val > sens:
                if not active:
                    c += 1
                    active = True
            else:
                active = False
        return c

    pw = sobely.shape[1]
    c1 = count_on_line(int(pw*0.4), target_sens)
    c2 = count_on_line(int(pw*0.5), target_sens)
    c3 = count_on_line(int(pw*0.6), target_sens)

    # 平均をとる
    final_count = int(np.median([c1, c2, c3]))
    
    # 自信度：3本のラインがどれだけ一致しているか
    variance = np.std([c1, c2, c3])
    conf_score = max(0, min(100, 100 - int(variance * 20)))
    if len(st.session_state.color_profiles) == 0:
        conf_score = min(conf_score, 20) # 未学習なら低く

    # --- 表示 ---
    st.image(raw_img, use_column_width=True)
    col1, col2 = st.columns(2)
    col1.metric("判定結果", f"{final_count} 段")
    col2.metric("AI自信度", f"{conf_score} %")

    # --- 学習処理 ---
    if learn_button:
        # 最適感度を総当たりで逆算 (10〜150)
        best_s = target_sens
        min_diff = 999
        for s in range(10, 150, 2):
            test_c = count_on_line(int(pw*0.5), s)
            diff = abs(test_c - true_count)
            if diff < min_diff:
                min_diff = diff
                best_s = s
        
        st.session_state.color_profiles.append(avg_color)
        st.session_state.learned_sens.append(best_s)
        st.success(f"緑色テープ（感度:{best_s}）を学習しました！")
        st.rerun()
