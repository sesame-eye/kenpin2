import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="段数カウンター Ver.7 (可視化版)", layout="centered")
st.title("📏 均一ピッチ解析カウンター (検知ライン表示)")

# --- 設定データの初期化 ---
if 'brain' not in st.session_state:
    st.session_state.brain = {"白": 40, "黄": 35, "緑": 45, "青": 50, "茶": 40}

# --- サイドバー ---
st.sidebar.header("🎨 カラー/モード")
color_mode = st.sidebar.selectbox("テープの色", ["白", "黄", "緑", "青", "茶"])
true_count = st.sidebar.number_input("正解の段数を入力", min_value=1, value=20)
learn_button = st.sidebar.button(f"{color_mode}を学習/最適化")

# --- 写真入力 ---
option = st.radio("写真入力", ("カメラ撮影", "ライブラリ参照"), horizontal=True)
uploaded_file = st.camera_input("撮影") if option == "カメラ撮影" else st.file_uploader("写真を選択", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    proc_h = 1000
    img = cv2.resize(img, (int(w * (proc_h/h)), proc_h))
    
    # 表示用（線を引くターゲット画像）
    display_img = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.absolute(edges)
    edges = np.uint8(edges)

    target_sens = st.session_state.brain.get(color_mode, 40)

    def analyze_stack(sens):
        pw = edges.shape[1]
        # 中央付近で5本スキャン
        x_points = [int(pw*0.4), int(pw*0.45), int(pw*0.5), int(pw*0.55), int(pw*0.6)]
        all_peaks = []
        
        for x in x_points:
            line = edges[:, x]
            peaks = []
            for y, val in enumerate(line):
                if val > sens:
                    if not peaks or (y - peaks[-1] > 6): 
                        peaks.append(y)
            all_peaks.append(peaks)
        
        counts = [len(p) for p in all_peaks]
        median_count = int(np.median(counts))
        
        # 画面に赤い「検知線」を描画する処理（中央のラインを代表として描画）
        target_peaks = all_peaks[2] # 真ん中のスキャンライン
        for py in target_peaks:
            cv2.line(display_img, (int(pw*0.35), py), (int(pw*0.65), py), (0, 0, 255), 3) # 赤い太線

        # スキャンエリアの枠（青）
        cv2.rectangle(display_img, (int(pw*0.38), 0), (int(pw*0.62), 1000), (255, 0, 0), 2)
        
        intervals = []
        for p in all_peaks:
            if len(p) > 2:
                intervals.extend(np.diff(p))
        
        if intervals:
            std_dev = np.std(intervals)
            conf = max(0, min(100, 100 - int(std_dev * 5)))
        else:
            conf = 0
            
        return median_count, conf

    # 学習処理（逆算）
    if learn_button:
        best_s = 30
        min_error = 999
        for s in range(10, 120, 2):
            # 描画が干渉しないようダミーで計算
            pw_d = edges.shape[1]
            x_points_d = [int(pw_d*0.4), int(pw_d*0.45), int(pw_d*0.5), int(pw_d*0.55), int(pw_d*0.6)]
            counts_d = []
            for x in x_points_d:
                line = edges[:, x]
                peaks = []
                for y, val in enumerate(line):
                    if val > s:
                        if not peaks or (y - peaks[-1] > 6): peaks.append(y)
                counts_d.append(len(peaks))
            res_c = int(np.median(counts_d))
            
            if abs(res_c - true_count) < min_error:
                min_error = abs(res_c - true_count)
                best_s = s
        st.session_state.brain[color_mode] = best_s
        st.sidebar.success(f"最適化完了: 感度{best_s}")
        st.rerun()

    # 判定実行
    final_ans, final_conf = analyze_stack(target_sens)

    # 表示
    st.image(display_img, use_column_width=True, caption="青枠内をスキャン中。赤線＝AIが認識した隙間")
    col1, col2 = st.columns(2)
    col1.metric("判定結果", f"{final_ans} 段")
    col2.metric("均一性（自信度）", f"{final_conf} %")
    
    if final_conf < 50:
        st.warning("⚠️ 赤線が等間隔に並んでいない、または飛んでいます。学習ボタンで感度を最適化してください。")
