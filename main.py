import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="段数カウンター Ver.7", layout="centered")
st.title("📏 均一ピッチ解析カウンター")

# --- 設定データの初期化 ---
if 'brain' not in st.session_state:
    st.session_state.brain = {"白": 40, "黄": 35, "緑": 45, "青": 50, "茶": 40}

# --- サイドバー ---
st.sidebar.header("🎨 カラー/モード")
color_mode = st.sidebar.selectbox("テープの色", ["白", "黄", "緑", "青", "茶"])
true_count = st.sidebar.number_input("正解の段数を入力", min_value=1, value=20)
learn_button = st.sidebar.button(f"{color_mode}を学習/最適化")

# --- 写真入力 ---
uploaded_file = st.file_uploader("写真を選択 (image_9290d6.jpg など)", type=['jpg', 'jpeg', 'png'])
camera_file = st.camera_input("カメラで撮影")
active_file = camera_file if camera_file else uploaded_file

if active_file:
    file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    proc_h = 1000
    img = cv2.resize(img, (int(w * (proc_h/h)), proc_h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ノイズを抑えつつ境界線を強調
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.absolute(edges)
    edges = np.uint8(edges)

    target_sens = st.session_state.brain.get(color_mode, 40)

    def analyze_stack(sens):
        # 中央付近で5本スキャンして平均的な「幅」を算出
        pw = edges.shape[1]
        x_points = [int(pw*0.4), int(pw*0.45), int(pw*0.5), int(pw*0.55), int(pw*0.6)]
        all_peaks = []
        
        for x in x_points:
            line = edges[:, x]
            peaks = []
            for y, val in enumerate(line):
                if val > sens:
                    # ピークの開始位置を記録
                    if not peaks or (y - peaks[-1] > 5): 
                        peaks.append(y)
            all_peaks.append(peaks)
        
        # 各ラインの段数を取得
        counts = [len(p) for p in all_peaks]
        median_count = int(np.median(counts))
        
        # 【重要】ピッチ（幅）の均一性をチェック
        intervals = []
        for p in all_peaks:
            if len(p) > 2:
                intervals.extend(np.diff(p))
        
        # 最も多い「幅」を特定
        if intervals:
            expected_pitch = np.median(intervals)
            # 自信度：幅がどれだけ「均一」に近いか
            std_dev = np.std(intervals)
            conf = max(0, min(100, 100 - int(std_dev * 5)))
        else:
            conf = 0
            
        return median_count, conf

    # 学習処理
    if learn_button:
        best_s = 30
        min_error = 999
        for s in range(10, 120, 2):
            res_c, _ = analyze_stack(s)
            if abs(res_c - true_count) < min_error:
                min_error = abs(res_c - true_count)
                best_s = s
        st.session_state.brain[color_mode] = best_s
        st.sidebar.success(f"最適化完了: 感度{best_s}")
        st.rerun()

    # 判定実行
    final_ans, final_conf = analyze_stack(target_sens)

    # 表示
    st.image(img, use_column_width=True)
    col1, col2 = st.columns(2)
    col1.metric("判定結果", f"{final_ans} 段")
    col2.metric("均一性（自信度）", f"{final_conf} %")
    
    if final_conf < 50:
        st.warning("⚠️ テープの幅がバラバラに検知されています。背景や照明を確認してください。")
