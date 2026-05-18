import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="段数カウンター Ver.8", layout="centered")
st.title("📏 均一ピッチ・面投影カウンター")

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
    display_img = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 横線（境界の影）を強烈に引き出す
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.absolute(sobely)
    sobely = np.uint8(sobely)

    target_sens = st.session_state.brain.get(color_mode, 40)

    def analyze_by_projection(sens):
        pw = sobely.shape[1]
        # 中央の30%の「面」を切り出す
        x_start = int(pw * 0.35)
        x_end = int(pw * 0.65)
        roi = sobely[:, x_start:x_end]
        
        # 【新ロジック】横方向にエッジ強度を合計（面での投影）
        # これにより、斜めの線や薄い影でも「横線の塊」として綺麗に波形になります
        projection = np.sum(roi, axis=1)
        
        # 波形からピーク（境界線）を探す
        peaks, _ = find_peaks(projection, height=sens * (x_end - x_start) * 0.3, distance=10)
        
        if len(peaks) < 2:
            return len(peaks), 0, peaks, x_start, x_end
        
        # 隣り合う境界線の「間隔（テープの幅）」をすべて計算
        intervals = np.diff(peaks)
        # 最も頻出する「正しい1段の幅（ピッチ）」を決定
        typical_pitch = np.median(intervals)
        
        # 【物理補正】全体のタワーの高さ（一番上の線から一番下の線まで）を、決定した1段の幅で割る
        total_height = peaks[-1] - peaks[0]
        estimated_count = int(round(total_height / typical_pitch)) + 1
        
        # 均一性（自信度）の計算
        std_dev = np.std(intervals)
        conf = max(0, min(100, 100 - int(std_dev * 8)))
        
        return estimated_count, conf, peaks, x_start, x_end

    # 学習・感度最適化処理
    if learn_button:
        best_s = 40
        min_error = 999
        for s in range(10, 150, 2):
            est_c, _, _, _, _ = analyze_by_projection(s)
            error = abs(est_c - true_count)
            if error < min_error:
                min_error = error
                best_s = s
        st.session_state.brain[color_mode] = best_s
        st.sidebar.success(f"最適化完了: 感度 {best_s}")
        st.rerun()

    # 判定実行
    final_ans, final_conf, detected_peaks, x_start, x_end = analyze_by_projection(target_sens)

    # 画面への描画（面スキャンエリアと検知線）
    # スキャンエリアを薄い青の透過膜で表示
    overlay = display_img.copy()
    cv2.rectangle(overlay, (x_start, 0), (x_end, 1000), (255, 100, 0), -1)
    cv2.addWeighted(overlay, 0.15, display_img, 0.85, 0, display_img)
    
    # 認識した境界線（緑の線でプロット）
    for py in detected_peaks:
        cv2.line(display_img, (x_start - 20, py), (x_end + 20, py), (0, 255, 0), 2)

    # 表示
    st.image(display_img, use_column_width=True, caption="青いエリアの『面』で影の塊を解析中。緑線＝検知した段の区切り")
    
    col1, col2 = st.columns(2)
    col1.metric("判定結果", f"{final_ans} 段")
    col2.metric("テープ幅の均一度（自信度）", f"{final_conf} %")
    
    if final_conf < 60:
        st.warning("⚠️ 影の検知が一部飛んでいます。左の「学習ボタン」を押して、この色に対する感度を最適化してください。")
