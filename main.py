import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="段数カウンター Ver.10", layout="centered")
st.title("📏 物理ピッチフィルタ・カウンター")

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
    
    # 【新ロジック1】画像を鮮鋭化（シャープネス）
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp = cv2.filter2D(gray, -1, kernel_sharp)

    # 【新ロジック2】Canny法による輪郭抽出（Ver.9と同じ）
    edges = cv2.Canny(sharp, 20, 60)

    # 横線成分をさらに強調（Ver.9と同じ）
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobely = cv2.filter2D(edges, -1, kernel)
    sobely = np.uint8(np.absolute(sobely))

    target_sens = st.session_state.brain.get(color_mode, 40)

    def analyze_by_edge_projection_filtered(sens):
        pw = sobely.shape[1]
        # 中央の35%の「面」を切り出す
        x_start = int(pw * 0.325)
        x_end = int(pw * 0.675)
        roi = sobely[:, x_start:x_end]
        
        # 横方向にエッジ強度を合計
        projection = np.sum(roi, axis=1)
        
        # 波形からピーク（境界線）を探す。最初は過敏に拾う（distance=5）
        peaks, _ = find_peaks(projection, height=sens * (x_end - x_start) * 0.05, distance=5)
        
        if len(peaks) < 3:
            return len(peaks), 0, peaks, x_start, x_end
        
        # 【Ver.10 新フィルター】物理ピッチによる足切り
        # 1. すべての間隔を算出
        all_intervals = np.diff(peaks)
        # 2. 最頻値（正しいテープ幅）を特定
        typical_pitch = np.median(all_intervals)
        
        # 3. 再構築（典型的な幅の半分以下の間隔の線は「ノイズ」として捨てる）
        final_peaks = [peaks[0]]
        for p in peaks[1:]:
            interval = p - final_peaks[-1]
            if interval > typical_pitch * 0.6: # 典型的な幅の6割より広い場合だけ、新しい段と認める
                final_peaks.append(p)
            else:
                pass # 幅が狭すぎるのでノイズとして無視
        
        final_peaks = np.array(final_peaks)
        
        if len(final_peaks) < 2:
            return len(final_peaks), 0, final_peaks, x_start, x_end
            
        # 隣り合う境界線の「間隔（テープの幅）」を再計算
        intervals = np.diff(final_peaks)
        refined_pitch = np.median(intervals)
        
        # 【物理補正】全体のタワーの高さ（一番上の線から一番下の線まで）を、
        # 決定した1段の平均幅で割る
        total_height = final_peaks[-1] - final_peaks[0]
        estimated_count = int(round(total_height / refined_pitch)) + 1
        
        # 自信度の計算（ピッチのバラツキをみる）
        std_dev = np.std(intervals)
        conf = max(0, min(100, 100 - int(std_dev * 12)))
        
        return estimated_count, conf, final_peaks, x_start, x_end

    # 学習処理
    if learn_button:
        best_s = 40
        min_error = 999
        for s in range(10, 150, 2):
            est_c, _, _, _, _ = analyze_by_edge_projection_filtered(s)
            error = abs(est_c - true_count)
            if error < min_error:
                min_error = error
                best_s = s
        st.session_state.brain[color_mode] = best_s
        st.sidebar.success(f"最適化完了: 感度 {best_s}")
        st.rerun()

    # 判定実行
    final_ans, final_conf, detected_peaks, x_start, x_end = analyze_by_edge_projection_filtered(target_sens)

    # 画面への描画（面スキャンエリアと検知線）
    overlay = display_img.copy()
    cv2.rectangle(overlay, (x_start, 0), (x_end, 1000), (255, 100, 0), -1)
    cv2.addWeighted(overlay, 0.15, display_img, 0.85, 0, display_img)
    
    # 検知したエッジ（緑の線で描画）
    for py in detected_peaks:
        cv2.line(display_img, (x_start - 25, py), (x_end + 25, py), (0, 255, 0), 2)

    # 結果表示
    st.image(display_img, use_column_width=True, caption="面投影と物理ピッチフィルターで解析中。緑線＝検知した輪郭")
    
    col1, col2 = st.columns(2)
    col1.metric("判定結果", f"{final_ans} 段")
    col2.metric("テープ幅の均一度（自信度）", f"{final_conf} %")
    
    if final_conf < 60:
        st.warning("⚠️ 検知されたテープの幅がバラバラです。左の「学習ボタン」を押して、この色に対する感度を最適化してください。")
