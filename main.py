import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="段数カウンター Ver.14", layout="centered")
st.title("📐 斜めアライメント補正・カウンター Ver.14")

# --- 設定データの初期化 ---
if 'brain' not in st.session_state:
    st.session_state.brain = {"白": 50, "黄": 45, "緑": 55, "青": 60, "茶": 50}

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
    
    # 表面のざらざらノイズを潰す（平滑化）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 15, 75, 75)
    
    # 大きな陰影のうねりだけを抽出
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    sobely = np.uint8(np.absolute(sobely))

    target_sens = st.session_state.brain.get(color_mode, 50)

    def analyze_geometry_aligned(sens):
        pw = sobely.shape[1]
        x_start = int(pw * 0.35)
        x_end = int(pw * 0.65)
        roi = sobely[:, x_start:x_end]
        
        # 縦方向のプロファイル（中央エリアの平均）
        projection = np.mean(roi, axis=1)
        
        # 閾値を高めに設定して主要な境界を検出
        peaks, _ = find_peaks(projection, height=sens, distance=15)
        
        # 最低限のピークがない場合の安全装置
        if len(peaks) < 2:
            return len(peaks), 0, 30, [], [x_start + (x_end-x_start)//2, 0], [x_start + (x_end-x_start)//2, 1000]
            
        # タワー全体の高さを取得
        top_edge = peaks[0]
        bottom_edge = peaks[-1]
        total_height = bottom_edge - top_edge
        
        # 代表ピッチ（1段の厚み）の算出
        intervals = np.diff(peaks)
        valid_intervals = [i for i in intervals if (np.median(intervals)*0.5 < i < np.median(intervals)*1.5)]
        typical_pitch = np.mean(valid_intervals) if valid_intervals else (np.median(intervals) if len(intervals) > 0 else 45)
            
        # 【幾何学マクロ計算】全体の高さから段数を推測
        estimated_count = int(round(total_height / typical_pitch)) + 1
        
        # --- 傾き（斜め成分）の検出ロジック ---
        # 簡易的にお椀型の湾曲・傾きの中心軸を捉える
        top_roi = sobely[max(0, top_edge-10):min(1000, top_edge+10), x_start:x_end]
        bot_roi = sobely[max(0, bottom_edge-10):min(1000, bottom_edge+10), x_start:x_end]
        
        top_x_sum = np.mean(top_roi, axis=0) if top_roi.size > 0 else np.ones(x_end-x_start)
        bot_x_sum = np.mean(bot_roi, axis=0) if bot_roi.size > 0 else np.ones(x_end-x_start)
        
        # 最もエッジが強い中心位置を特定
        top_center_x = x_start + (np.argmax(top_x_sum) if len(top_x_sum) > 0 else (x_end - x_start) // 2)
        bot_center_x = x_start + (np.argmax(bot_x_sum) if len(bot_x_sum) > 0 else (x_end - x_start) // 2)
        
        top_edge_center = [top_center_x, top_edge]
        bottom_edge_center = [bot_center_x, bottom_edge]
        
        # タワーの傾きベクトル
        vec = np.array(bottom_edge_center) - np.array(top_edge_center)
        step_vec = vec / max(1, (estimated_count - 1))
        
        # 斜めアライメントされた予測ラインの左右の端点を生成
        aligned_peaks_ends = []
        for i in range(estimated_count):
            p_center = np.array(top_edge_center) + step_vec * i
            
            # 傾き方向と直交するベクトル（水平方向のラインにするため）
            perp_vec = np.array([-vec[1], vec[0]], dtype=np.float64)
            norm = np.linalg.norm(perp_vec)
            if norm > 0:
                perp_vec /= norm
                
            # 線の長さを設定
            line_length = (x_end - x_start) * 0.8
            start_point = p_center - perp_vec * line_length
            end_point = p_center + perp_vec * line_length
            aligned_peaks_ends.append([start_point, end_point])
            
        # 自信度の計算（ピッチの均一度から算出）
        std_dev = np.std(intervals) if len(intervals) > 0 else 100
        conf = max(5, min(100, 100 - int(std_dev * 4)))
        
        # ★ Ver.13のエラー原因：受け渡し変数の数をここで完全統一（6個）
        return estimated_count, conf, typical_pitch, aligned_peaks_ends, top_edge_center, bottom_edge_center

    # 学習・最適化処理
    if learn_button:
        best_s = 50
        min_error = 999
        for s in range(20, 180, 5):
            est_c, _, _, _, _, _ = analyze_geometry_aligned(s)
            error = abs(est_c - true_count)
            if error < min_error:
                min_error = error
                best_s = s
        st.session_state.brain[color_mode] = best_s
        st.sidebar.success(f"最適化完了: 感度 {best_s}")
        st.rerun()

    # 判定実行（受け取り側の変数を6個に完全一致）
    final_ans, final_conf, current_pitch, aligned_peaks_ends, top_edge_center, bottom_edge_center = analyze_geometry_aligned(target_sens)

    # --- 可視化の描画 ---
    display_img = img.copy()
    
    # タワーの中心軸（薄い青のシースルー線）
    overlay = display_img.copy()
    cv2.line(overlay, (int(top_edge_center[0]), int(top_edge_center[1])), (int(bottom_edge_center[0]), int(bottom_edge_center[1])), (255,
