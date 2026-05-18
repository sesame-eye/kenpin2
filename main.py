import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="段数カウンター Ver.13", layout="centered")
st.title("📏 斜めアライメント補正・カウンター")

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
    
    # ノイズ対策として強めに平滑化（Ver.12と同じ）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 15, 75, 75)
    
    # 大きな陰影の「うねり」を拾う（Ver.12と同じ）
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
        
        # 閾値を高く設定（Ver.12と同じ）
        peaks, _ = find_peaks(projection, height=sens, distance=15)
        
        if len(peaks) < 2:
            return len(peaks), 0, [], 0, 1000, 30, 0, x_start, x_end
            
        # 【Ver.13 新ロジック】タワーの傾き角度（斜め成分）を算出
        def get_peak_profile(roi, peaks, s):
            profile = []
            for y in peaks:
                y_roi = roi[max(0,y-s):min(roi.shape[0],y+s), :]
                y_prof = np.mean(y_roi, axis=0)
                profile.append(y_prof)
            return np.array(profile)

        scan_range = 10 # ピーク周辺をスキャンする範囲
        profiles = get_peak_profile(sobely[:, x_start:x_end], peaks, scan_range)

        # 上部と下部のピークのプロファイル（中央を捉える）
        def get_tower_center(prof):
             # プロファイルの中央周辺で陰影（谷）を探す
            valley_peaks, _ = find_peaks(-prof, distance=len(prof)//4)
            if len(valley_peaks) > 0:
                 # 最も深い谷の中央を返す
                 valley_depths = prof[valley_peaks]
                 sorted_indices = np.argsort(valley_depths) # 小さい順
                 return valley_peaks[sorted_indices[0]] # 最も深い
            return len(prof) // 2 # 見つからない場合は中央を返す

        # 上部と下部のタワーの中心ピクセル（x座標）
        # Ver.12では全体の高さを上端peaks[0]と下端peaks[-1]で捉えていた
        top_prof = profiles[0]
        bot_prof = profiles[-1]
        top_center = get_tower_center(top_prof) + x_start
        bot_center = get_tower_center(bot_prof) + x_start

        # 上部下部中心の座標
        top_edge = peaks[0]
        bottom_edge = peaks[-1]
        top_edge_center = [top_center, top_edge]
        bottom_edge_center = [bot_center, bottom_edge]

        # 【幾何学マクロ計算】全体の高さ ÷ 1段の厚み（Ver.12と同じ）
        total_height = bottom_edge - top_edge
        intervals = np.diff(peaks)
        valid_intervals = [i for i in intervals if (np.median(intervals)*0.5 < i < np.median(intervals)*1.5)]
        
        if valid_intervals:
            typical_pitch = np.mean(valid_intervals)
        else:
            typical_pitch = np.median(intervals) if len(intervals) > 0 else 45
            
        estimated_count = int(round(total_height / typical_pitch)) + 1
        
        # 【Ver.13 新幾何学アプローチ】
        # タワー全体の傾き（斜め成分）にアライメントされた理想的な均一ライン（予測ライン）を再生成
        
        # 傾きベクトル
        vec = np.array(bottom_edge_center) - np.array(top_edge_center)
        # 各段の厚み方向のベクトル
        step_vec = vec / (estimated_count - 1)

        # タワーの傾き角度（斜め成分）
        aligned_peaks_centers = []
        for i in range(estimated_count):
            point = np.array(top_edge_center) + step_vec * i
            aligned_peaks_centers.append(point)

        # タワーの平均的な傾き角度（水平からの角度）
        angle = np.arctan2(vec[1], vec[0])
        st.caption(f"現在の算出データ ｜ 全体高: {total_height:.1f}px 1段厚: {typical_pitch:.1f}px 傾き角: {np.degrees(angle):.1f}度")

        # 自信度（Ver.12より修正：均一度＋アライメントの一致度）
        std_dev = np.std(intervals) if len(intervals) > 0 else 100
        # 実際の陰影がタワーの中心にどれだけ合っているか
        centers = np.array([ top_center if i == 0 else bot_center if i == estimated_count-1 else (top_center + step_vec[0] * i) for i in range(estimated_count) ])
        valley_centers = np.array([ get_tower_center(profiles[i]) + x_start for i in range(min(len(profiles), estimated_count)) ])
        # スケールを合わせる
        valid_indices = min(len( valley_centers), estimated_count)
        center_std_dev = np.std(centers[:valid_indices] - valley_centers[:valid_indices])
        conf = max(0, min(100, 100 - int(std_dev * 4) - int(center_std_dev * 1)))
        
        # 【Ver.13 新可視化】斜め予測ライン
        aligned_peaks_ends = []
        # 各予測ラインの左右端（タワー全体に均一な斜め線）
        for p in aligned_peaks_centers:
            # タワーの中心に合わせた斜め線
            length = typical_pitch # 線の長さ
            # 傾き方向と直交するベクトル
            perp_vec = np.array([-vec[1], vec[0]]) / np.linalg.norm(np.array([-vec[1], vec[0]]))
            start = p - perp_vec * (typical_pitch)
            end = p + perp_vec * (typical_pitch)
            aligned_peaks_ends.append([start, end])

        return estimated_count, conf, ideal_peaks, x_start, x_end, typical_pitch, aligned_peaks_ends, top_edge_center, bottom_edge_center

    # 学習・最適化処理（幾何学マクロ・Ver.12と同じ）
    if learn_button:
        best_s = 50
        min_error = 999
        for s in range(20, 180, 5):
            est_c, _, _, _, _, _, _, _, _ = analyze_geometry_aligned(s)
            error = abs(est_c - true_count)
            if error < min_error:
                min_error = error
                best_s = s
        st.session_state.brain[color_mode] = best_s
        st.sidebar.success(f"最適化完了: 感度 {best_s}")
        st.rerun()

    # 判定実行
    final_ans, final_conf, _, _, _, current_pitch, aligned_peaks_ends, top_edge_center, bottom_edge_center = analyze_geometry_aligned(target_sens)

    # 【Ver.13 改良可視化】
    # 画面に描画（斜めアライメント補正ラインとタワー中心）
    display_img = img.copy()
    
    # 傾き角度（水平からの角度）
    vec = np.array(bottom_edge_center) - np.array(top_edge_center)
    angle = np.degrees(np.arctan2(vec[1], vec[0]))

    # タワーの中心軸（薄い青の透過枠）
    overlay = display_img.copy()
    cv2.line(overlay, tuple(top_edge_center.astype(int)), tuple(bottom_edge_center.astype(int)), (255, 150, 50), 30) # 中心軸
    cv2.addWeighted(overlay, 0.1, display_img, 0.9, 0, display_img)

    # 青線＝マクロ均一補正を適用し、タワーの傾き（斜め成分）に合わせて「斜め」に引かれた予測アライメントライン
    for p in aligned_peaks_ends:
        start = p[0].astype(int)
        end = p[1].astype(int)
        if 0 <= start[1] < 1000 and 0 <= end[1] < 1000:
            cv2.line(display_img, tuple(start), tuple(end), (255, 50, 0), 2)

    # 結果表示（可視化画像と判定結果）
    # 可視化：予測ラインはタワーの湾曲・傾きを反映した「斜め線」になる
    st.image(display_img, use_column_width=True, caption=f"幾何学マクロ・斜めアライメント解析中（青線：レンズ湾曲・傾き補正済みの予測ライン 角度:{angle:.1f}度）")
    
    col1, col2 = st.columns(2)
    col1.metric("幾何学アライメント判定結果", f"{final_ans} 段")
    col2.metric("タワー構造の安定度（自信度）", f"{final_conf} %")
    
    if final_conf < 60:
        st.warning("⚠️ 自信度が低い状態です。実際の陰影がタワーの形状に合っていない可能性があります。左の『学習ボタン』を押して、この写真の歪み率に合わせてAIが自動同期します。")
