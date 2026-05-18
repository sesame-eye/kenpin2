import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="段数カウンター Ver.12", layout="centered")
st.title("📏 幾何学マクロピッチ・カウンター")

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
    
    # ノイズ（表面のざらざら）を徹底的に潰すために強めに平滑化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 15, 75, 75) # 面の平滑性を保ちつつ、強い境界だけ残す
    
    # 大きな陰影の「うねり」だけを拾う（細かいエッジは無視）
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    sobely = np.uint8(np.absolute(sobely))

    target_sens = st.session_state.brain.get(color_mode, 50)

    def analyze_geometry(sens):
        pw = sobely.shape[1]
        x_start = int(pw * 0.4)
        x_end = int(pw * 0.6)
        roi = sobely[:, x_start:x_end]
        
        # 縦方向のプロファイル（中央エリアの平均）
        projection = np.mean(roi, axis=1)
        
        # かなり大きめのノイズ対策として閾値を高く設定
        peaks, _ = find_peaks(projection, height=sens, distance=15)
        
        if len(peaks) < 2:
            return len(peaks), 0, [], 0, 1000, 30
            
        # 【重要】一番最初（上端）と一番最後（下端）の明確な境界からタワー全体の高さを取得
        top_edge = peaks[0]
        bottom_edge = peaks[-1]
        total_height = bottom_edge - top_edge
        
        # はっきり検知できている主要な段差から「代表ピッチ（1段の厚み）」を計算
        intervals = np.diff(peaks)
        # 極端に狭い・広いノイズを外れ値として除外
        valid_intervals = [i for i in intervals if (np.median(intervals)*0.5 < i < np.median(intervals)*1.5)]
        
        if valid_intervals:
            typical_pitch = np.mean(valid_intervals)
        else:
            typical_pitch = np.median(intervals) if len(intervals) > 0 else 45
            
        # 【幾何学マクロ計算】
        # 細かい線を数えるのではなく、「全体の高さ ÷ 1段の厚み」で段数を決定！
        estimated_count = int(round(total_height / typical_pitch)) + 1
        
        # 算出したマクロピッチを基に、理想的な均一ライン（補正ライン）を再生成
        ideal_peaks = [int(top_edge + i * typical_pitch) for i in range(estimated_count)]
        
        # 自信度（実際のピークが理想の均一ピッチにどれだけ近いか）
        std_dev = np.std(intervals) if len(intervals) > 0 else 100
        conf = max(0, min(100, 100 - int(std_dev * 4)))
        
        return estimated_count, conf, ideal_peaks, x_start, x_end, typical_pitch

    # 学習・最適化処理
    if learn_button:
        best_s = 50
        min_error = 999
        for s in range(20, 180, 5):
            est_c, _, _, _, _, _ = analyze_geometry(s)
            error = abs(est_c - true_count)
            if error < min_error:
                min_error = error
                best_s = s
        st.session_state.brain[color_mode] = best_s
        st.sidebar.success(f"最適化完了: 感度 {best_s}")
        st.rerun()

    # 判定実行
    final_ans, final_conf, ideal_peaks, x_start, x_end, current_pitch = analyze_geometry(target_sens)

    # 画面への描画（マクロ補正された均一ライン）
    overlay = display_img.copy()
    cv2.rectangle(overlay, (x_start, 0), (x_end, 1000), (0, 150, 255), -1)
    cv2.addWeighted(overlay, 0.1, display_img, 0.9, 0, display_img)
    
    # 青線＝マクロ均一補正を適用した「物理的にここに段差があるはず」という予測線
    for py in ideal_peaks:
        if 0 <= py < 1000:
            cv2.line(display_img, (x_start - 30, py), (x_end + 30, py), (255, 50, 0), 2)

    # 結果表示
    st.image(display_img, use_column_width=True, caption="幾何学マクロピッチ解析中（青線：レンズ湾曲・歪み補正済みの予測ライン）")
    
    col1, col2 = st.columns(2)
    col1.metric("幾何学判定結果", f"{final_ans} 段")
    col2.metric("タワー構造の安定度", f"{final_conf} %")
    st.caption(f"現在の算出データ ｜ 全体高: {ideal_peaks[-1]-ideal_peaks[0] if ideal_peaks else 0}px  1段の厚み: {current_pitch:.1f}px")
    
    if abs(final_ans - true_count) > 0:
        st.info("💡 写真を読み込ませた後、左側の『学習/最適化』ボタンを押すと、この写真の歪み率に合わせてAIが自動同期します。")
