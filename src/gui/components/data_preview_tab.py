import streamlit as st
import numpy as np
from src.utils.logger_manager import log_info, log_error, log_warning

def render_data_preview_tab(app_instance, rotation_angle):
    """データプレビュータブを描画"""
    st.header("📊 データプレビュー")

    # エンディアン設定が変更された場合、既存のデータをクリア
    if hasattr(st.session_state, 'img_byte_order_changed') and st.session_state.img_byte_order_changed:
        if 'current_data' in st.session_state:
            # 削除後、必ずNoneに再初期化
            del st.session_state.current_data
            st.session_state.current_data = None
        st.session_state.img_byte_order_changed = False
        st.info("🔄 エンディアン設定変更により、ファイルを再読み込みします...")

    try:
        log_info(f"データプレビュー開始: ファイル={st.session_state.uploaded_file_path}, 回転角={rotation_angle}", "gui.preview")

        # データ読み込み
        data_array, error = app_instance.timing_analyzer.load_and_preview_data(
            st.session_state.uploaded_file_path, 
            angle=rotation_angle
        )

        if error:
            log_error(f"データ読み込みエラー: {error}", "gui.preview")
            st.error(f"❌ データ読み込みエラー: {error}")
            return

        if data_array is None:
            log_warning("データ配列がNullです", "gui.preview")
            st.error("❌ データが読み込けませんでした")
            return

        st.session_state.current_data = data_array

        log_info(f"データプレビュー成功: サイズ={data_array.shape}, データ型={data_array.dtype}", "gui.preview")

        # データ情報表示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📏 データサイズ", f"{data_array.shape[1]} × {data_array.shape[0]}")
        with col2:
            st.metric("📊 データ型", str(data_array.dtype))
        with col3:
            st.metric("📈 値の範囲", f"{data_array.min():.0f} - {data_array.max():.0f}")

        # プレビュープロット作成
        log_info("プレビュープロット作成開始", "gui.preview")
        fig = app_instance.timing_analyzer.create_preview_plot(
            data_array,
            gx_xmin=st.session_state.gx_xmin,
            gx_xmax=st.session_state.gx_xmax,
            gx_ymin=st.session_state.gx_ymin,
            gx_ymax=st.session_state.gx_ymax,
            lfex_xmin=st.session_state.lfex_xmin,
            lfex_xmax=st.session_state.lfex_xmax
        )

        # プロット表示
        st.pyplot(fig)
        log_info("プレビュープロット表示完了", "gui.preview")

    except Exception as e:
        log_error(f"プレビュー生成エラー: {str(e)}", "gui.preview", exc_info=True)
        st.error(f"❌ プレビュー生成エラー: {str(e)}")
