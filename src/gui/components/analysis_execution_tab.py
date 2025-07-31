import streamlit as st
from pathlib import Path
from src.config.settings import get_config, set_config, save_user_config
from src.utils.logger_manager import log_info, log_error

def render_analysis_execution_tab(app_instance):
    """解析実行タブを描画"""
    st.header("🔍 解析実行")

    # エンディアン設定が変更された場合のチェック
    if hasattr(st.session_state, 'img_byte_order_changed') and st.session_state.img_byte_order_changed:
        st.warning("⚠️ エンディアン設定が変更されました。データプレビュータブでファイルを再読み込みしてください。")
        return

    # session_stateにcurrent_dataがない場合もNoneに初期化
    if not hasattr(st.session_state, 'current_data'):
        st.session_state.current_data = None

    if st.session_state.current_data is None:
        st.warning("⚠️ 先にデータをプレビューしてください")
        return

    # 解析パラメータの詳細設定
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎛️ 解析設定")
        ma_window = st.slider("移動平均ウィンドウ", 1, 50, get_config('analysis.processing.ma_window', 20))
        peak_threshold = st.slider("ピーク検出閾値", 0.01, 1.0, get_config('analysis.processing.peak_threshold', 0.1), step=0.01)

    with col2:
        st.subheader("🔧 ピーク検出設定")
        peak_detection_mode = st.selectbox(
            "ピーク検出モード",
            ["2ピーク検出", "1ピーク検出", "固定オフセット2ピーク検出"],
            index=["2ピーク検出", "1ピーク検出", "固定オフセット2ピーク検出"].index(get_config('analysis.peak_detection.mode', "2ピーク検出"))
        )
        if peak_detection_mode == "1ピーク検出":
            peak_selection_method = st.selectbox(
                "ピーク選択方法",
                ["最大強度", "最初のピーク", "最後のピーク"],
                index=["最大強度", "最初のピーク", "最後のピーク"].index(get_config('analysis.peak_detection.selection_method', "最大強度"))
            )
            fixed_offset_value = None # 1ピーク検出モードではオフセットは不要
        elif peak_detection_mode == "固定オフセット2ピーク検出":
            peak_selection_method = "最大強度" # 固定オフセットモードでは常に最大強度を基準
            fixed_offset_value = st.number_input(
                "二次ピークオフセット (ns)",
                value=get_config('analysis.peak_detection.fixed_offset', 0.24),
                min_value=-10.0,  # 負の値も許可（二次ピークが主要ピークより前にある場合）
                max_value=10.0,
                step=0.01,
                format="%.2f",
                key="fixed_offset_value_input",
                help="主要ピークからの二次ピークの固定時間オフセット (ns)。正の値：二次ピークが後、負の値：二次ピークが前"
            )
        else: # 2ピーク検出モード
            peak_selection_method = "最大強度"
            fixed_offset_value = None # 2ピーク検出モードではオフセットは不要

    # 基準時間設定（統一版）
    st.subheader("⏰ 基準時間設定")
    col_ref1, col_ref2 = st.columns(2)

    with col_ref1:
        reference_time_mode = st.selectbox(
            "基準時間モード",
            options=["gxii_peak", "gxii_rise", "lfex_peak", "absolute", "manual", "custom_t0"],
            format_func=lambda x: {
                "gxii_peak": "GXII ピーク基準",
                "gxii_rise": "GXII 立ち上がり基準",
                "lfex_peak": "LFEX ピーク基準",
                "absolute": "絶対時間基準",
                "manual": "手動設定基準",
                "custom_t0": "カスタム波形t0基準"
            }[x],
            index=["gxii_peak", "gxii_rise", "lfex_peak", "absolute", "manual", "custom_t0"].index(get_config('reference_time.mode', "gxii_peak")),
            help="タイミング解析とプロットの基準となる時間を選択"
        )

    with col_ref2:
        if reference_time_mode == "absolute":
            reference_value = st.number_input(
                "絶対基準時間 (ns)",
                value=get_config('reference_time.absolute_value', 0.0),
                step=0.001,
                format="%.3f",
                help="絶対時間基準として使用する値"
            )
        elif reference_time_mode == "manual":
            reference_value = st.number_input(
                "手動基準時間 (ns)",
                value=get_config('reference_time.manual_value', 0.0),
                step=0.001,
                format="%.3f",
                help="手動で設定する基準時間"
            )
        elif reference_time_mode == "gxii_rise":
            reference_value = st.number_input(
                "立ち上がり閾値 (%)",
                value=get_config('analysis.gxii_rise_percentage', 10.0),
                min_value=1.0,
                max_value=50.0,
                step=1.0,
                format="%.1f",
                help="GXII信号の立ち上がりを検出する閾値（最大値に対する%）"
            )
        else:
            reference_value = None
            if reference_time_mode == "gxii_peak":
                st.info("GXII ピークを基準時間として使用")
            elif reference_time_mode == "lfex_peak":
                st.info("LFEX ピークを基準時間として使用")
            elif reference_time_mode == "custom_t0":
                st.info("カスタム波形ファイルのt=0を基準時間として使用")

    # 回転角度設定
    angle = st.slider("回転角度 (度)", -45.0, 45.0, get_config('analysis.angle', 0.0), step=0.1)

    # ショット日時入力
    shot_datetime_str = st.text_input(
        "ショット日時 (例: 2025/06/25 16:59)",
        value=st.session_state.get('shot_datetime_str', ''),
        help="レポートに表示するショット日時を入力します。空欄の場合、日時は表示されません。"
    )
    st.session_state.shot_datetime_str = shot_datetime_str

    # 解析実行ボタン
    if st.button("🚀 解析実行", type="primary", use_container_width=True):
        with st.spinner("解析中..."):
            try:
                # 基準時間設定を保存
                set_config('reference_time.mode', reference_time_mode)
                if reference_time_mode == "absolute":
                    set_config('reference_time.absolute_value', reference_value)
                elif reference_time_mode == "manual":
                    set_config('reference_time.manual_value', reference_value)
                elif reference_time_mode == "gxii_rise":
                    set_config('analysis.gxii_rise_percentage', reference_value)
                
                set_config('analysis.processing.ma_window', ma_window)
                set_config('analysis.processing.peak_threshold', peak_threshold)
                set_config('analysis.peak_detection.mode', peak_detection_mode)
                set_config('analysis.peak_detection.selection_method', peak_selection_method)
                set_config('analysis.angle', angle)
                set_config('analysis.peak_detection.fixed_offset', fixed_offset_value) # オフセット値を保存
                save_user_config()

                # 解析実行（基準時間モードに応じた解析）
                gxii_rise_percentage = reference_value if reference_time_mode == "gxii_rise" else st.session_state.gxii_rise_percentage

                # 波形設定をパラメータとして構築
                waveform_config = {
                    'type': st.session_state.waveform_type,
                    'gaussian': {
                        'method': st.session_state.gaussian_method,
                        'fwhm': st.session_state.gaussian_fwhm
                    },
                    'custom_pulse': {
                        'enabled': st.session_state.custom_pulse_enabled
                    },
                    'custom_file': {
                        'file_path': st.session_state.custom_file_path,
                        'time_unit': st.session_state.get('custom_waveform_time_unit', 'ns')
                    }
                }

                results, error = app_instance.timing_analyzer.analyze_timing(
                    filepath=st.session_state.uploaded_file_path,
                    filename=Path(st.session_state.uploaded_file_path).name,
                    angle=angle,
                    gx_xmin=st.session_state.gx_xmin,
                    gx_xmax=st.session_state.gx_xmax,
                    gx_ymin=st.session_state.gx_ymin,
                    gx_ymax=st.session_state.gx_ymax,
                    lfex_xmin=st.session_state.lfex_xmin,
                    lfex_xmax=st.session_state.lfex_xmax,
                    ma_window=ma_window,
                    peak_threshold=peak_threshold,
                    peak_detection_mode=peak_detection_mode,
                    peak_selection_method=peak_selection_method,
                    fixed_offset_value=fixed_offset_value, # 新しい引数
                    time_calibration_mode=st.session_state.time_calibration_mode,
                    full_width_time=st.session_state.full_width_time,
                    time_per_pixel=st.session_state.time_per_pixel,
                    reference_time_mode=reference_time_mode,  # 選択された基準時間モードを使用
                    gxii_rise_percentage=gxii_rise_percentage,
                    waveform_type=st.session_state.waveform_type,
                    waveform_config=waveform_config,
                    shot_datetime_str=shot_datetime_str # ショット日時を渡す
                )

                if error:
                    st.error(f"❌ 解析エラー: {error}")
                else:
                    st.session_state.analysis_results = results
                    st.session_state.waveform_r_squared = results.get('waveform_r_squared', 0.0)
                    st.session_state.waveform_fitting_success = results.get('fitting_success', False)
                    st.success("✅ 解析が完了しました！")

                    # 結果の簡易表示
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔴 GXII ピーク", f"{results['gxii_peak_relative']:.3f} ns")
                    with col2:
                        st.metric("🔵 LFEX ピーク", f"{results['lfex_peak_1_relative']:.3f} ns")
                    with col3:
                        # タイミング差の意味を基準時間モードに応じて表示
                        if results['reference_time_mode'] == 'gxii_rise':
                            timing_label = "⏱️ 立ち上がり→LFEX差"
                        elif results['reference_time_mode'] == 'lfex_peak':
                            timing_label = "⏱️ GXII→LFEX差"
                        else:
                            timing_label = "⏱️ ピーク間差"
                        st.metric(timing_label, f"{results['time_diff']:.3f} ns")

            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ 解析エラー: {error_msg}")

                # 特定のエラーメッセージに対する詳細な説明
                if "カスタムパルスファイルが見つかりません" in error_msg:
                    st.error("💡 **カスタム波形ファイルの問題:**")
                    st.write("- ファイルパスが正しいか確認してください")
                    st.write("- ファイルが存在するか確認してください")
                    st.write("- ファイルの読み取り権限があるか確認してください")
                elif "サポートされていないファイル形式" in error_msg:
                    st.error("💡 **ファイル形式の問題:**")
                    st.write("- サポートされているファイル形式: .csv, .txt, .dat")
                    st.write("- ファイル形式を確認して再試行してください")
                elif "入力データ" in error_msg:
                    st.error("💡 **データの問題:**")
                    st.write("- データファイルが破損している可能性があります")
                    st.write("- 別のデータファイルで試してください")
                    st.write("- 領域設定を確認してください")
                elif "フィッティング" in error_msg:
                    st.error("💡 **フィッティングの問題:**")
                    st.write("- 別の波形タイプを試してください")
                    st.write("- 信号品質が低い可能性があります")
                    st.write("- 領域設定や前処理パラメータを調整してください")
                else:
                    st.error("💡 **一般的な解決策:**")
                    st.write("- データファイルと設定を確認してください")
                    st.write("- パラメータを調整して再試行してください")
                    st.write("- 問題が続く場合は、サポートに連絡してください")

                # デバッグ情報の表示（開発用）
                if st.checkbox("🔧 詳細なエラー情報を表示", key="show_debug_info"):
                    import traceback
                    st.text("詳細なエラー情報:")
                    st.code(traceback.format_exc())

                    # システム情報
                    st.text("システム情報:")
                    st.write(f"- 波形タイプ: {st.session_state.waveform_type}")
                    st.write(f"- ガウシアン手法: {st.session_state.gaussian_method}")
                    if st.session_state.waveform_type == "custom_pulse":
                        st.write(f"- カスタムファイルパス: {st.session_state.custom_file_path}")
                    st.write(f"- データファイル: {st.session_state.uploaded_file_path}")
                    st.write(f"- 基準時間モード: {reference_time_mode}")
