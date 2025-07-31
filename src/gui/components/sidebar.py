import streamlit as st
import os
from pathlib import Path
from src.config.settings import get_config, set_config, save_user_config
from src.core.data_loader import DataLoader
from src.core.timing_analyzer import TimingAnalyzer
from src.utils.file_utils import FileUtils
from src.utils.logger_manager import log_info, log_error, log_warning
from src.utils.file_operations import save_uploaded_file
from src.gui.components.log_panel import render_log_display_panel
from src.gui.components.preset_manager import load_preset, save_current_as_preset


def render_sidebar(app_instance) -> float:
    """サイドバーを描画"""
    with st.sidebar:
        st.header("📁 ファイル選択")

        # ファイルアップロード
        uploaded_file = st.file_uploader(
            "データファイルを選択",
            type=['txt', 'img'],
            help="対応形式: .txt, .img"
        )

        if uploaded_file is not None:
            # ファイルを一時保存
            temp_path = save_uploaded_file(uploaded_file.getbuffer(), uploaded_file.name)
            if temp_path:
                st.session_state.uploaded_file_path = temp_path
                st.success(f"✅ ファイルが読み込まれました: {uploaded_file.name}")

                # ファイル情報表示
                file_info = app_instance.file_utils.get_file_info(temp_path)
                st.info(f"📄 サイズ: {file_info['size_formatted']}")

        # IMG ファイル設定
        with st.expander("🔧 IMG ファイル設定", expanded=False):
            current_img_byte_order = st.session_state.img_byte_order
            st.session_state.img_byte_order = st.selectbox(
                "エンディアン設定",
                ["auto", "little", "big"],
                index=["auto", "little", "big"].index(st.session_state.img_byte_order),
                format_func=lambda x: {
                    "auto": "自動検出",
                    "little": "リトルエンディアン",
                    "big": "ビッグエンディアン"
                }[x],
                key="img_byte_order_select",
                help="IMGファイルのバイト順序を指定します。通常は自動検出で問題ありません。"
            )

            # 設定が変更された場合に同期
            if st.session_state.img_byte_order != current_img_byte_order:
                app_instance._sync_session_to_config()
                # エンディアン設定変更時はキャッシュされたデータと解析結果をクリア
                if 'current_data' in st.session_state:
                    del st.session_state.current_data
                    st.session_state.current_data = None
                if 'analysis_results' in st.session_state:
                    st.session_state.analysis_results = None
                st.session_state.img_byte_order_changed = True
                st.success("🔄 エンディアン設定が変更されました。ファイルを再読み込みします。")
                log_info(f"エンディアン設定変更: {current_img_byte_order} → {st.session_state.img_byte_order}", "gui.settings")
                st.rerun()  # ページを再実行してファイルを再読み込み

            # 現在の設定を表示
            if st.session_state.img_byte_order == "auto":
                st.info("💡 ファイル読み込み時に自動的に適切なエンディアンを検出します")
            elif st.session_state.img_byte_order == "little":
                st.info("📊 リトルエンディアン（Intel x86系）として読み込みます")
            else:
                st.info("📊 ビッグエンディアン（ネットワークバイト順）として読み込みます")

        st.markdown("---")

        # プリセット設定
        st.header("🎛️ プリセット設定")

        col_preset1, col_preset2 = st.columns(2)
        with col_preset1:
            if st.button("📋 標準", help="一般的な実験設定"):
                load_preset(app_instance.config_manager, 'standard')
            if st.button("🎯 高精度", help="精密測定向け設定"):
                load_preset(app_instance.config_manager, 'high_precision')

        with col_preset2:
            if st.button("📐 広域", help="広い測定範囲"):
                load_preset(app_instance.config_manager, 'wide_range')
            if st.button("💾 現在の設定を保存", help="現在の設定をプリセットとして保存"):
                save_current_as_preset(app_instance.config_manager, app_instance._sync_session_to_config)

        st.markdown("---")

        # 解析パラメータ
        st.header("⚙️ 解析パラメータ")

        with st.expander("🎯 GXII領域設定", expanded=True):
            st.session_state.gx_xmin = st.number_input("X最小", value=st.session_state.gx_xmin, key="gx_xmin_input")
            st.session_state.gx_xmax = st.number_input("X最大", value=st.session_state.gx_xmax, key="gx_xmax_input")
            st.session_state.gx_ymin = st.number_input("Y最小", value=st.session_state.gx_ymin, key="gx_ymin_input")
            st.session_state.gx_ymax = st.number_input("Y最大", value=st.session_state.gx_ymax, key="gx_ymax_input")

        with st.expander("🎯 LFEX領域設定", expanded=True):
            st.session_state.lfex_xmin = st.number_input("X最小", value=st.session_state.lfex_xmin, key="lfex_xmin_input")
            st.session_state.lfex_xmax = st.number_input("X最大", value=st.session_state.lfex_xmax, key="lfex_xmax_input")

        with st.expander("⏰ 時間校正設定", expanded=False):
            st.session_state.time_calibration_mode = st.selectbox(
                "校正モード",
                ["全幅指定", "1pixel指定", "フィッティング"],
                index=["全幅指定", "1pixel指定", "フィッティング"].index(st.session_state.time_calibration_mode),
                key="time_calibration_mode_select"
            )

            if st.session_state.time_calibration_mode == "全幅指定":
                st.session_state.full_width_time = st.number_input(
                    "全幅の時間 (ns)",
                    value=st.session_state.full_width_time,
                    min_value=0.1,
                    max_value=100.0,
                    step=0.1,
                    key="full_width_time_input"
                )
                # 1pxlあたりの時間を表示（計算値）
                pixel_count = 1024  # デフォルト値、実際はデータサイズから取得
                # session_stateにcurrent_dataがない場合もNoneに初期化
                if not hasattr(st.session_state, 'current_data'):
                    st.session_state.current_data = None
                if st.session_state.current_data is not None:
                    pixel_count = st.session_state.current_data.shape[0]
                calculated_time_per_pixel = st.session_state.full_width_time / pixel_count
                st.info(f"📊 1pxlあたりの時間: {calculated_time_per_pixel:.6f} ns/pixel")
            elif st.session_state.time_calibration_mode == "1pixel指定":
                st.session_state.time_per_pixel = st.number_input(
                    "1pxlあたりの時間 (ns/pixel)",
                    value=st.session_state.time_per_pixel,
                    min_value=0.0001,
                    max_value=1.0,
                    step=0.0001,
                    format="%.6f",
                    key="time_per_pixel_input"
                )
                # 全幅時間を表示（計算値）
                pixel_count = 1024  # デフォルト値、実際はデータサイズから取得
                # session_stateにcurrent_dataがない場合もNoneに初期化
                if not hasattr(st.session_state, 'current_data'):
                    st.session_state.current_data = None
                if st.session_state.current_data is not None:
                    pixel_count = st.session_state.current_data.shape[0]
                calculated_full_width = st.session_state.time_per_pixel * pixel_count
                st.info(f"📊 全幅の時間: {calculated_full_width:.3f} ns")
            else:
                st.session_state.full_width_time = st.number_input(
                    "初期全幅の時間 (ns)",
                    value=st.session_state.full_width_time,
                    min_value=0.1,
                    max_value=100.0,
                    step=0.1,
                    key="full_width_time_fit_input"
                )
                pixel_count = 1024
                if not hasattr(st.session_state, 'current_data'):
                    st.session_state.current_data = None
                if st.session_state.current_data is not None:
                    pixel_count = st.session_state.current_data.shape[0]
                calc_tpp = st.session_state.full_width_time / pixel_count
                st.info(f"📊 初期1pxlあたりの時間: {calc_tpp:.6f} ns/pixel (フィッティング)" )

        # 波形設定
        with st.expander("🌊 波形設定", expanded=False):
            st.session_state.waveform_type = st.selectbox(
                "波形タイプ",
                ["gaussian", "custom_pulse", "custom_file"],
                index=["gaussian", "custom_pulse", "custom_file"].index(st.session_state.waveform_type),
                format_func=lambda x: {
                    "gaussian": "ガウシアン",
                    "custom_pulse": "カスタムパルス",
                    "custom_file": "カスタムファイル"
                }[x],
                key="waveform_type_select",
                help="GXII ピーク検出に使用する波形タイプを選択"
            )

            if st.session_state.waveform_type == "gaussian":
                st.session_state.gaussian_method = st.selectbox(
                    "ガウシアン手法",
                    ["fixed_pulse", "fwhm_input"],
                    index=["fixed_pulse", "fwhm_input"].index(st.session_state.gaussian_method),
                    format_func=lambda x: {
                        "fixed_pulse": "固定パルス (σ=0.553)",
                        "fwhm_input": "FWHM入力"
                    }[x],
                    key="gaussian_method_select",
                    help="ガウシアンフィッティング手法を選択"
                )

                if st.session_state.gaussian_method == "fwhm_input":
                    st.session_state.gaussian_fwhm = st.number_input(
                        "FWHM (ns)",
                        value=st.session_state.gaussian_fwhm,
                        min_value=0.1,
                        max_value=10.0,
                        step=0.1,
                        format="%.1f",
                        key="gaussian_fwhm_input",
                        help="ガウシアンの半値全幅"
                    )
                else:
                    st.info("固定σ値: 0.553 ns (従来実装と同一)")

            elif st.session_state.waveform_type == "custom_pulse":
                st.session_state.custom_pulse_enabled = st.checkbox(
                    "カスタムパルス有効化",
                    value=st.session_state.custom_pulse_enabled,
                    key="custom_pulse_enabled_checkbox",
                    help="事前定義されたカスタムパルス波形を使用"
                )
                if st.session_state.custom_pulse_enabled:
                    st.info("💡 内蔵カスタムパルス波形を使用")
                else:
                    st.warning("⚠️ カスタムパルスが無効です")

            elif st.session_state.waveform_type == "custom_file":
                st.session_state.custom_file_path = st.text_input(
                    "カスタム波形ファイルパス",
                    value=st.session_state.custom_file_path,
                    key="custom_file_path_input",
                    help="CSV/TXT形式の実験波形データファイル"
                )

                # ファイルアップロード機能
                uploaded_waveform = st.file_uploader(
                    "または波形ファイルをアップロード",
                    type=['csv', 'txt'],
                    key="waveform_file_uploader",
                    help="実験データから取得した波形ファイル"
                )

                if uploaded_waveform is not None:
                    # ファイル検証
                    if uploaded_waveform.size > 10 * 1024 * 1024:  # 10MB制限
                        st.error("❌ ファイルサイズが大きすぎます (最大10MB)")
                    else:
                        try:
                            # アップロードされたファイルを一時的に保存
                            import tempfile
                            import shutil
                            tmp_path = None
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_waveform.name.split(".")[-1]}') as tmp_file:
                                    shutil.copyfileobj(uploaded_waveform, tmp_file)
                                    tmp_path = tmp_file.name

                                st.session_state.custom_file_path = tmp_path
                                st.success(f"✅ ファイルアップロード完了: {uploaded_waveform.name}")

                                # 即座にファイル内容を検証 (ファイルクローズ後)
                                test_data = app_instance.timing_analyzer.waveform_library._load_waveform_from_file(tmp_path)
                                if test_data is None:
                                    st.error("❌ ファイル形式が無効です")
                                    os.unlink(tmp_path)
                                    st.session_state.custom_file_path = ""
                                else:
                                    st.info(f"📊 検証完了: {len(test_data['time'])} データポイント")
                            except Exception as e:
                                st.error(f"❌ ファイル検証エラー: {str(e)}")
                                if tmp_path and os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                                st.session_state.custom_file_path = ""
                        except Exception as e:
                            st.error(f"❌ ファイルアップロードエラー: {str(e)}")

                if st.session_state.custom_file_path and os.path.exists(st.session_state.custom_file_path):
                    st.info(f"📄 選択ファイル: {os.path.basename(st.session_state.custom_file_path)}")

                    # 時間軸単位の選択
                    st.session_state.custom_waveform_time_unit = st.radio(
                        "時間軸の単位",
                        ("ns", "s"),
                        index=0 if st.session_state.get('custom_waveform_time_unit', 'ns') == 'ns' else 1,
                        key="custom_waveform_time_unit_radio",
                        help="カスタム波形ファイルの時間軸単位を選択します。nsに変換されます。"
                    )

                    if st.button("🔍 波形プレビュー", key="waveform_preview_button"):
                        try:
                            # 波形ライブラリで波形をロード
                            waveform_name = f"custom_file_{Path(st.session_state.custom_file_path).stem}"
                            load_success = app_instance.timing_analyzer.waveform_library.load_custom_waveform(
                                st.session_state.custom_file_path, 
                                waveform_name=waveform_name,
                                time_unit=st.session_state.custom_waveform_time_unit
                            )

                            if load_success:
                                waveform_data = app_instance.timing_analyzer.waveform_library.custom_waveforms.get(waveform_name)
                            else:
                                waveform_data = None

                            if waveform_data is not None:
                                # データ品質チェック
                                import numpy as np
                                import matplotlib.pyplot as plt
                                time_data = waveform_data['time']
                                amp_data = waveform_data['intensity']

                                # 基本統計
                                n_points = len(time_data)
                                time_range = time_data[-1] - time_data[0] if n_points > 1 else 0
                                amp_range = np.max(amp_data) - np.min(amp_data)

                                # シンプルな波形プロット
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.plot(time_data, amp_data, 'b-', linewidth=2)
                                ax.set_xlabel('Time (ns)') # 常にnsで表示
                                ax.set_ylabel('Normalized Amplitude')
                                ax.set_title('Custom Waveform Preview')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)

                                # データ統計
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("データ点数", f"{n_points}")
                                with col2:
                                    st.metric("時間範囲", f"{time_range:.3f} ns")
                                with col3:
                                    st.metric("振幅範囲", f"{amp_range:.3f}")

                                # 品質警告
                                if n_points < 10:
                                    st.warning("⚠️ データ点数が少ないです（推奨: 50点以上）")
                                elif n_points < 50:
                                    st.info("💡 より多くのデータ点数があると精度が向上します")

                                if time_range < 1.0:
                                    st.warning("⚠️ 時間範囲が狭いです")
                                if amp_range < 0.1:
                                    st.warning("⚠️ 振幅の変化が小さいです")
                            else:
                                st.error("❌ 波形ファイルの読み込みに失敗しました")
                        except Exception as e:
                            st.error(f"❌ 波形プレビューエラー: {str(e)}")
                else:
                    st.warning("⚠️ 有効な波形ファイルが選択されていません")

            # フィッティング結果表示
            if st.session_state.waveform_fitting_success:
                st.success(f"✅ フィッティング成功 (R² = {st.session_state.waveform_r_squared:.4f})")
            elif hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results is not None:
                st.warning("⚠️ 前回のフィッティング失敗 - 最大値検出を使用")

        # データ回転
        rotation_angle = st.slider("🔄 データ回転 (度)", -180, 180, 0, step=1)

        st.markdown("---")

        # ログ表示パネル
        render_log_display_panel(app_instance.logger_manager)

        return rotation_angle