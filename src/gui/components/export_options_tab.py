import streamlit as st
import pandas as pd
import io
from pathlib import Path
from src.config.settings import settings
from src.utils.logger_manager import log_info, log_error
from src.utils.file_utils import FileUtils
from src.core.report_generator import get_report_generator # 追加


def render_export_options_tab(app_instance):
    """エクスポートオプションタブを描画"""
    st.header("💾 エクスポート")

    if st.session_state.analysis_results is None:
        st.warning("⚠️ 先に解析を実行してください")
        return

    st.subheader("📊 データエクスポート")

    # CSVエクスポート
    if st.button("📄 結果をCSVでエクスポート", use_container_width=True):
        _export_results_csv(app_instance)

    # JSONエクスポート
    if st.button("📋 設定をJSONでエクスポート", use_container_width=True):
        _export_settings_json(app_instance)

    # レポート生成セクション
    st.subheader("📄 レポート生成")

    # レポート情報入力欄
    with st.expander("📝 レポート情報入力", expanded=True):
        report_author = st.text_input("レポート作成者名", value=st.session_state.get("report_author", ""))
        st.session_state.report_author = report_author # セッション状態に保存

        sweep_speed_setting = st.text_input("X線ストリークカメラ掃引速度設定 (例: 100 ps/mm)", value=st.session_state.get("sweep_speed_setting", ""))
        st.session_state.sweep_speed_setting = sweep_speed_setting # セッション状態に保存

        laser_comments = st.text_area("レーザーに関する備考/コメント", value=st.session_state.get("laser_comments", ""))
        st.session_state.laser_comments = laser_comments # セッション状態に保存

    # レポート生成オプション
    hide_lfex_2nd_peak_detected = st.checkbox(
        "LFEX 2nd ピーク (検出値) をレポートから非表示にする",
        value=st.session_state.get("hide_lfex_2nd_peak_detected", False),
        help="チェックすると、LFEX 2nd ピークが検出された場合でも、推定値のみがプロットされます。"
    )
    st.session_state.hide_lfex_2nd_peak_detected = hide_lfex_2nd_peak_detected

    # レポート生成ボタン
    if st.button("🚀 レポートを生成 (PDF/画像)", type="primary", use_container_width=True):
        # バージョン情報を取得
        app_version = FileUtils.get_version_from_file(str(Path(__file__).parents[3] / "VERSION"))
        
        _generate_single_page_report(
            app_instance,
            report_author=report_author,
            sweep_speed_setting=sweep_speed_setting,
            laser_comments=laser_comments,
            app_version=app_version,
            hide_lfex_2nd_peak_detected=hide_lfex_2nd_peak_detected
        )

    st.subheader("📁 プロジェクト保存")

    if st.button("💾 プロジェクト全体を保存", use_container_width=True):
        _save_project()

def _export_results_csv(app_instance):
    """結果をCSV形式でエクスポート"""
    import pandas as pd
    import io

    results = st.session_state.analysis_results

    # データフレーム作成
    df_data = {
        "Time_ns": results["streak_time"],
        "GXII_Normalized": results["gxii_norm"],
        "LFEX_Profile": results["lfex_time"]
    }

    df = pd.DataFrame(df_data)

    # メタデータ作成（基準時間情報を含む）
    metadata_dict = {
        "shotid": results["shotid"],
        "gxii_peak_ns": results["gxii_peak"],
        "gxii_peak_relative_ns": results["gxii_peak_relative"],
        "lfex_peak_1_ns": results["max_time_1"],
        "lfex_peak_1_relative_ns": results["lfex_peak_1_relative"],
        "lfex_peak_2_ns": results.get("max_time_2", 0),
        "lfex_peak_2_relative_ns": results.get("lfex_peak_2_relative", 0),
        "timing_difference_ns": results["time_diff"],
        "reference_time_mode": results["reference_time_mode"],
        "reference_time_ns": results["reference_time"],
        "gxii_rise_percentage": results.get("gxii_rise_percentage", 10.0),
        "time_calibration_mode": results["time_calibration_mode"],
        "full_width_time_ns": results["full_width_time"],
        "time_per_pixel_ns": results["time_per_pixel"]
    }

    metadata_df = pd.DataFrame([metadata_dict])

    # CSVに変換（メタデータとデータを分けて出力）
    csv_buffer = io.StringIO()

    # メタデータセクション
    csv_buffer.write("# GXII-LFEX Timing Analysis Results\n")
    csv_buffer.write("# Metadata\n")
    metadata_df.to_csv(csv_buffer, index=False)
    csv_buffer.write("\n# Time Series Data\n")

    # データセクション
    df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 CSVファイルをダウンロード",
        data=csv_buffer.getvalue(),
        file_name=f"analysis_results_{st.session_state.uploaded_file_path.split("/")[-1]}.csv",
        mime="text/csv"
    )

def _export_settings_json(app_instance):
    """設定をJSON形式でエクスポート（ConfigManagerベース）"""
    import json

    # 現在のセッション状態をConfigManagerに保存
    app_instance._sync_session_to_config()

    # ConfigManagerから全設定をエクスポート
    settings_data = app_instance.config_manager.get_all_settings()

    # エクスポート情報を追加
    settings_data["export_info"] = {
        "app_version": settings.get_version(),
        "export_timestamp": str(pd.Timestamp.now()),
        "session_data": {
            "gx_xmin": st.session_state.gx_xmin,
            "gx_xmax": st.session_state.gx_xmax,
            "gx_ymin": st.session_state.gx_ymin,
            "gx_ymax": st.session_state.gx_ymax,
            "lfex_xmin": st.session_state.lfex_xmin,
            "lfex_xmax": st.session_state.lfex_xmax,
            "time_calibration_mode": st.session_state.time_calibration_mode,
            "full_width_time": st.session_state.full_width_time,
            "time_per_pixel": st.session_state.time_per_pixel,
            # 基準時間設定を追加
            "reference_time_mode": st.session_state.reference_time_mode,
            "gxii_rise_percentage": st.session_state.gxii_rise_percentage
        }
    }

    json_str = json.dumps(settings_data, indent=2, ensure_ascii=False)

    st.download_button(
        label="📥 設定ファイルをダウンロード",
        data=json_str,
        file_name="analysis_settings.json",
        mime="application/json"
    )

def _generate_single_page_report(app_instance, report_author, sweep_speed_setting, laser_comments, app_version, hide_lfex_2nd_peak_detected):
    """単一ページレポートを生成"""
    try:
        log_info("単一ページレポート生成開始", "gui.export")

        report_generator = get_report_generator()
        report_bytes = report_generator.generate_single_page_report(
            app_instance,
            report_author,
            sweep_speed_setting,
            laser_comments,
            app_version,
            hide_lfex_2nd_peak_detected
        )

        if report_bytes:
            shot_id = st.session_state.analysis_results.get("shotid", "unknown")
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Timing_Report_{shot_id}_{timestamp}.png" # 仮でPNG

            st.success("✅ レポートが生成されました！")
            st.download_button(
                label="📥 レポートをダウンロード",
                data=report_bytes,
                file_name=filename,
                mime="image/png", # 仮でPNG
                key="single_page_report_download"
            )
        else:
            st.error("❌ レポート生成に失敗しました")

    except Exception as e:
        st.error(f"❌ レポート生成エラー: {str(e)}")
        log_error(f"レポート生成エラー: {str(e)}", "gui.export", exc_info=True)

def _convert_results_to_xsc_format(app_instance, results, data_info):
    """解析結果をXSCショット・サマリー形式に変換"""
    try:
        # 基本的なXSCショットデータ構造を作成
        # session_stateにcurrent_dataがない場合もNoneに初期化
        if not hasattr(st.session_state, 'current_data'):
            st.session_state.current_data = None

        xsc_shot_data = {
            'shot_id': results.get('shotid', data_info['filename']),
            'timestamp': pd.Timestamp.now(),
            'raw_data': st.session_state.current_data if st.session_state.current_data is not None else None,
            'processed_data': st.session_state.current_data,  # 処理済みデータ（現在は同じ）
            'time_axis': results['streak_time'],
            'gxii_profile': results['gxii_norm'],
            'lfex_profile': results['lfex_time'],
            'regions': {
                'gxii': {
                    'xmin': st.session_state.gx_xmin,
                    'xmax': st.session_state.gx_xmax,
                    'ymin': st.session_state.gx_ymin,
                    'ymax': st.session_state.gx_ymax
                },
                'lfex': {
                    'xmin': st.session_state.lfex_xmin,
                    'xmax': st.session_state.lfex_xmax,
                    'ymin': st.session_state.gx_ymin,
                    'ymax': st.session_state.gx_ymax
                }
            },
            'peaks': {
                'gxii_peak': {
                    'time': results['gxii_peak'],
                    'relative_time': results['gxii_peak_relative'],
                    'value': results.get('gxii_peak_value', 1.0)
                },
                'lfex_peak_1': {
                    'time': results['max_time_1'],
                    'relative_time': results['lfex_peak_1_relative'],
                    'value': results['max_value_1']
                },
                'lfex_peak_2': {
                    'time': results.get('max_time_2', 0),
                    'relative_time': results.get('lfex_peak_2_relative', 0),
                    'value': results.get('max_value_2', 0)
                }
            },
            'analysis_params': {
                'time_calibration_mode': st.session_state.time_calibration_mode,
                'full_width_time': st.session_state.full_width_time,
                'time_per_pixel': st.session_state.time_per_pixel,
                'ma_window': results.get('ma_window', 20),
                'peak_threshold': results.get('peak_threshold', 0.1),
                # 基準時間情報を追加
                'reference_time_mode': results['reference_time_mode'],
                'reference_time': results['reference_time'],
                'gxii_rise_percentage': results.get('gxii_rise_percentage', 10.0)
            },
            'metadata': {
                'filename': data_info['filename'],
                'filepath': data_info['filepath'],
                'app_version': settings.get_version(),
                'analysis_timestamp': str(pd.Timestamp.now()),
                'timing_difference': results['time_diff']
            }
        }

        log_info(f"XSCショットデータ変換完了: shot_id={xsc_shot_data['shot_id']}", "gui.export")
        return xsc_shot_data

    except Exception as e:
        log_error(f"XSCショットデータ変換エラー: {str(e)}", "gui.export", exc_info=True)
        raise

def _save_project():
    """プロジェクト全体を保存"""
    st.info("🚧 プロジェクト保存機能は実装予定です")
