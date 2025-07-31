import streamlit as st
from src.config.settings import get_config, set_config, save_user_config

def load_preset(config_manager, preset_name: str):
    """プリセット設定を読み込み"""
    try:
        if config_manager.load_preset(preset_name):
            # ConfigManagerから更新された設定を読み込み
            st.session_state.gx_xmin = get_config('analysis.gx_region.xmin')
            st.session_state.gx_xmax = get_config('analysis.gx_region.xmax')
            st.session_state.gx_ymin = get_config('analysis.gx_region.ymin')
            st.session_state.gx_ymax = get_config('analysis.gx_region.ymax')
            st.session_state.lfex_xmin = get_config('analysis.lfex_region.xmin')
            st.session_state.lfex_xmax = get_config('analysis.lfex_region.xmax')
            st.session_state.time_calibration_mode = get_config('time_calibration.mode')
            st.session_state.full_width_time = get_config('time_calibration.full_width_time')
            st.session_state.time_per_pixel = get_config('time_calibration.time_per_pixel')

            # 波形設定の読み込み
            st.session_state.waveform_type = get_config('waveform.type', 'gaussian')
            st.session_state.gaussian_method = get_config('waveform.gaussian.method', 'fixed_pulse')
            st.session_state.gaussian_fwhm = get_config('waveform.gaussian.fwhm', 1.3)
            st.session_state.custom_pulse_enabled = get_config('waveform.custom_pulse.enabled', False)
            st.session_state.custom_file_path = get_config('waveform.custom_file.default_file', '')

            # IMG設定の読み込み
            st.session_state.img_byte_order = get_config('files.img_settings.byte_order', 'auto')

            st.success(f"✅ プリセット「{preset_name}」を読み込みました")
            st.rerun()
        else:
            st.error(f"❌ プリセット「{preset_name}」の読み込みに失敗しました")
    except Exception as e:
        st.error(f"❌ プリセット読み込みエラー: {e}")

def save_current_as_preset(config_manager, sync_session_callback):
    """現在の設定をプリセットとして保存"""
    preset_name = st.text_input("プリセット名を入力", key="preset_name_input")
    if preset_name and st.button("保存実行", key="save_preset_button"):
        try:
            # 現在のセッション状態をConfigManagerに同期
            sync_session_callback()

            # プリセットとして保存
            preset_config = {
                'analysis': {
                    'gx_region': {
                        'xmin': st.session_state.gx_xmin,
                        'xmax': st.session_state.gx_xmax,
                        'ymin': st.session_state.gx_ymin,
                        'ymax': st.session_state.gx_ymax
                    },
                    'lfex_region': {
                        'xmin': st.session_state.lfex_xmin,
                        'xmax': st.session_state.lfex_xmax
                    }
                },
                'time_calibration': {
                    'mode': st.session_state.time_calibration_mode,
                    'full_width_time': st.session_state.full_width_time,
                    'time_per_pixel': st.session_state.time_per_pixel
                },
                'waveform': {
                    'type': st.session_state.waveform_type,
                    'gaussian': {
                        'method': st.session_state.gaussian_method,
                        'fwhm': st.session_state.gaussian_fwhm
                    },
                    'custom_pulse': {
                        'enabled': st.session_state.custom_pulse_enabled
                    },
                    'custom_file': {
                        'default_file': st.session_state.custom_file_path
                    }
                },
                'files': {
                    'img_settings': {
                        'byte_order': st.session_state.img_byte_order
                    }
                }
            }

            if config_manager.save_preset(preset_name, preset_config):
                st.success(f"✅ プリセット「{preset_name}」を保存しました")
            else:
                st.error("❌ プリセットの保存に失敗しました")
        except Exception as e:
            st.error(f"❌ プリセット保存エラー: {e}")
