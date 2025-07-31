"""
メインGUIアプリケーションモジュール
Streamlitベースの新しいアーキテクチャ
"""
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# 親ディレクトリをパスに追加
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.config.settings import settings, get_config_manager, get_config, set_config, save_user_config
from src.core.data_loader import DataLoader
from src.core.timing_analyzer import TimingAnalyzer
from src.core.waveform_library import WaveformLibrary # 追加
from src.utils.file_utils import FileUtils
from src.utils.plot_manager import PlotManager, PlotTheme
from src.utils.logger_manager import get_logger_manager, log_info, log_error, log_warning, log_debug
from src.core.report_generator import ReportGenerator

# UIコンポーネントのインポート
from src.gui.components.header import render_header
from src.gui.components.sidebar import render_sidebar
from src.gui.components.data_preview_tab import render_data_preview_tab
from src.gui.components.analysis_execution_tab import render_analysis_execution_tab
from src.gui.components.results_display_tab import render_results_display_tab
from src.gui.components.export_options_tab import render_export_options_tab


class GXIILFEXApp:
    """GXII-LFEX タイミング解析アプリケーション"""
    
    def __init__(self):
        # LoggerManagerの初期化（最初に実行）
        self.logger_manager = get_logger_manager()
        log_info("GXII-LFEX アプリケーション初期化開始", "gui.app")
        
        # WaveformLibraryのインスタンスを一つだけ作成し、共有する
        self.waveform_library = WaveformLibrary()

        self.data_loader = DataLoader()
        self.timing_analyzer = TimingAnalyzer(waveform_library=self.waveform_library)
        self.file_utils = FileUtils()
        
        # PlotManagerもセッション状態で管理して重複初期化を防ぐ
        if 'plot_manager' not in st.session_state:
            st.session_state.plot_manager = PlotManager()
        self.plot_manager = st.session_state.plot_manager
        
        self.config_manager = get_config_manager()

        # ReportGeneratorのインスタンスをセッション状態で管理し、waveform_libraryを渡す
        if 'report_generator' not in st.session_state:
            st.session_state.report_generator = ReportGenerator(waveform_library=self.waveform_library)
        self.report_generator = st.session_state.report_generator
        
        # セッション状態の初期化
        self._initialize_session_state()
        
        log_info("GXII-LFEX アプリケーション初期化完了", "gui.app")
    
    def _initialize_session_state(self):
        """セッション状態を初期化（ConfigManagerベース）"""
        # ファイル関連
        if 'uploaded_file_path' not in st.session_state:
            st.session_state.uploaded_file_path = None
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        
        # 解析パラメータ（ConfigManagerから初期値を読み込み）
        if 'gx_xmin' not in st.session_state:
            st.session_state.gx_xmin = get_config('analysis.gx_region.xmin', 520)
        if 'gx_xmax' not in st.session_state:
            st.session_state.gx_xmax = get_config('analysis.gx_region.xmax', 600)
        if 'gx_ymin' not in st.session_state:
            st.session_state.gx_ymin = get_config('analysis.gx_region.ymin', 4)
        if 'gx_ymax' not in st.session_state:
            st.session_state.gx_ymax = get_config('analysis.gx_region.ymax', 1020)
        if 'lfex_xmin' not in st.session_state:
            st.session_state.lfex_xmin = get_config('analysis.lfex_region.xmin', 700)
        if 'lfex_xmax' not in st.session_state:
            st.session_state.lfex_xmax = get_config('analysis.lfex_region.xmax', 800)
        
        # 時間校正設定（ConfigManagerから読み込み）
        if 'time_calibration_mode' not in st.session_state:
            st.session_state.time_calibration_mode = get_config('time_calibration.mode', '全幅指定')
        if 'full_width_time' not in st.session_state:
            st.session_state.full_width_time = get_config('time_calibration.full_width_time', 4.8)
        if 'time_per_pixel' not in st.session_state:
            st.session_state.time_per_pixel = get_config('time_calibration.time_per_pixel', 0.004688)
        
        # 基準時間設定（統一版）
        if 'gxii_rise_percentage' not in st.session_state:
            st.session_state.gxii_rise_percentage = get_config('analysis.gxii_rise_percentage', 10.0)
        
        # 波形設定（ConfigManagerから読み込み）
        if 'waveform_type' not in st.session_state:
            st.session_state.waveform_type = get_config('waveform.type', 'gaussian')
        if 'gaussian_method' not in st.session_state:
            st.session_state.gaussian_method = get_config('waveform.gaussian.method', 'fixed_pulse')
        if 'gaussian_fwhm' not in st.session_state:
            st.session_state.gaussian_fwhm = get_config('waveform.gaussian.fwhm', 1.3)
        if 'custom_pulse_enabled' not in st.session_state:
            st.session_state.custom_pulse_enabled = get_config('waveform.custom_pulse.enabled', False)
        if 'custom_file_path' not in st.session_state:
            st.session_state.custom_file_path = get_config('waveform.custom_file.default_file', '')
        if 'waveform_r_squared' not in st.session_state:
            st.session_state.waveform_r_squared = 0.0
        if 'waveform_fitting_success' not in st.session_state:
            st.session_state.waveform_fitting_success = False
        
        # IMG設定（ConfigManagerから読み込み）
        if 'img_byte_order' not in st.session_state:
            st.session_state.img_byte_order = get_config('files.img_settings.byte_order', 'auto')
        if 'img_byte_order_changed' not in st.session_state:
            st.session_state.img_byte_order_changed = False
        
        # 解析結果
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
    
    def configure_page(self):
        """ページ設定を構成（ConfigManagerベース）"""
        st.set_page_config(
            page_title=get_config('gui.page_title', 'GXII-LFEX Timing Analysis'),
            page_icon=get_config('gui.page_icon', '🔬'),
            layout=get_config('gui.layout', 'wide'),
            initial_sidebar_state=get_config('gui.sidebar_state', 'expanded')
        )
        
        # カスタムCSS
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stAlert > div {
            padding: 0.5rem 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 日本語フォント設定
        PlotTheme.setup_japanese_font()
    
    def render_main_content(self, rotation_angle):
        """メインコンテンツを描画"""
        if st.session_state.uploaded_file_path is None:
            st.info("👈 サイドバーからデータファイルを選択してください")
            return
        
        # タブ構成
        tab1, tab2, tab3, tab4 = st.tabs(["📊 データプレビュー", "🔍 解析実行", "📈 結果表示", "💾 エクスポート"])
        
        with tab1:
            render_data_preview_tab(self, rotation_angle)
        
        with tab2:
            render_analysis_execution_tab(self)
        
        with tab3:
            render_results_display_tab(self)
        
        with tab4:
            render_export_options_tab(self)
    
    def _sync_session_to_config(self):
        """セッション状態をConfigManagerに同期"""
        # 解析領域設定
        set_config('analysis.gx_region.xmin', st.session_state.gx_xmin)
        set_config('analysis.gx_region.xmax', st.session_state.gx_xmax)
        set_config('analysis.gx_region.ymin', st.session_state.gx_ymin)
        set_config('analysis.gx_region.ymax', st.session_state.gx_ymax)
        set_config('analysis.lfex_region.xmin', st.session_state.lfex_xmin)
        set_config('analysis.lfex_region.xmax', st.session_state.lfex_xmax)
        
        # 時間校正設定
        set_config('time_calibration.mode', st.session_state.time_calibration_mode)
        set_config('time_calibration.full_width_time', st.session_state.full_width_time)
        set_config('time_calibration.time_per_pixel', st.session_state.time_per_pixel)
        
        # 波形設定
        set_config('waveform.type', st.session_state.waveform_type)
        set_config('waveform.gaussian.method', st.session_state.gaussian_method)
        set_config('waveform.gaussian.fwhm', st.session_state.gaussian_fwhm)
        set_config('waveform.custom_pulse.enabled', st.session_state.custom_pulse_enabled)
        set_config('waveform.custom_file.default_file', st.session_state.custom_file_path)
        
        # IMG ファイル設定
        set_config('files.img_settings.byte_order', st.session_state.img_byte_order)
        
        # ユーザー設定を保存
        save_user_config()
    
    def run(self):
        """アプリケーションを実行"""
        # 初回実行時のみログを出力
        if 'app_run_initialized' not in st.session_state:
            log_info("アプリケーション実行開始", "gui.app")
            st.session_state.app_run_initialized = True
        
        self.configure_page()
        render_header()
        
        # サイドバー描画
        rotation_angle = render_sidebar(self)
        
        # メインコンテンツ描画
        self.render_main_content(rotation_angle)

        # セッションの設定をユーザー設定として保存
        # 各ウィジェットの変更が毎回ConfigManagerに反映されるようにする
        self._sync_session_to_config()


def main():
    """メイン関数"""
    # Streamlitのセッション状態を使用してアプリインスタンスを管理
    # これにより、UI操作による再実行時にも初期化ログが重複しない
    if 'app_instance' not in st.session_state:
        st.session_state.app_instance = GXIILFEXApp()
    
    st.session_state.app_instance.run()


if __name__ == "__main__":
    main()
