"""
GXII-LFEX Timing Analysis GUI 設定ファイル
ConfigManagerベースの統一設定システム
"""
from pathlib import Path
from .config_manager import ConfigManager

# グローバル設定マネージャー
_config_manager = None

def get_config_manager() -> ConfigManager:
    """ConfigManagerのシングルトンインスタンスを取得"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

class Settings:
    """
    アプリケーション設定 (ConfigManager互換レイヤー)
    既存コードとの後方互換性を保持
    """
    
    @classmethod
    def get_version(cls):
        """アプリケーションバージョンを取得"""
        config = get_config_manager()
        version = config.get('app.version')
        
        # VERSIONファイルからの動的読み込みも継続サポート
        if not version:
            try:
                version_file = Path(__file__).parent.parent.parent / "VERSION"
                if version_file.exists():
                    version = version_file.read_text().strip()
                    # ConfigManagerにも保存
                    
                else:
                    version = "Unknown"
            except Exception:
                version = "Unknown"
        return version
    
    # 後方互換性のためのプロパティ
    @property
    def APP_NAME(self):
        return get_config_manager().get('app.name')
    
    @property
    def DESCRIPTION(self):
        return get_config_manager().get('app.description')
    
    @property
    def PAGE_TITLE(self):
        return get_config_manager().get('gui.page_title')
    
    @property
    def PAGE_ICON(self):
        return get_config_manager().get('gui.page_icon')
    
    @property
    def LAYOUT(self):
        return get_config_manager().get('gui.layout')
    
    @property
    def INITIAL_SIDEBAR_STATE(self):
        return get_config_manager().get('gui.sidebar_state')
    
    @property
    def PORT_RANGE(self):
        port_range = get_config_manager().get('gui.port_range')
        return tuple(port_range) if port_range else (8501, 8510)
    
    @property
    def DEFAULT_NOISE_THRESHOLD(self):
        return get_config_manager().get('analysis.processing.noise_threshold')
    
    @property
    def DEFAULT_SMOOTHING_WINDOW(self):
        return get_config_manager().get('analysis.processing.smoothing_window')
    
    @property
    def DEFAULT_PROMINENCE(self):
        return get_config_manager().get('analysis.processing.prominence')
    
    @property
    def DEFAULT_FULL_WIDTH_TIME(self):
        return get_config_manager().get('time_calibration.full_width_time')
    
    @property
    def DEFAULT_PIXEL_COUNT(self):
        return get_config_manager().get('time_calibration.pixel_count')
    
    @property
    def DEFAULT_TIME_PER_PIXEL(self):
        return get_config_manager().get('time_calibration.time_per_pixel')
    
    @property
    def SUPPORTED_FORMATS(self):
        return get_config_manager().get('files.supported_formats')
    
    @property
    def OUTPUT_DIR(self):
        return get_config_manager().get('files.output_dir')
    
    @property
    def TEMP_DIR(self):
        return get_config_manager().get('files.temp_dir')
    
    @property 
    def DEFAULT_FIGURE_SIZE(self):
        size = get_config_manager().get('plot.figure_size')
        return tuple(size) if size else (12, 8)
    
    @property
    def DEFAULT_DPI(self):
        return get_config_manager().get('plot.dpi')
    
    @property
    def DEFAULT_LINE_WIDTH(self):
        return get_config_manager().get('plot.line_width')
    
    @property
    def DEFAULT_MARKER_SIZE(self):
        return get_config_manager().get('plot.marker_size')
    
    @property
    def COLORS(self):
        return get_config_manager().get('plot.colors')
    
    @property
    def IMG_DEFAULT_WIDTH(self):
        return get_config_manager().get('files.img_settings.default_width')
    
    @property
    def IMG_DEFAULT_HEIGHT(self):
        return get_config_manager().get('files.img_settings.default_height')
    
    @property
    def IMG_DEFAULT_DTYPE(self):
        return get_config_manager().get('files.img_settings.default_dtype')
    
    @property
    def IMG_BYTE_ORDER(self):
        return get_config_manager().get('files.img_settings.byte_order')
    
    # 静的設定（ConfigManagerに移行しない固定値）
    MAX_PEAKS = 10
    
    # 基準時間設定オプション
    REFERENCE_TIME_MODES = {
        'gxii_peak': 'GXIIピークタイミング',
        'streak_time': 'ストリーク画像時間（t=0基準）',
        'gxii_rise': 'GXIIの立ち上がり（n%）'
    }
    
    # GXIIの立ち上がり検出用デフォルト値
    DEFAULT_GXII_RISE_PERCENTAGE = 10.0  # 10%
    
    # セッション状態のキー
    SESSION_KEYS = {
        'uploaded_file': 'uploaded_file',
        'analysis_results': 'analysis_results',
        'plot_settings': 'plot_settings',
        'user_preferences': 'user_preferences'
    }
    
    # エラーメッセージ
    ERROR_MESSAGES = {
        'file_not_found': "ファイルが見つかりません",
        'unsupported_format': "サポートされていないファイル形式です",
        'data_load_error': "データの読み込みに失敗しました",
        'analysis_error': "解析処理でエラーが発生しました",
        'plot_error': "グラフの生成でエラーが発生しました"
    }

# グローバル設定インスタンス
settings = Settings()

# ConfigManager直接アクセス関数
def get_config(key: str, default=None):
    """設定値を取得する便利関数"""
    return get_config_manager().get(key, default)

def set_config(key: str, value):
    """設定値を設定する便利関数"""
    return get_config_manager().set(key, value)

def save_user_config():
    """ユーザー設定を保存する便利関数"""
    return get_config_manager().save_user_config()