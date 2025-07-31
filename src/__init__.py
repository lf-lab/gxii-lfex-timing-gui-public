"""
GXII-LFEX Timing Analysis GUI パッケージ
"""

__version__ = "1.9.0"
__author__ = "GXII-LFEX Team"
__description__ = "GXII-LFEX実験のタイミング解析GUI"

# パッケージの初期化時に基本設定を読み込み
try:
    from .config.settings import settings
except ImportError:
    # 設定ファイルが見つからない場合のフォールバック
    class DefaultSettings:
        APP_NAME = "GXII-LFEX Timing Analysis GUI"
        VERSION = "1.9.0"
    
    settings = DefaultSettings()

# 主要クラスのインポート（オプション）
try:
    from .core.data_loader import DataLoader
    from .core.timing_analyzer import TimingAnalyzer
    from .utils.file_utils import FileUtils
    from .utils.plot_utils import PlotUtils
except ImportError:
    # 依存関係が不足している場合はスキップ
    pass

__all__ = [
    'settings',
    'DataLoader', 
    'TimingAnalyzer',
    'FileUtils',
    'PlotUtils'
]