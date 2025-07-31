"""
統一設定管理モジュール
全アプリケーション設定を階層的に管理
"""
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import copy
from dataclasses import dataclass, asdict
import warnings


@dataclass
class ValidationRule:
    """設定値バリデーションルール"""
    type_check: Optional[type] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = False


class ConfigManager:
    """統一設定管理クラス"""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        ConfigManagerを初期化
        
        Args:
            config_dir: 設定ファイルディレクトリパス
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # 設定データ
        self._default_config: Dict = {}
        self._user_config: Dict = {}
        self._current_config: Dict = {}
        
        # バリデーションルール
        self._validation_rules: Dict[str, ValidationRule] = {}
        
        # 設定ファイルパス
        self.default_config_file = self.config_dir / "default.yml"
        self.user_config_file = self.config_dir / "user.yml"
        
        # 初期化
        self._setup_default_config()
        self._setup_validation_rules()
        self._load_configs()
        self._update_version_from_file()
        
    def _setup_default_config(self):
        """デフォルト設定を定義"""
        self._default_config = {
            'app': {
                'name': 'GXII-LFEX Timing Analysis GUI',
                'version': '0.0.0',  # VERSIONファイルから動的に更新される
                'description': 'GXII-LFEX実験のタイミング解析GUI',
                'debug_mode': False
            },
            'analysis': {
                'gx_region': {
                    'xmin': 520,
                    'xmax': 600,
                    'ymin': 4,
                    'ymax': 1020
                },
                'lfex_region': {
                    'xmin': 700,
                    'xmax': 800
                },
                'processing': {
                    'noise_threshold': 0.1,
                    'smoothing_window': 5,
                    'prominence': 0.1,
                    'ma_window': 20,
                    'peak_threshold': 0.1
                },
                'peak_detection': {
                    'mode': '2ピーク検出',
                    'selection_method': '最大強度'
                },
                'angle': 0.0
            },
            'time_calibration': {
                'mode': '全幅指定',
                'full_width_time': 4.8,
                'pixel_count': 1024,
                'time_per_pixel': 0.004688
            },
            'reference_time': {
                'mode': 'gxii_peak'
            },
            'plot': {
                'figure_size': [12, 8],
                'dpi': 100,
                'line_width': 2,
                'marker_size': 6,
                'colors': {
                    'primary': '#1f77b4',
                    'secondary': '#ff7f0e',
                    'success': '#2ca02c',
                    'warning': '#d62728',
                    'gxii': '#2ca02c',
                    'lfex': '#1f77b4',
                    'peak': '#d62728'
                }
            },
            'files': {
                'supported_formats': ['.txt', '.csv', '.dat', '.img'],
                'output_dir': 'output',
                'temp_dir': 'temp',
                'img_settings': {
                    'default_width': 1024,
                    'default_height': 1024,
                    'default_dtype': 'uint16',
                    'byte_order': 'little'
                }
            },
            'gui': {
                'page_title': 'GXII-LFEX Timing Analysis',
                'page_icon': '🔬',
                'layout': 'wide',
                'sidebar_state': 'expanded',
                'port_range': [8501, 8510]
            }
        }
        
    def _setup_validation_rules(self):
        """バリデーションルールを設定"""
        self._validation_rules = {
            'analysis.gx_region.xmin': ValidationRule(type_check=int, min_value=0, max_value=2048),
            'analysis.gx_region.xmax': ValidationRule(type_check=int, min_value=0, max_value=2048),
            'analysis.gx_region.ymin': ValidationRule(type_check=int, min_value=0, max_value=2048),
            'analysis.gx_region.ymax': ValidationRule(type_check=int, min_value=0, max_value=2048),
            'analysis.lfex_region.xmin': ValidationRule(type_check=int, min_value=0, max_value=2048),
            'analysis.lfex_region.xmax': ValidationRule(type_check=int, min_value=0, max_value=2048),
            'analysis.processing.noise_threshold': ValidationRule(type_check=float, min_value=0.0, max_value=1.0),
            'analysis.processing.smoothing_window': ValidationRule(type_check=int, min_value=1, max_value=100),
            'analysis.processing.ma_window': ValidationRule(type_check=int, min_value=1, max_value=100),
            'analysis.processing.peak_threshold': ValidationRule(type_check=float, min_value=0.0, max_value=1.0),
            'analysis.peak_detection.mode': ValidationRule(type_check=str, allowed_values=['2ピーク検出', '1ピーク検出']),
            'analysis.peak_detection.selection_method': ValidationRule(type_check=str, allowed_values=['最大強度', '最初のピーク', '最後のピーク']),
            'analysis.angle': ValidationRule(type_check=float, min_value=-45.0, max_value=45.0),
            'time_calibration.mode': ValidationRule(type_check=str, allowed_values=['全幅指定', '1pixel指定', 'フィッティング']),
            'time_calibration.full_width_time': ValidationRule(type_check=float, min_value=0.1, max_value=100.0),
            'time_calibration.time_per_pixel': ValidationRule(type_check=float, min_value=0.001, max_value=1.0),
            'reference_time.mode': ValidationRule(type_check=str, allowed_values=['gxii_peak', 'streak_time', 'gxii_rise', 'absolute', 'manual', 'custom_t0', 'lfex_peak']),
            'plot.figure_size': ValidationRule(type_check=list),
            'plot.dpi': ValidationRule(type_check=int, min_value=50, max_value=300),
        }
    
    def _load_configs(self):
        """設定ファイルを読み込み"""
        # デフォルト設定ファイルの作成/読み込み
        if not self.default_config_file.exists():
            self._save_config_file(self.default_config_file, self._default_config)
        else:
            try:
                self._default_config = self._load_config_file(self.default_config_file)
            except Exception as e:
                warnings.warn(f"デフォルト設定読み込み失敗: {e}")
        
        # ユーザー設定ファイルの読み込み
        if self.user_config_file.exists():
            try:
                self._user_config = self._load_config_file(self.user_config_file)
            except Exception as e:
                warnings.warn(f"ユーザー設定読み込み失敗: {e}")
                self._user_config = {}
        
        # 現在の設定を統合
        self._merge_configs()
    
    def _load_config_file(self, filepath: Path) -> Dict:
        """設定ファイルを読み込み"""
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.suffix in ['.yml', '.yaml']:
                return yaml.safe_load(f) or {}
            elif filepath.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"サポートされていない設定ファイル形式: {filepath.suffix}")
    
    def _save_config_file(self, filepath: Path, config: Dict):
        """設定ファイルを保存"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            if filepath.suffix in ['.yml', '.yaml']:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            elif filepath.suffix == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _merge_configs(self):
        """デフォルト設定とユーザー設定を統合"""
        self._current_config = copy.deepcopy(self._default_config)
        self._deep_merge(self._current_config, self._user_config)
    
    def _deep_merge(self, base: Dict, override: Dict):
        """辞書を深くマージ"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        階層キーで設定値を取得
        
        Args:
            key: 階層キー (例: 'analysis.gx_region.xmin')
            default: デフォルト値
            
        Returns:
            設定値
        """
        keys = key.split('.')
        current = self._current_config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any, save_user_config: bool = True) -> bool:
        """
        階層キーで設定値を設定
        
        Args:
            key: 階層キー
            value: 設定値
            save_user_config: ユーザー設定として保存するか
            
        Returns:
            設定成功フラグ
        """
        # バリデーション
        if not self._validate_value(key, value):
            return False
        
        # 現在の設定を更新
        keys = key.split('.')
        current = self._current_config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        
        # ユーザー設定を更新
        if save_user_config:
            user_current = self._user_config
            for k in keys[:-1]:
                if k not in user_current:
                    user_current[k] = {}
                user_current = user_current[k]
            user_current[keys[-1]] = value
            
            # ユーザー設定ファイルに保存
            self.save_user_config()
        
        return True
    
    def _validate_value(self, key: str, value: Any) -> bool:
        """設定値をバリデーション"""
        if key not in self._validation_rules:
            return True  # ルールがない場合は通す
        
        rule = self._validation_rules[key]
        
        # 型チェック
        if rule.type_check and not isinstance(value, rule.type_check):
            warnings.warn(f"設定値の型が不正です: {key} = {value} (期待型: {rule.type_check.__name__})")
            return False
        
        # 範囲チェック
        if rule.min_value is not None and hasattr(value, '__lt__') and value < rule.min_value:
            warnings.warn(f"設定値が最小値を下回っています: {key} = {value} (最小値: {rule.min_value})")
            return False
            
        if rule.max_value is not None and hasattr(value, '__gt__') and value > rule.max_value:
            warnings.warn(f"設定値が最大値を上回っています: {key} = {value} (最大値: {rule.max_value})")
            return False
        
        # 許可値チェック
        if rule.allowed_values and value not in rule.allowed_values:
            warnings.warn(f"設定値が許可値リストにありません: {key} = {value} (許可値: {rule.allowed_values})")
            return False
        
        return True
    
    def save_user_config(self):
        """ユーザー設定を保存"""
        self._save_config_file(self.user_config_file, self._user_config)
    
    def reset_user_config(self):
        """ユーザー設定をリセット"""
        self._user_config = {}
        if self.user_config_file.exists():
            self.user_config_file.unlink()
        self._merge_configs()
    
    def load_preset(self, preset_name: str) -> bool:
        """プリセット設定を読み込み"""
        preset_file = self.config_dir / "presets" / f"{preset_name}.yml"
        if not preset_file.exists():
            warnings.warn(f"プリセットファイルが見つかりません: {preset_file}")
            return False
        
        try:
            preset_config = self._load_config_file(preset_file)
            self._user_config.update(preset_config)
            self._merge_configs()
            self.save_user_config()
            return True
        except Exception as e:
            warnings.warn(f"プリセット読み込み失敗: {e}")
            return False
    
    def save_preset(self, preset_name: str, config_subset: Optional[Dict] = None) -> bool:
        """現在の設定をプリセットとして保存"""
        preset_dir = self.config_dir / "presets"
        preset_dir.mkdir(exist_ok=True)
        preset_file = preset_dir / f"{preset_name}.yml"
        
        try:
            save_config = config_subset if config_subset else self._user_config
            self._save_config_file(preset_file, save_config)
            return True
        except Exception as e:
            warnings.warn(f"プリセット保存失敗: {e}")
            return False
    
    def list_presets(self) -> List[str]:
        """利用可能なプリセット一覧を取得"""
        preset_dir = self.config_dir / "presets"
        if not preset_dir.exists():
            return []
        
        presets = []
        for preset_file in preset_dir.glob("*.yml"):
            presets.append(preset_file.stem)
        return sorted(presets)
    
    def get_all_settings(self) -> Dict:
        """全設定を取得"""
        return copy.deepcopy(self._current_config)
    
    def get_user_settings(self) -> Dict:
        """ユーザー設定のみを取得"""
        return copy.deepcopy(self._user_config)
    
    def export_settings(self, filepath: Union[str, Path], format: str = 'yaml') -> bool:
        """設定をファイルにエクスポート"""
        try:
            export_path = Path(filepath)
            if format.lower() == 'yaml':
                export_path = export_path.with_suffix('.yml')
            elif format.lower() == 'json':
                export_path = export_path.with_suffix('.json')
            
            self._save_config_file(export_path, self._current_config)
            return True
        except Exception as e:
            warnings.warn(f"設定エクスポート失敗: {e}")
            return False
    
    def _update_version_from_file(self):
        """ルートのVERSIONファイルからバージョンを読み取って更新"""
        try:
            # プロジェクトルートのVERSIONファイルパス
            version_file = self.config_dir.parent / "VERSION"
            
            if version_file.exists():
                with open(version_file, 'r', encoding='utf-8') as f:
                    version = f.read().strip()
                
                # 現在の設定にバージョンを更新
                if 'app' not in self._current_config:
                    self._current_config['app'] = {}
                self._current_config['app']['version'] = version
                
                # デフォルト設定も更新
                if 'app' not in self._default_config:
                    self._default_config['app'] = {}
                self._default_config['app']['version'] = version
                
                # default.ymlファイルも更新
                self._save_config_file(self.default_config_file, self._default_config)
                
        except Exception as e:
            warnings.warn(f"VERSIONファイル読み込み失敗: {e}")


# グローバルConfigManagerインスタンス
config_manager = ConfigManager()
