"""
波形ライブラリモジュール
カスタム波形システムの核となる波形関数とフィッティング機能を提供
"""
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import curve_fit
from typing import Tuple, Optional, List, Dict, Any, Callable
import os
from pathlib import Path

from ..config.settings import settings
from ..utils.logger_manager import log_info, log_error, log_debug, log_warning


class WaveformLibrary:
    """波形ライブラリクラス - 各種波形関数とフィッティング機能を提供"""
    
    def __init__(self):
        """初期化"""
        self.custom_waveforms = {}  # カスタム波形データのキャッシュ
        self.waveform_functions = {
            'gaussian': self._gaussian_function,
            'gaussian_fwhm': self._gaussian_fwhm_function,
            'gaussian_fixpulse': self._gaussian_fixpulse_function,
            'custom_pulse': self._custom_pulse_function
        }
        
    # ========== ガウシアン波形関数群 ==========
    
    @staticmethod
    def _gaussian_function(x: np.ndarray, amp: float, mean: float, sigma: float) -> np.ndarray:
        """標準ガウシアン関数"""
        return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
    
    @staticmethod
    def _gaussian_fwhm_function(x: np.ndarray, amp: float, mean: float, fwhm: float) -> np.ndarray:
        """FWHM入力対応ガウシアン関数"""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM → σ変換
        return WaveformLibrary._gaussian_function(x, amp, mean, sigma)
    
    @staticmethod
    def _gaussian_fixpulse_function(x: np.ndarray, amp: float, mean: float) -> np.ndarray:
        """固定パルス幅ガウシアン関数（元のコードと互換性保持）"""
        sigma = 1.3 / 2.35482  # 元のハードコード値: σ ≈ 0.553 ns
        return WaveformLibrary._gaussian_function(x, amp, mean, sigma)
    
    def _custom_pulse_function(self, x: np.ndarray, amp: float, mean: float,
                              waveform_name: str = 'default') -> np.ndarray:
        """カスタムパルス波形関数

        範囲外の時間では振幅0となるようゼロパディングした波形を返す。
        """
        if waveform_name not in self.custom_waveforms:
            # カスタム波形が見つからない場合はエラーを発生させるか、適切なデフォルト値を返す
            # ここではエラーを発生させ、呼び出し元で処理させる
            raise ValueError(f"Custom waveform '{waveform_name}' not found in library.")
        
        waveform_data = self.custom_waveforms[waveform_name]
        time_data = waveform_data['time']
        intensity_data = waveform_data['intensity']
        
        # 時間軸をシフトして、meanの位置にピークが来るように調整
        original_peak_time = time_data[np.argmax(intensity_data)]
        shifted_time = time_data - original_peak_time + mean
        
        # 補間関数を作成
        # 範囲外ではゼロとなるよう fill_value=0.0 を指定
        interp_func = interpolate.interp1d(
            shifted_time, intensity_data,
            kind='cubic', bounds_error=False, fill_value=0.0
        )
        
        # 振幅調整して返す
        return amp * interp_func(x)
    
    # ========== カスタム波形データ管理 ==========
    
    def _load_waveform_from_file(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        ファイルから波形データを読み込み（テスト用）
        
        Args:
            file_path: 波形データファイルパス
            
        Returns:
            読み込み成功時、{'time': time_array, 'amplitude': amplitude_array}のディクショナリ
            失敗時はNone
        """
        try:
            # ファイル存在確認
            if not os.path.exists(file_path):
                return None
                
            # ファイル形式に応じて読み込み
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.txt'):
                data = pd.read_csv(file_path, delimiter='\t')
            else:
                return None
            
            # データ列の確認と抽出
            if 'time' in data.columns and 'amplitude' in data.columns:
                time_data = data['time'].values
                amplitude_data = data['amplitude'].values
            elif len(data.columns) >= 2:
                # 最初の2列を時間と振幅として使用
                time_data = data.iloc[:, 0].values
                amplitude_data = data.iloc[:, 1].values
            else:
                return None
            
            return {
                'time': time_data,
                'amplitude': amplitude_data
            }
            
        except Exception:
            return None

    def load_custom_waveform(self, file_path: str, waveform_name: str = None,
                           normalize: bool = True, interpolation_factor: int = 10, time_unit: str = 'ns') -> bool:
        """
        カスタム波形データをファイルから読み込み
        
        Args:
            file_path: 波形データファイルパス
            waveform_name: 波形の識別名（Noneの場合、ファイル名から自動生成）
            normalize: 正規化するかどうか
            interpolation_factor: 補間倍率
            
        Returns:
            読み込み成功時True
        """
        try:
            if waveform_name is None:
                waveform_name = Path(file_path).stem
                
            log_info(f"カスタム波形読み込み開始: {file_path}", "waveform_library")
            
            # ファイル形式に応じて読み込み
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.txt'):
                data = pd.read_csv(file_path, delimiter='\t')
            else:
                # numpy形式も対応
                data = np.loadtxt(file_path)
                if data.shape[1] == 2:
                    data = pd.DataFrame(data, columns=['time', 'intensity'])
                else:
                    log_error(f"未対応のファイル形式: {file_path}", "waveform_library")
                    return False
            
            # データ列の確認と抽出
            if 'time' in data.columns and 'intensity' in data.columns:
                time_data = data['time'].values
                intensity_data = data['intensity'].values
            elif len(data.columns) >= 2:
                # 最初の2列を時間と強度として使用
                time_data = data.iloc[:, 0].values
                intensity_data = data.iloc[:, 1].values
                log_warning(f"列名を自動検出: 時間={data.columns[0]}, 強度={data.columns[1]}", "waveform_library")
            else:
                log_error(f"データ列が不足: {file_path}", "waveform_library")
                return False
            
            # データの前処理
            processed_data = self._preprocess_waveform(
                time_data, intensity_data, normalize, interpolation_factor, time_unit
            )
            
            # カスタム波形データとして保存
            self.custom_waveforms[waveform_name] = processed_data
            
            log_info(f"カスタム波形読み込み完了: {waveform_name} "
                    f"({len(processed_data['time'])}点, FWHM: {processed_data['fwhm']:.3f}ns)", 
                    "waveform_library")
            
            return True
            
        except Exception as e:
            log_error(f"カスタム波形読み込みエラー: {str(e)}", "waveform_library", exc_info=True)
            return False
    
    def _preprocess_waveform(self, time_data: np.ndarray, intensity_data: np.ndarray,
                           normalize: bool = True, interpolation_factor: int = 10, time_unit: str = 'ns') -> Dict[str, Any]:
        """
        波形データの前処理
        
        Args:
            time_data: 時間データ
            intensity_data: 強度データ
            normalize: 正規化するかどうか
            interpolation_factor: 補間倍率
            time_unit: 時間軸の単位 ('ns' or 's')
            
        Returns:
            前処理済みデータ辞書
        """
        # 時間軸の単位変換 (全てnsに統一)
        if time_unit == 's':
            time_data = time_data * 1e9  # 秒をナノ秒に変換
            log_info("時間軸を秒からナノ秒に変換しました。", "waveform_library")
        # ソート（時間順に並び替え）
        sort_indices = np.argsort(time_data)
        time_sorted = time_data[sort_indices]
        intensity_sorted = intensity_data[sort_indices]
        
        # 補間（データ点数を増やして滑らかに）
        if interpolation_factor > 1:
            time_interp = np.linspace(time_sorted[0], time_sorted[-1], 
                                    len(time_sorted) * interpolation_factor)
            interp_func = interpolate.interp1d(time_sorted, intensity_sorted, 
                                             kind='cubic', bounds_error=False, fill_value=0.0)
            intensity_interp = interp_func(time_interp)
            time_sorted = time_interp
            intensity_sorted = intensity_interp
        
        # 正規化
        if normalize:
            intensity_sorted = intensity_sorted / np.max(intensity_sorted)
        
        # FWHM計算
        fwhm = self._calculate_fwhm(time_sorted, intensity_sorted)
        
        # ピーク位置計算
        peak_index = np.argmax(intensity_sorted)
        peak_time = time_sorted[peak_index]
        
        return {
            'time': time_sorted,
            'intensity': intensity_sorted,
            'fwhm': fwhm,
            'peak_time': peak_time,
            'peak_intensity': intensity_sorted[peak_index],
            'time_range': (time_sorted[0], time_sorted[-1])
        }
    
    def _calculate_fwhm(self, time_data: np.ndarray, intensity_data: np.ndarray) -> float:
        """FWHM（半値全幅）を計算"""
        try:
            max_intensity = np.max(intensity_data)
            half_max = max_intensity / 2.0
            
            # 半値を超える点を見つける
            above_half = intensity_data >= half_max
            if not np.any(above_half):
                return 0.0
            
            # 左端と右端を見つける
            indices = np.where(above_half)[0]
            left_index = indices[0]
            right_index = indices[-1]
            
            # より正確なFWHM計算（線形補間を使用）
            if left_index > 0:
                # 左側の補間
                x1, x2 = time_data[left_index-1], time_data[left_index]
                y1, y2 = intensity_data[left_index-1], intensity_data[left_index]
                left_time = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            else:
                left_time = time_data[left_index]
            
            if right_index < len(time_data) - 1:
                # 右側の補間
                x1, x2 = time_data[right_index], time_data[right_index+1]
                y1, y2 = intensity_data[right_index], intensity_data[right_index+1]
                right_time = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            else:
                right_time = time_data[right_index]
            
            return right_time - left_time
            
        except Exception:
            log_warning("FWHM計算に失敗しました", "waveform_library")
            return 0.0
    
    # ========== フィッティング機能 ==========
    
    def fit_waveform(self, x_data: np.ndarray, y_data: np.ndarray,
                     waveform_type: str = 'gaussian_fixpulse',
                     initial_guess: Optional[List[float]] = None,
                     bounds: Optional[Tuple[List[float], List[float]]] = None,
                     enforce_positive_amp: bool = True,
                     amp_bounds: Optional[Tuple[float, float]] = (0.8, 1.2),
                     **kwargs) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
        """
        指定された波形関数でフィッティングを実行
        
        Args:
            x_data: X軸データ（時間）
            y_data: Y軸データ（強度）
            waveform_type: 波形タイプ ('gaussian', 'gaussian_fwhm', 'gaussian_fixpulse', 'custom_pulse')
            initial_guess: 初期推定値
            bounds: パラメータ境界値
            enforce_positive_amp: 振幅を正の範囲に制限するかどうか
            amp_bounds: 振幅を1.0付近に制約する範囲 (下限, 上限)
            **kwargs: 追加パラメータ（custom_pulse用のwaveform_nameなど）
            
        Returns:
            (フィッティングパラメータ, R二乗値, エラーメッセージ)のタプル
        """
        try:
            if waveform_type not in self.waveform_functions:
                return None, None, f"未対応の波形タイプ: {waveform_type}"
            
            # データ検証
            if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
                return None, None, "データが空です"
            
            if len(x_data) != len(y_data):
                return None, None, "x_dataとy_dataの長さが一致しません"
            
            # 波形関数を取得
            waveform_func = self.waveform_functions[waveform_type]
            
            # カスタムパルスの場合、波形名を渡しつつピークを合わせる
            if waveform_type == 'custom_pulse':
                waveform_name = kwargs.get('waveform_name', 'default')
                # 読み込んだ波形が正規化されていない場合に備えて
                amp_scale = 1.0
                waveform_data = self.custom_waveforms.get(waveform_name)
                if waveform_data is not None:
                    peak_amp = waveform_data.get('peak_intensity', 1.0)
                    if peak_amp != 0:
                        amp_scale = 1.0 / peak_amp

                fitted_func = lambda x, amp, mean: waveform_func(x, amp * amp_scale, mean, waveform_name)
            else:
                fitted_func = waveform_func
            
            # 初期推定値の自動生成
            if initial_guess is None:
                initial_guess = self._generate_initial_guess(x_data, y_data, waveform_type)
            
            if initial_guess is None:
                return None, None, "初期推定値の生成に失敗しました"
            
            log_debug(f"フィッティング開始: type={waveform_type}, initial_guess={initial_guess}", "waveform_library")
            
            # フィッティング実行
            if bounds is None:
                lower = [-np.inf] * len(initial_guess)
                upper = [np.inf] * len(initial_guess)
            else:
                lower, upper = [list(b) for b in bounds]

            if enforce_positive_amp:
                lower[0] = max(lower[0], 1e-4)
            if amp_bounds is not None:
                lower[0] = max(lower[0], amp_bounds[0])
                upper[0] = min(upper[0], amp_bounds[1])

            bounds = (lower, upper)

            popt, pcov = curve_fit(
                fitted_func, x_data, y_data,
                p0=initial_guess, bounds=bounds,
                maxfev=kwargs.get('max_iterations', 1000)
            )
            
            # R二乗値計算
            y_pred = fitted_func(x_data, *popt)
            r_squared = self._calculate_r_squared(y_data, y_pred)
            
            log_debug(f"フィッティング成功: params={popt}, R²={r_squared:.4f}", "waveform_library")
            
            return popt, r_squared, None
            
        except Exception as e:
            error_msg = f"フィッティングエラー: {str(e)}"
            log_warning(error_msg, "waveform_library")
            return None, None, error_msg
    
    def _generate_initial_guess(self, x_data: np.ndarray, y_data: np.ndarray, 
                              waveform_type: str) -> List[float]:
        """初期推定値を自動生成"""
        try:
            max_intensity = np.max(y_data)
            peak_position = x_data[np.argmax(y_data)]
            
            if waveform_type in ['gaussian_fixpulse', 'custom_pulse']:
                # 振幅と平均のみ
                return [max_intensity, peak_position]
            elif waveform_type == 'gaussian':
                # 振幅、平均、標準偏差
                sigma_guess = (x_data[-1] - x_data[0]) / 6  # データ範囲の1/6をσの初期値とする
                return [max_intensity, peak_position, sigma_guess]
            elif waveform_type == 'gaussian_fwhm':
                # 振幅、平均、FWHM
                fwhm_guess = (x_data[-1] - x_data[0]) / 3  # データ範囲の1/3をFWHMの初期値とする
                return [max_intensity, peak_position, fwhm_guess]
            else:
                return [max_intensity, peak_position]
        except Exception as e:
            log_error(f"初期推定値生成エラー: {str(e)}", "waveform_library")
            return None
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """決定係数（R²）を計算"""
        try:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        except Exception:
            return 0.0
    
    # ========== ユーティリティ関数 ==========
    
    def get_waveform_function(self, waveform_type: str, **kwargs) -> Callable:
        """指定された波形関数を取得"""
        if waveform_type == 'custom_pulse':
            waveform_name = kwargs.get('waveform_name', 'default')
            return lambda x, amp, mean: self._custom_pulse_function(x, amp, mean, waveform_name)
        elif waveform_type in self.waveform_functions:
            return self.waveform_functions[waveform_type]
        else:
            log_warning(f"未対応の波形タイプ: {waveform_type}。gaussian_fixpulseを使用します。", "waveform_library")
            return self.waveform_functions['gaussian_fixpulse']
    
    def list_available_waveforms(self) -> Dict[str, Any]:
        """利用可能な波形のリストを取得"""
        return {
            'built_in': list(self.waveform_functions.keys()),
            'custom': list(self.custom_waveforms.keys()),
            'custom_details': {name: {
                'fwhm': data['fwhm'],
                'peak_time': data['peak_time'],
                'time_range': data['time_range']
            } for name, data in self.custom_waveforms.items()}
        }
    
    def get_waveform_info(self, waveform_name: str) -> Optional[Dict[str, Any]]:
        """特定の波形の詳細情報を取得"""
        if waveform_name in self.custom_waveforms:
            return self.custom_waveforms[waveform_name].copy()
        elif waveform_name in self.waveform_functions:
            return {
                'type': 'built_in',
                'name': waveform_name,
                'parameters': self._get_function_parameters(waveform_name)
            }
        else:
            return None
    
    def _get_function_parameters(self, waveform_type: str) -> List[str]:
        """波形関数のパラメータリストを取得"""
        param_map = {
            'gaussian': ['amplitude', 'mean', 'sigma'],
            'gaussian_fwhm': ['amplitude', 'mean', 'fwhm'],
            'gaussian_fixpulse': ['amplitude', 'mean'],
            'custom_pulse': ['amplitude', 'mean']
        }
        return param_map.get(waveform_type, ['amplitude', 'mean'])
    
    def save_waveform_data(self, waveform_name: str, file_path: str) -> bool:
        """
        カスタム波形データをファイルに保存
        """
        try:
            if waveform_name not in self.custom_waveforms:
                log_error(f"波形データが見つかりません: {waveform_name}", "waveform_library")
                return False
            
            data = self.custom_waveforms[waveform_name]
            df = pd.DataFrame({
                'time': data['time'],
                'intensity': data['intensity']
            })
            
            df.to_csv(file_path, index=False)
            log_info(f"波形データ保存完了: {file_path}", "waveform_library")
            return True
            
        except Exception as e:
            log_error(f"波形データ保存エラー: {str(e)}", "waveform_library", exc_info=True)
            return False


# ========== モジュールレベル関数（後方互換性のため） ==========

def create_waveform_library() -> WaveformLibrary:
    """波形ライブラリインスタンスを作成"""
    return WaveformLibrary()

def load_experimental_waveforms(library: WaveformLibrary, data_dir: str = None) -> int:
    """実験波形データを一括読み込み"""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "waveforms" / "experimental"
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        log_warning(f"実験データディレクトリが見つかりません: {data_dir}", "waveform_library")
        return 0
    
    loaded_count = 0
    for file_path in data_dir.glob("*.csv"):
        if library.load_custom_waveform(str(file_path)):
            loaded_count += 1
    
    for file_path in data_dir.glob("*.txt"):
        if library.load_custom_waveform(str(file_path)):
            loaded_count += 1
    
    log_info(f"実験波形データ読み込み完了: {loaded_count}個のファイル", "waveform_library")
    return loaded_count