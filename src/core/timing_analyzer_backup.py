"""
タイミング解析モジュール
GXII-LFEX実験のタイミング解析機能を提供
"""
import numpy as np
from scipy import signal, interpolate
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from typing import Tuple, Optional, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..config.settings import settings
from ..utils.plot_manager import PlotManager
from ..utils.logger_manager import log_info, log_error, log_debug, log_warning


class TimingAnalyzer:
    """タイミング解析クラス"""
    
    def __init__(self):
        """初期化"""
        self.plot_manager = PlotManager()
    
    def gaussian_fixpulse(self, x, amp, mean):
        """ガウシアン関数（固定幅）"""
        sigma = 1
        return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))
    
    def moving_average(self, data, window_size):
        """移動平均"""
        if window_size <= 1:
            return data
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')
    
    def calculate_gxii_rise_time(self, gxii_norm: np.ndarray, streak_time: np.ndarray, rise_percentage: float) -> float:
        """
        GXIIの立ち上がり時間を計算（n%閾値）
        
        Args:
            gxii_norm: 正規化されたGXII信号
            streak_time: 時間軸
            rise_percentage: 立ち上がり閾値（%）
            
        Returns:
            立ち上がり時間
        """
        # 最大値のn%を閾値として計算
        max_intensity = np.max(gxii_norm)
        threshold = max_intensity * (rise_percentage / 100.0)
        
        # 閾値を超えた最初のインデックスを検出
        rise_indices = np.where(gxii_norm > threshold)[0]
        if len(rise_indices) > 0:
            rise_index = rise_indices[0]
            if rise_index < len(streak_time):
                return streak_time[rise_index]
        
        # 検出できない場合は0を返す
        return 0.0
    
    def calculate_reference_time(self, reference_mode: str, gxii_peak: float, gxii_norm: np.ndarray, 
                                streak_time: np.ndarray, rise_percentage: float = 10.0) -> float:
        """
        基準時間を計算
        
        Args:
            reference_mode: 基準時間モード ("gxii_peak", "streak_time", "gxii_rise")
            gxii_peak: GXIIピーク時間
            gxii_norm: 正規化されたGXII信号
            streak_time: 時間軸
            rise_percentage: 立ち上がり閾値（%）
            
        Returns:
            基準時間
        """
        if reference_mode == "gxii_peak":
            return gxii_peak
        elif reference_mode == "streak_time":
            return 0.0  # ストリーク画像時間（t=0基準）
        elif reference_mode == "gxii_rise":
            return self.calculate_gxii_rise_time(gxii_norm, streak_time, rise_percentage)
        else:
            # デフォルトはGXIIピーク
            return gxii_peak
    
    def analyze_timing(self, filepath: str, filename: str, angle: float = 0, 
                      gx_xmin: int = 520, gx_xmax: int = 600, 
                      gx_ymin: int = 4, gx_ymax: int = 1020,
                      lfex_xmin: int = 700, lfex_xmax: int = 800,
                      ma_window: int = 20, peak_threshold: float = 0.1,
                      peak_detection_mode: str = "2ピーク検出",
                      peak_selection_method: str = "最大強度",
                      time_calibration_mode: str = "全幅指定",
                      full_width_time: float = None,
                      time_per_pixel: float = None,
                      reference_time_mode: str = "gxii_peak",
                      gxii_rise_percentage: float = 10.0) -> Tuple[Optional[Dict], Optional[str]]:
from scipy.ndimage import rotate
from typing import Tuple, Optional, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..config.settings import settings
from ..utils.plot_manager import PlotManager
from ..utils.logger_manager import log_info, log_error, log_debug, log_warning


class TimingAnalyzer:
    """タイミング解析クラス"""
    
    def __init__(self):
        self.noise_threshold = settings.DEFAULT_NOISE_THRESHOLD
        self.smoothing_window = settings.DEFAULT_SMOOTHING_WINDOW
        self.prominence = settings.DEFAULT_PROMINENCE
        self.plot_manager = PlotManager()
    
    @staticmethod
    def moving_average(x: np.ndarray, w: int) -> np.ndarray:
        """移動平均を計算"""
        return np.convolve(x, np.ones(w), 'valid') / w
    
    @staticmethod
    def gaussian(x: np.ndarray, amp: float, mean: float, stddev: float) -> np.ndarray:
        """ガウス関数"""
        return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    
    @staticmethod
    def gaussian_fixpulse(x: np.ndarray, amp: float, mean: float) -> np.ndarray:
        """固定パルス幅のガウス関数"""
        return amp * np.exp(-((x - mean) ** 2) / (2 * (1.3/2.35482) ** 2))
    
    def load_and_preview_data(self, filepath: str, angle: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        データを読み込んでプレビュー用に表示する関数
        
        Args:
            filepath: ファイルパス
            angle: 回転角度
            
        Returns:
            データ配列とエラーメッセージのタプル
        """
        log_debug(f"データ読み込みとプレビュー準備開始: {filepath}, angle={angle}", "timing_analyzer")
        
        try:
            from .data_loader import DataLoader
            
            # データ読み込み
            data_array, error = DataLoader.load_data_file(filepath)
            if error:
                log_error(f"データ読み込みエラー: {error}", "timing_analyzer")
                return None, error
            
            log_info(f"データ読み込み成功: shape={data_array.shape}", "timing_analyzer")
            
            # データの回転
            if angle != 0.0:
                log_debug(f"データ回転実行: {angle}度", "timing_analyzer")
                data_array = rotate(data_array, angle, reshape=True)
                log_debug(f"データ回転完了: 新shape={data_array.shape}", "timing_analyzer")
            
            return data_array, None
        except Exception as e:
            error_msg = str(e)
            log_error(f"データ読み込み・プレビュー準備エラー: {error_msg}", "timing_analyzer", exc_info=True)
            return None, error_msg
    
    def create_preview_plot(self, data_array: np.ndarray, gx_xmin: int = 520, gx_xmax: int = 600, 
                           gx_ymin: int = 4, gx_ymax: int = 1020, 
                           lfex_xmin: int = 700, lfex_xmax: int = 800) -> plt.Figure:
        """
        データプレビュープロットを作成（PlotManagerを使用）
        
        Args:
            data_array: データ配列
            gx_xmin, gx_xmax, gx_ymin, gx_ymax: GXII領域
            lfex_xmin, lfex_xmax: LFEX領域（Y範囲はGXIIと同じ）
            
        Returns:
            matplotlib図オブジェクト
        """
        return self.plot_manager.create_preview_plot(
            data_array=data_array,
            title="Data Preview",
            gx_params={'xmin': gx_xmin, 'xmax': gx_xmax, 
                      'ymin': gx_ymin, 'ymax': gx_ymax},
            lfex_params={'xmin': lfex_xmin, 'xmax': lfex_xmax}
        )
    
    def extract_region_data(self, data_array: np.ndarray, xmin: int, xmax: int, 
                           ymin: int, ymax: int) -> np.ndarray:
        """
        指定された領域のデータを抽出
        
        Args:
            data_array: データ配列
            xmin, xmax, ymin, ymax: 抽出領域の座標
            
        Returns:
            抽出されたデータ配列
        """
        # 境界チェック
        ymin = max(0, min(ymin, data_array.shape[0] - 1))
        ymax = max(ymin + 1, min(ymax, data_array.shape[0]))
        xmin = max(0, min(xmin, data_array.shape[1] - 1))
        xmax = max(xmin + 1, min(xmax, data_array.shape[1]))
        
        return data_array[ymin:ymax, xmin:xmax]
    
    def analyze_timing(self, filepath: str, filename: str, angle: float = 0.0,
                      gx_xmin: int = 520, gx_xmax: int = 600, 
                      gx_ymin: int = 4, gx_ymax: int = 1020,
                      lfex_xmin: int = 700, lfex_xmax: int = 800,
                      ma_window: int = 20, peak_threshold: float = 0.1,
                      peak_detection_mode: str = "2ピーク検出",
                      peak_selection_method: str = "最大強度",
                      time_calibration_mode: str = "全幅指定",
                      full_width_time: float = None,
                      time_per_pixel: float = None) -> Tuple[Optional[Dict], Optional[str]]:
        """
        元のアルゴリズムを忠実に再現したタイミング解析
        
        Args:
            filepath: ファイルパス
            filename: ファイル名
            angle: 回転角度
            gx_xmin, gx_xmax, gx_ymin, gx_ymax: GXII領域の座標
            lfex_xmin, lfex_xmax: LFEX領域の座標
            ma_window: 移動平均ウィンドウサイズ
            peak_threshold: ピーク検出閾値
            peak_detection_mode: ピーク検出モード
            peak_selection_method: ピーク選択方法
            time_calibration_mode: 時間校正モード ("全幅指定" または "1pixel指定")
            full_width_time: 全幅の時間 (ns) - 全幅指定モード時に使用
            time_per_pixel: 1pxlあたりの時間 (ns) - 1pixel指定モード時に使用
            reference_time_mode: 基準時間モード ("gxii_peak", "streak_time", "gxii_rise")
            gxii_rise_percentage: GXIIの立ち上がり閾値 (%)
            
        Returns:
            解析結果辞書とエラーメッセージのタプル
        """
        log_info(f"タイミング解析開始: {filename}", "timing_analyzer")
        log_debug(f"解析パラメータ: angle={angle}, ma_window={ma_window}, peak_threshold={peak_threshold}", "timing_analyzer")
        
        try:
            # データ読み込み
            data_array, error = self.load_and_preview_data(filepath, angle)
            if error:
                return None, error
            
            log_debug(f"領域設定: GXII[{gx_xmin}-{gx_xmax}, {gx_ymin}-{gx_ymax}], LFEX[{lfex_xmin}-{lfex_xmax}, {gx_ymin}-{gx_ymax}]", "timing_analyzer")
            
            # 領域の分離（元のコードと同じ手順）
            gxii_data = data_array[gx_ymin:gx_ymax, gx_xmin:gx_xmax]
            lfex_data = data_array[gx_ymin:gx_ymax, lfex_xmin:lfex_xmax]
            log_debug(f"領域データ抽出完了: GXII={gxii_data.shape}, LFEX={lfex_data.shape}", "timing_analyzer")
            
            # 信号処理（元のコードと完全同一）
            total_time = np.average(gxii_data, 1)  # axis=1 で横方向平均
            total_time = self.moving_average(total_time, ma_window)
            lfex_time = np.average(lfex_data, 1)
            lfex_time = self.moving_average(lfex_time, ma_window)
            gxii_time = total_time - lfex_time
            gxii_norm = gxii_time / max(self.moving_average(gxii_time, ma_window))
            
            # 時間軸の設定（ユーザー設定可能に変更）
            if time_calibration_mode == "全幅指定":
                # 全幅時間から1pxlあたりの時間を計算
                if full_width_time is None:
                    full_width_time = settings.DEFAULT_FULL_WIDTH_TIME
                pixel_count = len(gxii_norm)
                t_pxl = full_width_time / pixel_count
            else:  # "1pixel指定"
                # 1pxlあたりの時間を直接使用
                if time_per_pixel is None:
                    t_pxl = settings.DEFAULT_TIME_PER_PIXEL
                else:
                    t_pxl = time_per_pixel
            
            log_debug(f"時間校正設定: mode={time_calibration_mode}, t_pxl={t_pxl:.6f}ns", "timing_analyzer")
            # Use a fixed length streak_time array to avoid floating-point
            # rounding issues with np.arange when using a non-integer step.
            streak_time = np.arange(len(gxii_norm)) * t_pxl
            
            # LFEXピーク検出（元のコードと同一）
            maxid = signal.argrelmax(lfex_time, order=2)[0].tolist()
            peak_value = [lfex_time[c] for c in maxid]
            log_debug(f"LFEXピーク検出: {len(peak_value)}個のピーク発見", "timing_analyzer")
            
            if peak_detection_mode == "1ピーク検出":
                # 1ピーク検出モード
                if len(peak_value) == 0:
                    error_msg = "No peaks detected in LFEX data"
                    log_warning(error_msg, "timing_analyzer")
                    return None, error_msg
                
                log_debug(f"1ピーク検出モード: 選択方法={peak_selection_method}", "timing_analyzer")
                
                if peak_selection_method == "最大強度":
                    max_value_1 = max(peak_value)
                    max_index_1 = peak_value.index(max_value_1)
                    max_time_1 = streak_time[maxid[max_index_1]]
                elif peak_selection_method == "最初のピーク":
                    max_value_1 = peak_value[0]
                    max_time_1 = streak_time[maxid[0]]
                elif peak_selection_method == "最後のピーク":
                    max_value_1 = peak_value[-1]
                    max_time_1 = streak_time[maxid[-1]]
                else:
                    # デフォルトは最大強度
                    max_value_1 = max(peak_value)
                    max_index_1 = peak_value.index(max_value_1)
                    max_time_1 = streak_time[maxid[max_index_1]]
                
                max_value_2 = 0
                max_time_2 = 0
                
            else:
                # 2ピーク検出モード（従来の動作）
                log_debug("2ピーク検出モード", "timing_analyzer")
                
                if len(peak_value) >= 2:
                    max_value_1 = max(peak_value)
                    max_index_1 = peak_value.index(max_value_1)
                    max_time_1 = streak_time[maxid[max_index_1]]
                    maxid.pop(max_index_1)
                    peak_value.pop(max_index_1)
                    max_value_2 = max(peak_value)
                    max_time_2 = streak_time[maxid[peak_value.index(max_value_2)]]
                    log_debug(f"2ピーク検出成功: ピーク1={max_time_1:.3f}ns, ピーク2={max_time_2:.3f}ns", "timing_analyzer")
                elif len(peak_value) == 1:
                    max_value_1 = peak_value[0]
                    max_time_1 = streak_time[maxid[0]]
                    max_value_2 = 0
                    max_time_2 = 0
                    log_warning("LFEXピークが1個のみ検出されました", "timing_analyzer")
                else:
                    error_msg = "No peaks detected in LFEX data"
                    log_warning(error_msg, "timing_analyzer")
                    return None, error_msg
            
            # GXIIピーク検出（ガウシアンフィッティング）
            gxii_rise = np.where(gxii_norm > peak_threshold)[0][0] if len(np.where(gxii_norm > peak_threshold)[0]) > 0 else 0
            
            initial_guess = [1, np.mean(streak_time)]
            try:
                params, covariance = curve_fit(self.gaussian_fixpulse, streak_time, gxii_norm, p0=initial_guess)
                amp, mean = params
                gxii_peak = mean
                log_debug(f"GXIIガウシアンフィッティング成功: peak={gxii_peak:.3f}ns", "timing_analyzer")
            except:
                gxii_peak = streak_time[np.argmax(gxii_norm)]
                params = [1, gxii_peak]
                log_warning(f"GXIIガウシアンフィッティング失敗、最大値で代用: peak={gxii_peak:.3f}ns", "timing_analyzer")
            
            # タイミング計算
            time_diff = max_time_1 - gxii_peak
            log_info(f"タイミング計算完了: GXII={gxii_peak:.3f}ns, LFEX={max_time_1:.3f}ns, 差={time_diff:.3f}ns", "timing_analyzer")
            
            # ショットID生成
            filename_base = filepath.rsplit('/', 1)[1].rsplit('.', 1)[0]
            shotid = "G" + filename_base.split('Va', 1)[1] if 'Va' in filename_base else filename_base
            log_debug(f"ショットID生成: {shotid}", "timing_analyzer")
            
            # 結果をまとめる
            results = {
                'filename': filename,
                'shotid': shotid,
                'gxii_peak': gxii_peak,
                'max_time_1': max_time_1,
                'max_time_2': max_time_2,
                'max_value_1': max_value_1,
                'max_value_2': max_value_2,
                'time_diff': time_diff,
                'streak_time': streak_time,
                'gxii_norm': gxii_norm,
                'lfex_time': lfex_time,
                'data_array': data_array,
                'total_time': total_time,
                'params': params,
                'gxii_rise_time': streak_time[gxii_rise] if gxii_rise < len(streak_time) else 0,
                # 座標変換のためのパラメーター - XSC表示に必要
                'gx_xmin': gx_xmin,
                'gx_xmax': gx_xmax,
                'gx_ymin': gx_ymin,
                'gx_ymax': gx_ymax,
                'lfex_xmin': lfex_xmin,
                'lfex_xmax': lfex_xmax,
                'ma_window': ma_window,
                # 時間校正パラメータ
                't_pxl': t_pxl,
                'time_calibration_mode': time_calibration_mode,
                'full_width_time': full_width_time if time_calibration_mode == "全幅指定" else t_pxl * len(gxii_norm),
                'time_per_pixel': t_pxl
            }
            
            log_info(f"タイミング解析完了: {filename}", "timing_analyzer")
            return results, None
            
        except Exception as e:
            error_msg = str(e)
            log_error(f"タイミング解析エラー: {error_msg}", "timing_analyzer", exc_info=True)
            return None, error_msg
    
    def create_plots(self, results: Dict) -> List[plt.Figure]:
        """
        タイミング解析結果のプロット作成（PlotManagerを使用）
        
        Args:
            results: analyze_timing()の結果辞書
            
        Returns:
            matplotlib図オブジェクトのリスト
        """
        return self.plot_manager.create_timing_analysis_plots(results)
    
    def calculate_profile(self, region_data: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        指定された軸に沿ってプロファイルを計算
        
        Args:
            region_data: 領域データ
            axis: 平均を取る軸 (0: Y軸方向の平均, 1: X軸方向の平均)
            
        Returns:
            プロファイル配列
        """
        return np.mean(region_data, axis=axis)
    
    def find_peaks(self, profile: np.ndarray, prominence: Optional[float] = None, 
                   height: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        プロファイルからピークを検出
        
        Args:
            profile: プロファイル配列
            prominence: ピークの突出度
            height: ピークの最小高さ
            
        Returns:
            ピーク位置とプロパティのタプル
        """
        if prominence is None:
            prominence = self.prominence
        
        peaks, properties = signal.find_peaks(profile, prominence=prominence, height=height)
        return peaks, properties
    
    def fit_gaussian(self, x_data: np.ndarray, y_data: np.ndarray, 
                     initial_guess: Optional[List[float]] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        ガウス関数をフィッティング
        
        Args:
            x_data: X軸データ
            y_data: Y軸データ
            initial_guess: 初期推定値 [amp, mean, stddev]
            
        Returns:
            フィッティングパラメータとR二乗値のタプル
        """
        try:
            if initial_guess is None:
                # 自動初期推定
                amp_guess = np.max(y_data)
                mean_guess = x_data[np.argmax(y_data)]
                stddev_guess = len(x_data) / 10
                initial_guess = [amp_guess, mean_guess, stddev_guess]
            
            popt, _ = curve_fit(self.gaussian, x_data, y_data, p0=initial_guess)
            
            # R二乗値を計算
            y_pred = self.gaussian(x_data, *popt)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return popt, r_squared
        except Exception:
            return None, None
    
    def calculate_timing_difference(self, gxii_peak_pos: float, lfex_peak_pos: float, 
                                   pixel_to_time_factor: float = 1.0) -> float:
        """
        タイミング差を計算
        
        Args:
            gxii_peak_pos: GXIIピーク位置
            lfex_peak_pos: LFEXピーク位置
            pixel_to_time_factor: ピクセルから時間への変換係数
            
        Returns:
            タイミング差
        """
        return (lfex_peak_pos - gxii_peak_pos) * pixel_to_time_factor
    
    def smooth_data(self, data: np.ndarray, window_size: Optional[int] = None) -> np.ndarray:
        """
        データを平滑化
        
        Args:
            data: 入力データ
            window_size: 平滑化ウィンドウサイズ
            
        Returns:
            平滑化されたデータ
        """
        if window_size is None:
            window_size = self.smoothing_window
        
        if len(data) < window_size:
            return data
        
        return self.moving_average(data, window_size)
    
    def interpolate_profile(self, x_data: np.ndarray, y_data: np.ndarray, 
                           interpolation_factor: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        プロファイルを補間して解像度を向上
        
        Args:
            x_data: X軸データ
            y_data: Y軸データ
            interpolation_factor: 補間倍率
            
        Returns:
            補間されたX軸とY軸データのタプル
        """
        f = interpolate.interp1d(x_data, y_data, kind='cubic', bounds_error=False, fill_value=0)
        x_new = np.linspace(x_data[0], x_data[-1], len(x_data) * interpolation_factor)
        y_new = f(x_new)
        return x_new, y_new