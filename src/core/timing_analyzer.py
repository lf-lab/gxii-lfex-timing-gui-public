"""
タイミング解析モジュール
GXII-LFEX実験のタイミング解析機能を提供
"""
import numpy as np
import os
from pathlib import Path
from scipy import signal, interpolate
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from typing import Tuple, Optional, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..config.settings import settings
from ..utils.plot_manager import PlotManager
from ..utils.logger_manager import log_info, log_error, log_debug, log_warning
from ..utils.file_utils import FileUtils
from .waveform_library import WaveformLibrary


class TimingAnalyzer:
    """タイミング解析クラス"""
    
    def __init__(self, waveform_library: Optional[WaveformLibrary] = None):
        """初期化"""
        self.plot_manager = PlotManager()
        self.waveform_library = waveform_library if waveform_library is not None else WaveformLibrary()
    
    def moving_average(self, data, window_size):
        """
        移動平均
        """
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
                                streak_time: np.ndarray, rise_percentage: float = 10.0,
                                waveform_name: Optional[str] = None) -> float:
        """
        基準時間を計算
        
        Args:
            reference_mode: 基準時間モード ("gxii_peak", "lfex_peak", "absolute", "manual", "gxii_rise")
            gxii_peak: GXIIピーク時間
            gxii_norm: 正規化されたGXII信号
            streak_time: 時間軸
            rise_percentage: 立ち上がり閾値（%）
            
        Returns:
            基準時間
        """
        from src.config.settings import get_config
        
        if reference_mode == "gxii_peak":
            return gxii_peak
        elif reference_mode == "lfex_peak":
            # LFEXピークは後で設定される（max_time_1）
            return 0.0  # プレースホルダー
        elif reference_mode == "absolute":
            return get_config('reference_time.absolute_value', 0.0)
        elif reference_mode == "manual":
            return get_config('reference_time.manual_value', 0.0)
        elif reference_mode == "gxii_rise":
            return self.calculate_gxii_rise_time(gxii_norm, streak_time, rise_percentage)
        elif reference_mode == "streak_time":
            return 0.0  # ストリーク画像時間（t=0基準）
        elif reference_mode == "custom_t0":
            if waveform_name and waveform_name in self.waveform_library.custom_waveforms:
                wf_data = self.waveform_library.custom_waveforms[waveform_name]
                peak_time = wf_data.get('peak_time', 0.0)
                return gxii_peak - peak_time
            return gxii_peak
        else:
            # デフォルトはGXIIピーク
            return gxii_peak
    
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
    
    def analyze_timing(self, filepath: str, filename: str, angle: float = 0, 
                      gx_xmin: int = 520, gx_xmax: int = 600, 
                      gx_ymin: int = 4, gx_ymax: int = 1020,
                      lfex_xmin: int = 700, lfex_xmax: int = 800,
                      ma_window: int = 20, peak_threshold: float = 0.1,
                      peak_detection_mode: str = "2ピーク検出",
                      peak_selection_method: str = "最大強度",
                      fixed_offset_value: Optional[float] = None,
                      time_calibration_mode: str = "全幅指定",
                      full_width_time: float = None,
                      time_per_pixel: float = None,
                      reference_time_mode: str = "gxii_peak",
                      gxii_rise_percentage: float = 10.0,
                      waveform_type: str = "gaussian",
                      waveform_config: Dict[str, Any] = None,
                      shot_datetime_str: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
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
            time_calibration_mode: 時間校正モード ("全幅指定", "1pixel指定", "フィッティング")
            full_width_time: 全幅の時間 (ns) - 全幅指定モード時に使用
            time_per_pixel: 1pxlあたりの時間 (ns) - 1pixel指定モード時に使用
            reference_time_mode: 基準時間モード ("gxii_peak", "streak_time", "gxii_rise")
            gxii_rise_percentage: GXIIの立ち上がり閾値 (%)
            waveform_type: 波形タイプ ("gaussian", "custom_pulse", "custom_file")
            waveform_config: 波形設定辞書（各波形タイプの詳細パラメータ）
            
        Returns:
            解析結果辞書とエラーメッセージのタプル
        """
        log_info(f"タイミング解析開始: {filename}", "timing_analyzer")
        log_debug(f"解析パラメータ: angle={angle}, ma_window={ma_window}, peak_threshold={peak_threshold}", "timing_analyzer")
        log_debug(f"基準時間設定: mode={reference_time_mode}, rise_percentage={gxii_rise_percentage}%", "timing_analyzer")
        
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
            elif time_calibration_mode == "1pixel指定":
                # 1pxlあたりの時間を直接使用
                if time_per_pixel is None:
                    t_pxl = settings.DEFAULT_TIME_PER_PIXEL
                else:
                    t_pxl = time_per_pixel
            else:  # フィッティング
                if full_width_time is None:
                    full_width_time = settings.DEFAULT_FULL_WIDTH_TIME
                pixel_count = len(gxii_norm)
                search_range = np.linspace(full_width_time * 0.5, full_width_time * 1.5, 51)
                best_r2 = -np.inf
                best_t_pxl = full_width_time / pixel_count
                best_width = full_width_time
                best_peak = None
                best_params = None
                best_success = False
                best_name = None
                for width in search_range:
                    t_cand = width / pixel_count
                    st_time = np.arange(len(gxii_norm)) * t_cand
                    peak_c, params_c, success_c, name_c, r2_c = self._detect_gxii_peak_with_waveform(
                        st_time, gxii_norm, waveform_type, waveform_config)
                    if r2_c is None:
                        r2_c = -np.inf
                    if r2_c > best_r2:
                        best_r2 = r2_c
                        best_t_pxl = t_cand
                        best_width = width
                        best_peak = peak_c
                        best_params = params_c
                        best_success = success_c
                        best_name = name_c
                        streak_time = st_time
                t_pxl = best_t_pxl
                full_width_time = best_width
                gxii_peak = best_peak
                fitting_params = best_params
                fitting_success = best_success
                waveform_name = best_name
                waveform_r_squared = best_r2
            
            log_debug(f"時間校正設定: mode={time_calibration_mode}, t_pxl={t_pxl:.6f}ns", "timing_analyzer")
            # np.arange with floating step can introduce rounding errors.
            # Compute the streak time using a fixed element count to ensure
            # the array length matches gxii_norm precisely.
            streak_time = np.arange(len(gxii_norm)) * t_pxl
            
            # LFEXピーク検出（元のコードと同一）
            maxid = signal.argrelmax(lfex_time, order=2)[0].tolist()
            peak_value = [lfex_time[c] for c in maxid]
            log_debug(f"LFEXピーク検出: {len(peak_value)}個のピーク発見", "timing_analyzer")
            
            # offset_valueを初期化（すべてのモードで利用可能にするため）
            offset_value = fixed_offset_value if fixed_offset_value is not None else 0.24
            
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
                lfex_peak_2_status = "not_applicable" # 1ピーク検出モードでは適用外

            elif peak_detection_mode == "固定オフセット2ピーク検出":
                log_debug("固定オフセット2ピーク検出モード", "timing_analyzer")
                
                if len(peak_value) == 0:
                    error_msg = "No primary peak detected in LFEX data for fixed offset mode"
                    log_warning(error_msg, "timing_analyzer")
                    return None, error_msg
                
                # 主要LFEXピークの検出 (最も強度の高いもの)
                max_value_1 = max(peak_value)
                max_index_1 = peak_value.index(max_value_1)
                max_time_1 = streak_time[maxid[max_index_1]]
                
                # 二次ピークの推定位置を計算 (オフセット値: 正の場合は後に、負の場合は前に)
                expected_t2 = max_time_1 + offset_value
                
                # 探索ウィンドウを設定 (±0.1ns)
                search_window_min = expected_t2 - 0.1
                search_window_max = expected_t2 + 0.1
                
                # 時間軸の範囲内に制限
                search_window_min = max(search_window_min, streak_time[0])
                search_window_max = min(search_window_max, streak_time[-1])
                
                offset_direction = "後" if offset_value >= 0 else "前"
                log_debug(f"主要LFEXピーク: {max_time_1:.3f}ns, オフセット: {offset_value:.3f}ns ({offset_direction}), 推定二次ピーク位置: {expected_t2:.3f}ns, 探索ウィンドウ: [{search_window_min:.3f}, {search_window_max:.3f}]ns", "timing_analyzer")
                
                # 探索ウィンドウ内のデータに限定
                window_indices = np.where((streak_time >= search_window_min) & (streak_time <= search_window_max))[0]
                
                max_value_2 = 0
                max_time_2 = 0
                lfex_peak_2_status = "estimated" # デフォルトは推定
                
                if len(window_indices) > 0:
                    lfex_time_window = lfex_time[window_indices]
                    streak_time_window = streak_time[window_indices]
                    
                    # 探索ウィンドウ内でピークを検出
                    window_maxid = signal.argrelmax(lfex_time_window, order=2)[0].tolist()
                    
                    if len(window_maxid) > 0:
                        # 探索ウィンドウ内で最も高いピークを二次ピークとする
                        window_peak_values = [lfex_time_window[idx] for idx in window_maxid]
                        max_value_2 = max(window_peak_values)
                        max_index_2_in_window = window_peak_values.index(max_value_2)
                        max_time_2 = streak_time_window[window_maxid[max_index_2_in_window]]
                        lfex_peak_2_status = "detected"
                        log_info(f"二次LFEXピーク検出: {max_time_2:.3f}ns (detected)", "timing_analyzer")
                    else:
                        # argrelmaxで見つからない場合、ウィンドウ内の最大値をピークとする
                        if len(lfex_time_window) > 0:
                            max_index_in_window = np.argmax(lfex_time_window)
                            max_value_2 = lfex_time_window[max_index_in_window]
                            max_time_2 = streak_time_window[max_index_in_window]
                            lfex_peak_2_status = "detected_fallback" # 新しいステータス
                            log_warning(f"二次LFEXピークがargrelmaxで見つかりませんでした。ウィンドウ内の最大値 {max_time_2:.3f}ns を使用 (detected_fallback)", "timing_analyzer")
                        else:
                            # ウィンドウが空の場合、推定位置を使用
                            max_time_2 = expected_t2
                            offset_desc = f"オフセット{offset_value:+.3f}ns"
                            log_warning(f"二次LFEXピーク探索ウィンドウが空です。推定位置 {max_time_2:.3f}ns ({offset_desc}) を使用 (estimated)", "timing_analyzer")
                else:
                    # 探索ウィンドウが空の場合も推定位置を使用
                    max_time_2 = expected_t2
                    offset_desc = f"オフセット{offset_value:+.3f}ns"
                    log_warning(f"二次LFEXピーク探索ウィンドウが空です。推定位置 {max_time_2:.3f}ns ({offset_desc}) を使用 (estimated)", "timing_analyzer")

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
                    lfex_peak_2_status = "detected" # 2ピーク検出モードでは常に検出
                    log_debug(f"2ピーク検出成功: ピーク1={max_time_1:.3f}ns, ピーク2={max_time_2:.3f}ns", "timing_analyzer")
                elif len(peak_value) == 1:
                    max_value_1 = peak_value[0]
                    max_time_1 = streak_time[maxid[0]]
                    max_value_2 = 0
                    max_time_2 = 0
                    lfex_peak_2_status = "not_applicable" # 1個しか検出されない場合は適用外
                    log_warning("LFEXピークが1個のみ検出されました", "timing_analyzer")
                else:
                    error_msg = "No peaks detected in LFEX data"
                    log_warning(error_msg, "timing_analyzer")
                    return None, error_msg
            
            # GXIIピーク検出（カスタム波形システム対応）
            gxii_rise = np.where(gxii_norm > peak_threshold)[0][0] if len(np.where(gxii_norm > peak_threshold)[0]) > 0 else 0
            
            # 波形設定の決定（パラメータを優先）
            if waveform_config is None:
                waveform_config = {}
            
            # 設定されたパラメータを使用、フォールバックは設定ファイル
            final_waveform_type = waveform_type
            final_waveform_config = waveform_config
            
            if time_calibration_mode != "フィッティング":
                gxii_peak, fitting_params, fitting_success, waveform_name, waveform_r_squared = self._detect_gxii_peak_with_waveform(
                    streak_time, gxii_norm, final_waveform_type, final_waveform_config
                )
            actual_waveform_type = final_waveform_type
            
            if gxii_peak is None:
                # フォールバック：最大値検出
                gxii_peak = streak_time[np.argmax(gxii_norm)]
                fitting_params = [1, gxii_peak]
                fitting_success = False
                waveform_name = None
                actual_waveform_type = 'gaussian'
                log_warning(f"波形フィッティング失敗、最大値で代用: peak={gxii_peak:.3f}ns", "timing_analyzer")
            
            # 基準時間の計算
            reference_time = self.calculate_reference_time(
                reference_time_mode, gxii_peak, gxii_norm, streak_time, gxii_rise_percentage,
                waveform_name=waveform_name
            )
            
            # LFEXピーク基準の場合は、LFEXピーク時間を基準時間とする
            if reference_time_mode == "lfex_peak":
                reference_time = max_time_1
            
            log_debug(f"基準時間計算: mode={reference_time_mode}, reference_time={reference_time:.3f}ns", "timing_analyzer")
            
            # 基準時間で時間軸をシフト
            streak_time_relative = streak_time - reference_time
            
            # タイミング差の計算（基準時間モードに応じて適切に計算）
            if reference_time_mode == "gxii_rise":
                # GXII立ち上がり基準の場合：立ち上がり時間とLFEXピークの差
                time_diff = max_time_1 - reference_time
                log_info(f"タイミング計算完了（立ち上がり基準）: 立ち上がり時間={reference_time:.3f}ns, LFEX={max_time_1:.3f}ns, 差={time_diff:.3f}ns", "timing_analyzer")
            elif reference_time_mode == "lfex_peak":
                # LFEXピーク基準の場合：GXIIピークとLFEXピークの差（LFEX基準でのGXII→LFEX差）
                time_diff = gxii_peak - max_time_1  # GXIIピーク - LFEXピーク（LFEXベース）
                log_info(f"タイミング計算完了（LFEX基準）: GXII={gxii_peak:.3f}ns, LFEX={max_time_1:.3f}ns, GXII→LFEX差={time_diff:.3f}ns", "timing_analyzer")
            else:  # gxii_peak, absolute, manual
                # GXIIピーク、絶対時間、手動設定基準の場合：標準的なピーク間差（LFEXピーク - GXIIピーク）
                time_diff = (max_time_1 - reference_time) - (gxii_peak - reference_time)
                log_info(f"タイミング計算完了（ピーク間差）: GXII={gxii_peak - reference_time:.3f}ns, LFEX={max_time_1 - reference_time:.3f}ns, ピーク間差={time_diff:.3f}ns", "timing_analyzer")
            
            # ショットID生成
            shotid = FileUtils.generate_shot_id_from_filepath(filepath)
            log_debug(f"ショットID生成: {shotid}", "timing_analyzer")
            
            # 結果をまとめる
            results = {
                'filename': filename,
                'shotid': shotid,
                'shot_datetime': shot_datetime_str, # ショット日時を追加
                'gxii_peak': gxii_peak,
                'max_time_1': max_time_1,
                'max_time_2': max_time_2,
                # 基準時間基準の相対ピーク時刻
                'gxii_peak_relative': gxii_peak - reference_time,
                'lfex_peak_1_relative': max_time_1 - reference_time,
                'lfex_peak_2_relative': max_time_2 - reference_time,
                'max_value_1': max_value_1,
                'max_value_2': max_value_2,
                'time_diff': time_diff,
                'streak_time': streak_time_relative,  # 基準時間でシフトした時間軸
                'gxii_norm': gxii_norm,
                'lfex_time': lfex_time,
                'data_array': data_array,
                'total_time': total_time,
                'fitting_params': fitting_params if fitting_params else [1, gxii_peak],
                'fitting_success': fitting_success,
                'waveform_type': waveform_type,
                'actual_waveform_type': actual_waveform_type,
                'waveform_name': waveform_name,
                'waveform_r_squared': waveform_r_squared,
                'gxii_rise_time': streak_time[gxii_rise] if gxii_rise < len(streak_time) else 0,
                # 基準時間情報を追加
                'reference_time': reference_time,
                'reference_time_mode': reference_time_mode,
                'gxii_rise_percentage': gxii_rise_percentage,
                'lfex_peak_2_status': lfex_peak_2_status, # 二次ピークの検出ステータスを追加
                'fixed_offset_value': offset_value, # 固定オフセット値を追加
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
    
    def calculate_timing_difference(self, gxii_peak_pos: float, lfex_peak_pos: float, 
                                  pixel_to_time_factor: float = 1.0, reference_mode: str = None, 
                                  gxii_rise_time: float = None) -> float:
        """
        文脈対応タイミング差を計算
        
        Args:
            gxii_peak_pos: GXIIピーク位置
            lfex_peak_pos: LFEXピーク位置
            pixel_to_time_factor: ピクセルから時間への変換係数
            reference_mode: 基準時間モード ('gxii_rise', 'lfex_peak', その他)
            gxii_rise_time: GXII立ち上がり時間 (gxii_riseモード時のみ)
            
        Returns:
            タイミング差 (基準モードに応じて計算方法が変わる)
        """
        # 基準モードに応じた計算
        if reference_mode == 'gxii_rise' and gxii_rise_time is not None:
            # GXII rise mode: LFEX peak - GXII rise
            return (lfex_peak_pos - gxii_rise_time) * pixel_to_time_factor
        elif reference_mode == 'lfex_peak':
            # LFEX peak mode: GXII peak - LFEX peak (GXII→LFEX差)
            return (gxii_peak_pos - lfex_peak_pos) * pixel_to_time_factor
        else:
            # 標準モード: LFEX peak - GXII peak (ピーク間差)
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
    
    def _detect_gxii_peak_with_waveform(self, streak_time: np.ndarray, gxii_norm: np.ndarray,
                                      waveform_type: str, waveform_config: Dict[str, Any]) -> Tuple[Optional[float], Optional[List[float]], bool, Optional[str], Optional[float]]:
        """
        カスタム波形システムを使用したGXIIピーク検出
        
        Args:
            streak_time: 時間軸データ
            gxii_norm: 正規化されたGXII信号
            waveform_type: 波形タイプ ('gaussian', 'custom_pulse', 'custom_file')
            waveform_config: 波形設定
            
        Returns:
            (ピーク時間, フィッティングパラメータ, 成功フラグ, 波形名, R二乗値)のタプル
        """
        # 入力データの検証
        if streak_time is None or gxii_norm is None:
            log_error("入力データがNullです", "timing_analyzer")
            return None, None, False, None
            
        if len(streak_time) == 0 or len(gxii_norm) == 0:
            log_error("入力データが空です", "timing_analyzer")
            return None, None, False, None
            
        if len(streak_time) != len(gxii_norm):
            log_error(f"時間軸とGXII信号の長さが不一致: {len(streak_time)} vs {len(gxii_norm)}", "timing_analyzer")
            return None, None, False, None
            
        if np.any(np.isnan(streak_time)) or np.any(np.isnan(gxii_norm)):
            log_error("入力データにNaNが含まれています", "timing_analyzer")
            return None, None, False, None
            
        if np.max(gxii_norm) <= 0:
            log_error("GXII信号の最大値が0以下です", "timing_analyzer")
            return None, None, False
        
        try:
            if waveform_type == 'gaussian':
                peak, params, success, r2 = self._fit_gaussian_waveform(streak_time, gxii_norm, waveform_config.get('gaussian', {}))
                return peak, params, success, None, r2
            elif waveform_type == 'custom_pulse':
                peak, params, success, r2, name = self._fit_custom_pulse(streak_time, gxii_norm, waveform_config.get('custom_pulse', {}))
                if not success:
                    log_warning(f"カスタムパルスフィッティング失敗、ガウシアンで代用: {name}", "timing_analyzer")
                    peak, params, success, r2 = self._fit_gaussian_waveform(streak_time, gxii_norm, {})
                return peak, params, success, name, r2
            elif waveform_type == 'custom_file':
                peak, params, success, r2, name = self._fit_custom_file(streak_time, gxii_norm, waveform_config.get('custom_file', {}))
                if not success:
                    log_warning(f"カスタムファイル波形フィッティング失敗、ガウシアンで代用: {name}", "timing_analyzer")
                    peak, params, success, r2 = self._fit_gaussian_waveform(streak_time, gxii_norm, {})
                return peak, params, success, name, r2
            else:
                log_warning(f"未対応の波形タイプ: {waveform_type}、ガウシアンで代用", "timing_analyzer")
                peak, params, success, r2 = self._fit_gaussian_waveform(streak_time, gxii_norm, {})
                return peak, params, success, None, r2
                
        except Exception as e:
            log_error(f"波形フィッティングエラー: {str(e)}", "timing_analyzer", exc_info=True)
            return None, None, False, None, None
    
    def _fit_gaussian_waveform(self, streak_time: np.ndarray, gxii_norm: np.ndarray,
                             gaussian_config: Dict[str, Any]) -> Tuple[Optional[float], Optional[List[float]], bool, Optional[float]]:
        """ガウシアン波形フィッティング"""
        method = gaussian_config.get('method', 'fixed_pulse')
        
        if method == 'fixed_pulse':
            # 元の固定パルス幅ガウシアン（後方互換性）
            popt, r_squared, error = self.waveform_library.fit_waveform(
                streak_time, gxii_norm, 'gaussian_fixpulse'
            )
            if popt is not None:
                amp, mean = popt
                log_debug(f"固定パルスガウシアンフィッティング成功: peak={mean:.3f}ns", "timing_analyzer")
                return mean, [amp, mean], True, r_squared
            else:
                log_warning(f"固定パルスガウシアンフィッティング失敗: {error}", "timing_analyzer")
                return None, None, False, None
                
        elif method == 'fwhm_input':
            # FWHM入力対応ガウシアン
            fwhm = gaussian_config.get('fwhm', 1.3)  # デフォルト値
            popt, r_squared, error = self.waveform_library.fit_waveform(
                streak_time, gxii_norm, 'gaussian', 
                initial_guess=[1, np.mean(streak_time), fwhm]
            )
            if popt is not None:
                amp, mean, fitted_fwhm = popt
                log_debug(f"FWHM入力ガウシアンフィッティング成功: peak={mean:.3f}ns, FWHM={fitted_fwhm:.3f}ns, R²={r_squared:.4f}", "timing_analyzer")
                return mean, [amp, mean, fitted_fwhm], True, r_squared
            else:
                log_warning(f"FWHM入力ガウシアンフィッティング失敗: {error}", "timing_analyzer")
                return None, None, False, None
        
        else:
            log_warning(f"未対応のガウシアン手法: {method}", "timing_analyzer")
            return None, None, False, None
    
    def _fit_custom_pulse(self, streak_time: np.ndarray, gxii_norm: np.ndarray,
                         custom_config: Dict[str, Any]) -> Tuple[Optional[float], Optional[List[float]], bool, Optional[float], Optional[str]]:
        """カスタムパルス波形フィッティング"""
        if not custom_config.get('enabled', False):
            log_warning("カスタムパルスが無効化されています", "timing_analyzer")
            return None, None, False, None
        
        file_path = custom_config.get('file_path', '')
        if not file_path:
            log_error("カスタムパルスファイルパスが指定されていません", "timing_analyzer")
            return None, None, False, None
            
        if not os.path.exists(file_path):
            log_error(f"カスタムパルスファイルが見つかりません: {file_path}", "timing_analyzer")
            return None, None, False, None
        
        # ファイル拡張子の検証
        if not file_path.lower().endswith(('.csv', '.txt', '.dat')):
            log_error(f"サポートされていないファイル形式: {file_path}", "timing_analyzer")
            return None, None, False, None
        
        try:
            # カスタム波形を読み込み
            waveform_name = f"custom_{Path(file_path).stem}"
            
            if not self.waveform_library.load_custom_waveform(
                file_path, waveform_name, 
                normalize=custom_config.get('preprocessing', {}).get('normalize', True),
                interpolation_factor=custom_config.get('preprocessing', {}).get('interpolation_factor', 10)
            ):
                log_error(f"カスタム波形読み込み失敗: {file_path}", "timing_analyzer")
                return None, None, False, None, None
            
            # フィッティング実行
            popt, r_squared, error = self.waveform_library.fit_waveform(
                streak_time, gxii_norm, 'custom_pulse', 
                waveform_name=waveform_name
            )
            
            if popt is not None:
                amp, mean = popt
                log_debug(f"カスタムパルスフィッティング成功: peak={mean:.3f}ns, R²={r_squared:.4f}", "timing_analyzer")
                return mean, [amp, mean], True, r_squared, waveform_name
            else:
                log_warning(f"カスタムパルスフィッティング失敗: {error}", "timing_analyzer")
                return None, None, False, None, waveform_name
                
        except FileNotFoundError:
            log_error(f"カスタムパルスファイルが見つかりません: {file_path}", "timing_analyzer")
            return None, None, False, None, None
        except PermissionError:
            log_error(f"カスタムパルスファイルへの読み取り権限がありません: {file_path}", "timing_analyzer")
            return None, None, False, None, None
        except Exception as e:
            log_error(f"カスタムパルス処理中にエラーが発生しました: {str(e)}", "timing_analyzer", exc_info=True)
            return None, None, False, None, None
    
    def _fit_custom_file(self, streak_time: np.ndarray, gxii_norm: np.ndarray,
                        custom_file_config: Dict[str, Any]) -> Tuple[Optional[float], Optional[List[float]], bool, Optional[float], Optional[str]]:
        """カスタムファイル波形フィッティング"""
        file_path = custom_file_config.get('file_path', '')
        if not file_path:
            log_error("カスタムファイルパスが指定されていません", "timing_analyzer")
            return None, None, False, None, None
            
        if not os.path.exists(file_path):
            log_error(f"カスタムファイルが見つかりません: {file_path}", "timing_analyzer")
            return None, None, False, None, None
        
        # ファイル拡張子の検証
        if not file_path.lower().endswith(('.csv', '.txt', '.dat')):
            log_error(f"サポートされていないファイル形式: {file_path}", "timing_analyzer")
            return None, None, False
        
        try:
            # カスタム波形を読み込み
            waveform_name = f"custom_file_{Path(file_path).stem}"
            
            # デフォルトの前処理設定を使用
            time_unit = custom_file_config.get('time_unit', 'ns')
            if not self.waveform_library.load_custom_waveform(
                file_path, waveform_name, 
                normalize=True,  # デフォルトで正規化
                interpolation_factor=10,  # デフォルト補間係数
                time_unit=time_unit
            ):
                log_error(f"カスタムファイル波形読み込み失敗: {file_path}", "timing_analyzer")
                return None, None, False, None, None
            
            # フィッティング実行
            popt, r_squared, error = self.waveform_library.fit_waveform(
                streak_time, gxii_norm, 'custom_pulse', 
                waveform_name=waveform_name
            )
            
            if popt is not None:
                amp, mean = popt
                log_debug(f"カスタムファイル波形フィッティング成功: peak={mean:.3f}ns, R²={r_squared:.4f}", "timing_analyzer")
                return mean, [amp, mean], True, r_squared, waveform_name
            else:
                log_warning(f"カスタムファイル波形フィッティング失敗: {error}", "timing_analyzer")
                return None, None, False, None, waveform_name
                
        except FileNotFoundError:
            log_error(f"カスタムファイルが見つかりません: {file_path}", "timing_analyzer")
            return None, None, False, None, None
        except PermissionError:
            log_error(f"カスタムファイルへの読み取り権限がありません: {file_path}", "timing_analyzer")
            return None, None, False, None, None
        except Exception as e:
            log_error(f"カスタムファイル処理中にエラーが発生しました: {str(e)}", "timing_analyzer", exc_info=True)
            return None, None, False, None, None
