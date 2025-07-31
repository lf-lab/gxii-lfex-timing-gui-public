
"""
統合プロット管理モジュール
全プロット機能を統一管理し、重複コードを解消
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path

from ..config.settings import settings


class PlotTheme:
    """プロットテーマ設定"""
    
    # フォント設定完了フラグ（クラス変数）
    _font_setup_completed = False
    
    # カラーパレット
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'error': '#d62728',
        'gxii': '#2ca02c',
        'lfex': '#1f77b4',
        'peak': '#d62728',
        'fit': '#ff7f0e'
    }
    
    # スタイル設定
    STYLE = {
        'figure_size': (10, 6),
        'dpi': 100,
        'line_width': 2,
        'marker_size': 6,
        'font_size': 12,
        'title_size': 14,
        'label_size': 12,
        'legend_size': 10
    }
    
    @staticmethod
    def reset_font_setup():
        """
        フォント設定をリセットして再設定を強制
        """
        PlotTheme._font_setup_completed = False
        PlotTheme.setup_japanese_font()

    # 日本語フォント対応
    @staticmethod
    def setup_japanese_font():
        """
        日本語フォント設定（改良版・安定化）
        """
        import matplotlib.font_manager as fm
        from src.utils.logger_manager import log_info, log_warning
        import matplotlib.pyplot as plt
        
        # グローバル変数でフォント設定の重複実行を防ぐ
        if hasattr(PlotTheme, '_font_setup_completed') and PlotTheme._font_setup_completed:
            return
        
        try:
            # 利用可能なフォントを確認
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            # デバッグ：実際に利用可能な日本語フォントを確認
            japanese_fonts = [font for font in available_fonts if any(jp in font for jp in ['Hiragino', 'Yu', 'Gothic', 'Sans'])]
            log_info(f"利用可能な日本語フォント: {len(japanese_fonts)}個", "plot_manager")
            
            # 存在確認済みの推奨フォント（厳密チェック）
            font_candidates = [
                ('DejaVu Sans', True),      # 常に利用可能
                ('Hiragino Sans', 'Hiragino Sans' in available_fonts),
                ('YuGothic', 'YuGothic' in available_fonts),
                ('Noto Sans CJK JP', 'Noto Sans CJK JP' in available_fonts),
                ('MS Gothic', 'MS Gothic' in available_fonts),
                ('MS PGothic', 'MS PGothic' in available_fonts),
                ('sans-serif', True)        # フォールバック
            ]
            
            # 存在するフォントのみを選択
            available_preferred = []
            for font_name, is_available in font_candidates:
                if is_available:
                    available_preferred.append(font_name)
                    if font_name not in ['DejaVu Sans', 'sans-serif']:
                        log_info(f"フォント確認済み: {font_name}", "plot_manager")
            
            # matplotlib のフォント設定を更新
            plt.rcParams['font.family'] = available_preferred
            
            # 設定後の確認とログ出力
            log_info(f"フォント設定完了: {available_preferred}", "plot_manager")
            PlotTheme._font_setup_completed = True
            
            # 問題のあるフォント名の警告
            problematic_fonts = ['Yu Gothic', 'Meirio']
            for font in problematic_fonts:
                if font in available_fonts:
                    log_warning(f"問題のあるフォント名を検出: '{font}' - 使用を避けています", "plot_manager")
            
        except Exception as e:
            log_warning(f"フォント設定に失敗: {e}", "plot_manager")
            # 最小限のフォールバック設定
            plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
            log_info("フォールバックフォントを設定: ['DejaVu Sans', 'sans-serif']", "plot_manager")
            PlotTheme._font_setup_completed = True


class PlotManager:
    """統合プロット管理クラス"""
    
    def __init__(self, theme: Optional[PlotTheme] = None):
        self.theme = theme or PlotTheme()
        self._setup_matplotlib()
        # XSCショット・サマリー生成器をインポート（遅延インポートで循環参照を回避）
        self._xsc_generator = None
    
    def _setup_matplotlib(self):
        """
        matplotlibの基本設定
        """
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': self.theme.STYLE['figure_size'],
            'figure.dpi': self.theme.STYLE['dpi'],
            'lines.linewidth': self.theme.STYLE['line_width'],
            'lines.markersize': self.theme.STYLE['marker_size'],
            'font.size': self.theme.STYLE['font_size'],
            'axes.labelsize': self.theme.STYLE['label_size'],
            'axes.titlesize': self.theme.STYLE['title_size'],
            'legend.fontsize': self.theme.STYLE['legend_size'],
            'grid.alpha': 0.3
        })
        # 日本語フォント設定は外部から呼び出す
    
    def create_data_heatmap(self, data_array: np.ndarray, 
                           title: str = "Data Heatmap",
                           cmap: str = 'hot',
                           vmin: Optional[float] = None,
                           vmax: Optional[float] = None,
                           regions: Optional[Dict[str, Dict[str, int]]] = None) -> Figure:
        """
        データヒートマップを作成
        
        Args:
            data_array: データ配列
            title: プロットタイトル
            cmap: カラーマップ
            vmin, vmax: 表示範囲
            regions: 領域設定 {'gxii': {'xmin': 520, ...}, 'lfex': {...}}
            
        Returns:
            matplotlib図オブジェクト
        """
        fig, ax = plt.subplots(figsize=self.theme.STYLE['figure_size'])
        
        # 自動範囲設定
        if vmin is None:
            vmin = float(np.percentile(data_array, 5))
        if vmax is None:
            vmax = float(np.percentile(data_array, 95))
        
        # ヒートマップ描画
        im = ax.imshow(data_array, cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=(0, data_array.shape[1], data_array.shape[0], 0),
                      aspect='auto')
        
        # 領域表示
        if regions:
            self._add_region_rectangles(ax, regions)
        
        # 中心線表示
        ax.axvline(x=data_array.shape[1]/2, color='white', 
                  linestyle='--', alpha=0.5, label='Center X')
        ax.axhline(y=data_array.shape[0]/2, color='white', 
                  linestyle='--', alpha=0.5, label='Center Y')
        
        ax.set_title(title)
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        
        # カラーバー
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Intensity")
        
        if regions:
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_preview_plot(self, data_array: np.ndarray,
                           gx_params: Dict[str, int],
                           lfex_params: Dict[str, int],
                           title: str = "Data Preview") -> Figure:
        """
        統合されたプレビュープロット作成
        
        Args:
            data_array: データ配列
            gx_params: GXII領域パラメータ {'xmin': 520, 'xmax': 600, 'ymin': 4, 'ymax': 1020}
            lfex_params: LFEX領域パラメータ {'xmin': 700, 'xmax': 800}
            title: プロットタイトル
            
        Returns:
            matplotlib図オブジェクト
        """
        regions = {
            'gxii': gx_params,
            'lfex': {**lfex_params, 'ymin': gx_params['ymin'], 'ymax': gx_params['ymax']}
        }
        
        return self.create_data_heatmap(
            data_array=data_array,
            title=f"{title}\n(Red=GXII, Blue=LFEX, White dashed=Center)",
            regions=regions
        )
    
    def create_profile_plot(self, x_data: np.ndarray, y_data: np.ndarray,
                           title: str = "Profile",
                           xlabel: str = "Position", ylabel: str = "Intensity",
                           peaks: Optional[np.ndarray] = None,
                           fitted_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                           color: str = None) -> Figure:
        """
        プロファイルプロット作成
        
        Args:
            x_data, y_data: プロットデータ
            title, xlabel, ylabel: ラベル
            peaks: ピーク位置配列
            fitted_curve: フィッティング曲線 (x, y)
            color: プロット色
            
        Returns:
            matplotlib図オブジェクト
        """
        fig, ax = plt.subplots(figsize=self.theme.STYLE['figure_size'])
        
        # メインプロット
        plot_color = color or self.theme.COLORS['primary']
        ax.plot(x_data, y_data, color=plot_color, 
               linewidth=self.theme.STYLE['line_width'], label='Data')
        
        # ピーク表示
        if peaks is not None and len(peaks) > 0:
            ax.plot(x_data[peaks], y_data[peaks], 'o', 
                   color=self.theme.COLORS['peak'],
                   markersize=self.theme.STYLE['marker_size'] + 2, 
                   label='Peaks')
            
            # ピーク値表示
            for i, peak in enumerate(peaks):
                ax.annotate(f'P{i+1}', 
                           xy=(x_data[peak], y_data[peak]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=self.theme.STYLE['font_size']-1)
        
        # フィッティング曲線
        if fitted_curve is not None:
            fit_x, fit_y = fitted_curve
            ax.plot(fit_x, fit_y, '--', 
                   color=self.theme.COLORS['fit'],
                   linewidth=self.theme.STYLE['line_width'], 
                   label='Fitting')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(self, x_data: np.ndarray,
                              gxii_data: np.ndarray, lfex_data: np.ndarray,
                              title: str = "GXII vs LFEX Comparison",
                              gxii_peaks: Optional[np.ndarray] = None,
                              lfex_peaks: Optional[np.ndarray] = None) -> Figure:
        """
        GXII vs LFEX 比較プロット
        
        Args:
            x_data: X軸データ
            gxii_data, lfex_data: プロファイルデータ
            title: プロットタイトル
            gxii_peaks, lfex_peaks: ピーク位置
            
        Returns:
            matplotlib図オブジェクト
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # GXIIプロット
        ax1.plot(x_data, gxii_data, color=self.theme.COLORS['gxii'], 
                linewidth=self.theme.STYLE['line_width'], label='GXII')
        if gxii_peaks is not None and len(gxii_peaks) > 0:
            ax1.plot(x_data[gxii_peaks], gxii_data[gxii_peaks], 'o',
                    color=self.theme.COLORS['peak'],
                    markersize=self.theme.STYLE['marker_size'] + 2,
                    label='GXII Peaks')
        
        ax1.set_title('GXII Profile')
        ax1.set_ylabel('Intensity')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # LFEXプロット
        ax2.plot(x_data, lfex_data, color=self.theme.COLORS['lfex'],
                linewidth=self.theme.STYLE['line_width'], label='LFEX')
        if lfex_peaks is not None and len(lfex_peaks) > 0:
            ax2.plot(x_data[lfex_peaks], lfex_data[lfex_peaks], 'o',
                    color=self.theme.COLORS['peak'],
                    markersize=self.theme.STYLE['marker_size'] + 2,
                    label='LFEX Peaks')
        
        ax2.set_title('LFEX Profile')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Intensity')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(title, fontsize=self.theme.STYLE['title_size'])
        plt.tight_layout()
        return fig
    
    def create_timing_analysis_plots(self, results: Dict) -> List[Figure]:
        """
        タイミング解析結果の統合プロット作成
        
        Args:
            results: analyze_timing()の結果辞書
            
        Returns:
            matplotlib図オブジェクトのリスト
        """
        figures = []
        
        # 1. 生データヒートマップ
        fig1 = self._create_raw_data_heatmap(results)
        figures.append(fig1)
        
        # 2. プロファイル比較プロット
        fig2 = self._create_profile_comparison(results)
        figures.append(fig2)
        
        return figures
    
    def _create_raw_data_heatmap(self, results: Dict) -> Figure:
        """生データヒートマップ作成（内部メソッド）"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_array = results['data_array']
        total_time = results['total_time']

        # 画像高さから実際の時間範囲を計算
        full_time = results['time_per_pixel'] * data_array.shape[0]

        # カラースケールの最適化
        vmin = float(np.percentile(data_array, 5))
        vmax = float(np.percentile(data_array, 95))

        # ヒートマップ描画（元のコードの座標系を維持）
        heatmap = ax.imshow(
            data_array,
            cmap='hot',
            vmin=vmin,
            vmax=vmax,
            extent=[
                -(100 / 78) * data_array.shape[1] / 2,
                (100 / 78) * data_array.shape[1] / 2,
                full_time,
                0,
            ],
            aspect=(100 / 78) * (data_array.shape[0] / full_time),
        )
        
        # LFEXピーク表示（破線で表示）
        ax.axhline(results['max_time_1'], color='red', 
                  linestyle='--', linewidth=1, label='1st LFEX Peak')
        
        # 2次ピークの表示制御：not_applicableの場合は表示しない
        lfex_peak_2_status = results.get('lfex_peak_2_status', 'unknown')
        if results['max_value_2'] > 0 and lfex_peak_2_status != 'not_applicable':
            ax.axhline(results['max_time_2'], color='blue', 
                      linestyle=':', linewidth=1, label='2nd LFEX Peak')
        
        # GXIIピーク表示を削除
        # if 'gxii_peak' in results:
        #     ax.axhline(results['gxii_peak'], color='green', 
        #               linestyle='-', linewidth=2, label='GXII Peak')
        
        ax.set_title(f"{results['shotid']} XSC Raw Data")
        ax.set_xlabel("On-target length (µm)")
        ax.set_ylabel("Time on streak (ns)")
        
        plt.colorbar(heatmap, ax=ax)
        plt.legend()
        plt.tight_layout()
        
        return fig
    
    def _create_profile_comparison(self, results: Dict) -> Figure:
        """プロファイル比較プロット作成（内部メソッド）"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        
        # LFEX時間プロファイル
        ax1.plot(results['streak_time'], results['lfex_time'], 'b-', 
                linewidth=2, label='LFEX Profile')
        ax1.plot(results['max_time_1'], results['max_value_1'], 'ro', 
                markersize=8, label='1st Peak')
        
        # 2次ピークの表示制御：not_applicableの場合は表示しない
        lfex_peak_2_status = results.get('lfex_peak_2_status', 'unknown')
        if results['max_value_2'] > 0 and lfex_peak_2_status != 'not_applicable':
            ax1.plot(results['max_time_2'], results['max_value_2'], 'bo', 
                    markersize=8, label=f'2nd Peak ({lfex_peak_2_status})')
        
        ax1.set_title(f"{results['shotid']} LFEX Profile")
        ax1.set_xlabel("Time (ns)")
        ax1.set_ylabel("Intensity")
        ax1.legend()
        ax1.grid(True)
        
        # GXII正規化プロファイル
        ax2.plot(results['streak_time'], results['gxii_norm'], 'g-', 
                linewidth=2, label='GXII Normalized')
        
        # ガウシアンフィット表示
        if len(results['fitting_params']) >= 2:
            from ..core.waveform_library import WaveformLibrary
            waveform_library = WaveformLibrary()
            fitted_curve = waveform_library.waveform_functions['gaussian_fixpulse'](
                results['streak_time'], *results['fitting_params'])
            ax2.plot(results['streak_time'], fitted_curve, 'r--', 
                    linewidth=2, label='Gaussian Fit')
            # GXIIピーク縦線を削除
        
        ax2.set_title(f"{results['shotid']} GXII Normalized Profile")
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("Normalized Intensity")
        ax2.legend()
        ax2.grid(True)
        
        # タイミング差表示
        self._create_timing_difference_bar_chart(ax3, results)
        
        plt.tight_layout()
        return fig
    
    def _create_timing_difference_bar_chart(self, ax: plt.Axes, results: Dict):
        """タイミング差のバーチャートを作成（内部メソッド）"""
        timing_values = [results['gxii_peak'], results['max_time_1'], results['time_diff']]
        timing_labels = ['GXII Peak', '1st LFEX Peak', 'Time Difference']
        colors = ['green', 'red', 'purple']
        
        bars = ax.bar(timing_labels, timing_values, color=colors)
        ax.set_title(f"{results['shotid']} Timing Analysis Results")
        ax.set_ylabel("Time (ns)")
        
        # 値をバーの上に表示
        for bar, value in zip(bars, timing_values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        ax.grid(True)

    def get_xsc_shot_summary_generator(self):
        """
        XSCショット・サマリー生成器を取得（遅延初期化）
        """
        if self._xsc_generator is None:
            from .xsc_shot_summary import XSCShotSummaryGenerator
            self._xsc_generator = XSCShotSummaryGenerator(plot_manager=self)
        return self._xsc_generator
    
    def create_xsc_shot_summary_pdf(self, shot_data: Dict[str, Any], 
                                   output_path: Union[str, Path],
                                   shot_id: str = None) -> bool:
        """
        XSCショット・サマリーPDFを生成
        
        Args:
            shot_data: ショットデータ辞書
            output_path: 出力PDFファイルパス
            shot_id: ショットID（オプション）
            
        Returns:
            bool: 生成成功フラグ
        """
        generator = self.get_xsc_shot_summary_generator()
        return generator.generate_shot_summary_pdf(shot_data, output_path, shot_id)
    
    def create_xsc_shot_summary_pdf_bytes(self, shot_data: Dict[str, Any], 
                                         shot_id: str = None,
                                         dpi: int = 300,
                                         colormap: str = 'viridis',
                                         annotation_level: str = '詳細',
                                         include_metadata_page: bool = True) -> Optional[bytes]:
        """
        XSCショット・サマリーPDFをバイト形式で生成（Streamlit用）
        
        Args:
            shot_data: ショットデータ辞書
            shot_id: ショットID（オプション）
            dpi: PDF解像度
            colormap: カラーマップ
            annotation_level: 注釈レベル
            include_metadata_page: メタデータページを含めるか
            
        Returns:
            Optional[bytes]: PDFバイトデータ（失敗時はNone）
        """
        generator = self.get_xsc_shot_summary_generator()
        return generator.generate_shot_summary_pdf_bytes(
            shot_data=shot_data,
            shot_id=shot_id,
            dpi=dpi,
            colormap=colormap,
            annotation_level=annotation_level,
            include_metadata_page=include_metadata_page
        )
    
    def create_xsc_result_display_plots(self, results: Dict,
                                        waveform_library=None) -> List[Figure]:
        """
        XSC結果表示用の3つのプロット作成（PDF生成と同じ構成）
        
        Args:
            results: analyze_timing()の結果辞書
            waveform_library: 波形ライブラリ（カスタム波形保持用）
            
        Returns:
            matplotlib図オブジェクトのリスト [ヒートマップ, 2パネル, スペースラインアウト]
        """
        figures = []
        
        # 1. 生データヒートマップ（オーバーレイ付き）
        fig1 = self._create_raw_data_heatmap(results)
        figures.append(fig1)
        
        # 2. 2パネル垂直レイアウト（バーチャート除去版）
        fig2 = self._create_two_panel_comparison(results, waveform_library)
        figures.append(fig2)
        
        # 3. スペースラインアウト（PDFパネル4相当）
        fig3 = self._create_space_lineout_plot(results)
        figures.append(fig3)
        
        return figures
    
    def _create_two_panel_comparison(self, results: Dict, waveform_library=None, hide_lfex_2nd_peak_detected: bool = False) -> Figure:
        """2パネル比較プロット作成（設定に基づく基準時間軸）

        Args:
            results: analyze_timing() の結果辞書
            waveform_library: 波形ライブラリ（Noneの場合は新規作成）
            hide_lfex_2nd_peak_detected: LFEX 2nd ピークの検出値を非表示にするか
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        from ..config.settings import settings
        
        # 基準時間設定を取得
        from ..config.settings import get_config
        reference_mode = get_config('reference_time.mode', 'gxii_peak')
        reference_labels = get_config('reference_time.labels', {})
        
        # 基準時間を決定
        if reference_mode == 'gxii_peak':
            reference_time = results.get('gxii_peak', 0)
            xlabel = reference_labels.get('gxii_peak', 'Relative Time from GXII Peak (ns)')
            reference_label = f'GXII Peak: 0.000 ns (reference)'
        elif reference_mode == 'gxii_rise':
            reference_time = results.get('gxii_rise_time', 0)
            xlabel = reference_labels.get('gxii_rise', 'Relative Time from GXII Rise (ns)')
            reference_label = f'GXII Rise: 0.000 ns (reference)'
        elif reference_mode == 'lfex_peak':
            reference_time = results.get('max_time_1', 0)
            xlabel = reference_labels.get('lfex_peak', 'Relative Time from LFEX Peak (ns)')
            reference_label = f'LFEX Peak: 0.000 ns (reference)'
        elif reference_mode == 'absolute':
            reference_time = get_config('reference_time.absolute_value', 0.0)
            xlabel = reference_labels.get('absolute', 'Absolute Time (ns)')
            reference_label = f'Reference: {reference_time:.3f} ns'
        elif reference_mode == 'manual':
            reference_time = get_config('reference_time.manual_value', 0.0)
            xlabel = reference_labels.get('manual', 'Relative Time from Reference (ns)')
            reference_label = f'Manual Reference: 0.000 ns'
        elif reference_mode == 'custom_t0':
            reference_time = results.get('reference_time', 0)
            xlabel = reference_labels.get('custom_t0', 'Custom Waveform t0 (ns)')
            reference_label = 'Custom Waveform t0: 0.000 ns'
        else:
            # デフォルトはGXII基準
            reference_time = results.get('gxii_peak', 0)
            xlabel = 'Relative Time from GXII Peak (ns)'
            reference_label = f'GXII Peak: 0.000 ns (reference)'
        
        # 相対時間軸を取得
        relative_time = results['streak_time']
        
        # LFEX時間プロファイル
        ax1.plot(relative_time, results['lfex_time'], 'b-', 
                linewidth=2, label='LFEX Profile')
        
        # ピークを破線で表示（相対時間）
        ax1.axvline(results['lfex_peak_1_relative'], color='orange', 
                   linestyle='--', linewidth=2, label='1st Peak')
        
        # 2次ピークの表示制御：not_applicableの場合は表示しない
        lfex_peak_2_status = results.get('lfex_peak_2_status', 'unknown')
        if lfex_peak_2_status != 'not_applicable':
            # LFEX 2nd ピークのプロット (推定値と検出値)
            # マイナスオフセットも考慮した推定位置計算
            offset_value = results.get('fixed_offset_value', 0.24)
            estimated_lfex_peak_2_relative = results['lfex_peak_1_relative'] + offset_value
            
            # オフセットの方向を示すラベル
            offset_direction = "後" if offset_value >= 0 else "前"
            estimated_label = f'2nd Peak (Estimated, {offset_value:+.3f}ns {offset_direction})'
            
            ax1.axvline(estimated_lfex_peak_2_relative, color='gray', 
                       linestyle=':', linewidth=1, label=estimated_label)
            
            if results['max_value_2'] > 0 and not hide_lfex_2nd_peak_detected:
                ax1.axvline(results['lfex_peak_2_relative'], color='purple', 
                           linestyle='--', linewidth=2, label=f'2nd Peak (Detected)')
        
        ax1.set_title(f"{results['shotid']} LFEX Profile")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Intensity")
        ax1.legend()
        ax1.grid(True)
        
        # GXII正規化プロファイル
        ax2.plot(relative_time, results['gxii_norm'], 'g-', 
                linewidth=2, label='GXII Normalized')
        
        # 波形フィット表示 (実際に使用された波形タイプに基づく)
        if len(results['fitting_params']) >= 2:
            from ..core.waveform_library import WaveformLibrary
            
            waveform_type = results.get('actual_waveform_type', results.get('waveform_type', 'gaussian'))
            waveform_name = results.get('waveform_name')
            fitting_params = results['fitting_params']

            # 波形ライブラリを用意（引数がなければ新規作成）
            if waveform_library is None:
                from ..core.waveform_library import WaveformLibrary
                waveform_library = WaveformLibrary()
            
            # 波形タイプに応じた適切な関数を選択
            if waveform_type == 'gaussian':
                # ガウシアンの場合、パラメータ数に応じて適切な関数を選択
                if len(fitting_params) == 2:
                    # 固定パルス幅ガウシアン (amp, mean)
                    waveform_func = waveform_library.waveform_functions['gaussian_fixpulse']
                    fitted_curve = waveform_func(relative_time, *fitting_params)
                    fit_label = 'Gaussian Fixed-Pulse Fit'
                elif len(fitting_params) == 3:
                    # 標準ガウシアン (amp, mean, sigma) または FWHM ガウシアン (amp, mean, fwhm)
                    waveform_func = waveform_library.waveform_functions['gaussian']
                    fitted_curve = waveform_func(relative_time, *fitting_params)
                    fit_label = 'Gaussian Fit'
                else:
                    # フォールバック: 固定パルス幅ガウシアン
                    waveform_func = waveform_library.waveform_functions['gaussian_fixpulse']
                    fitted_curve = waveform_func(relative_time, *fitting_params[:2])
                    fit_label = 'Gaussian Fit (fallback)'
            elif waveform_type == 'custom_pulse':
                # カスタムパルスの場合、カスタム波形が存在するかチェック
                try:
                    if waveform_name is None:
                        custom_waveforms = list(waveform_library.custom_waveforms.keys())
                        waveform_name = custom_waveforms[0] if custom_waveforms else None
                    if waveform_name:
                        if reference_mode == 'custom_t0':
                            # custom_t0基準の場合、カスタム波形のt=0がプロットのt=0になるように調整
                            original_wf_data = waveform_library.custom_waveforms.get(waveform_name)
                            if original_wf_data:
                                target_peak_for_plot = original_wf_data['peak_time']
                                waveform_func = lambda x, amp, mean_dummy: waveform_library._custom_pulse_function(x, amp, target_peak_for_plot, waveform_name)
                                fitted_curve = waveform_func(relative_time, fitting_params[0], 0) # dummy mean
                            else:
                                # Fallback if original_wf_data not found
                                waveform_func = lambda x, amp, mean: waveform_library._custom_pulse_function(x, amp, mean, waveform_name)
                                fitted_curve = waveform_func(relative_time, *fitting_params[:2])
                        else:
                            waveform_func = lambda x, amp, mean: waveform_library._custom_pulse_function(x, amp, mean, waveform_name)
                            fitted_curve = waveform_func(relative_time, *fitting_params[:2])
                        fit_label = f'Custom Pulse Fit ({waveform_name})'
                    else:
                        # カスタム波形が見つからない場合はガウシアンにフォールバック
                        waveform_func = waveform_library.waveform_functions['gaussian_fixpulse']
                        fitted_curve = waveform_func(relative_time, *fitting_params[:2])
                        fit_label = 'Custom Pulse Fit (gaussian fallback)'
                except Exception:
                    # エラーの場合はガウシアンにフォールバック
                    waveform_func = waveform_library.waveform_functions['gaussian_fixpulse']
                    fitted_curve = waveform_func(relative_time, *fitting_params[:2])
                    fit_label = 'Custom Pulse Fit (gaussian fallback)'
            elif waveform_type == 'custom_file':
                # カスタムファイルの場合、カスタム波形が存在するかチェック
                try:
                    if waveform_name is None:
                        custom_waveforms = list(waveform_library.custom_waveforms.keys())
                        custom_file_waveforms = [name for name in custom_waveforms if name.startswith('custom_file_')]
                        waveform_name = custom_file_waveforms[0] if custom_file_waveforms else None
                    if waveform_name:
                        if reference_mode == 'custom_t0':
                            # custom_t0基準の場合、カスタム波形のt=0がプロットのt=0になるように調整
                            original_wf_data = waveform_library.custom_waveforms.get(waveform_name)
                            if original_wf_data:
                                target_peak_for_plot = original_wf_data['peak_time']
                                waveform_func = lambda x, amp, mean_dummy: waveform_library._custom_pulse_function(x, amp, target_peak_for_plot, waveform_name)
                                fitted_curve = waveform_func(relative_time, fitting_params[0], 0) # dummy mean
                            else:
                                # Fallback if original_wf_data not found
                                waveform_func = lambda x, amp, mean: waveform_library._custom_pulse_function(x, amp, mean, waveform_name)
                                fitted_curve = waveform_func(relative_time, *fitting_params[:2])
                        else:
                            waveform_func = lambda x, amp, mean: waveform_library._custom_pulse_function(x, amp, mean, waveform_name)
                            fitted_curve = waveform_func(relative_time, *fitting_params[:2])
                        fit_label = f'Custom File Fit ({waveform_name.replace("custom_file_", "")})'
                    else:
                        # カスタム波形が見つからない場合はガウシアンにフォールバック
                        waveform_func = waveform_library.waveform_functions['gaussian_fixpulse']
                        fitted_curve = waveform_func(relative_time, *fitting_params[:2])
                        fit_label = 'Custom File Fit (gaussian fallback)'
                except Exception:
                    # エラーの場合はガウシアンにフォールバック
                    waveform_func = waveform_library.waveform_functions['gaussian_fixpulse']
                    fitted_curve = waveform_func(relative_time, *fitting_params[:2])
                    fit_label = 'Custom File Fit (gaussian fallback)'
            else:
                # 不明な波形タイプの場合はガウシアンにフォールバック
                waveform_func = waveform_library.waveform_functions['gaussian_fixpulse']
                fitted_curve = waveform_func(relative_time, *fitting_params[:2])
                fit_label = f'{waveform_type} Fit (gaussian fallback)'
            
            ax2.plot(relative_time, fitted_curve, 'r--', 
                    linewidth=2, label=fit_label)
            
            # LFEXピークをGXIIプロットにも表示
            ax2.axvline(results['lfex_peak_1_relative'], color='orange', 
                       linestyle='--', linewidth=1, label='LFEX 1st Peak')
            
            # 2次ピークの表示制御：not_applicableの場合は表示しない
            lfex_peak_2_status = results.get('lfex_peak_2_status', 'unknown')
            if lfex_peak_2_status != 'not_applicable':
                # LFEX 2nd ピークのプロット (推定値と検出値)
                # マイナスオフセットも考慮した推定位置計算
                offset_value = results.get('fixed_offset_value', 0.24)
                estimated_lfex_peak_2_relative = results['lfex_peak_1_relative'] + offset_value
                
                # オフセットの方向を示すラベル
                offset_direction = "後" if offset_value >= 0 else "前"
                estimated_label = f'2nd Peak (Estimated, {offset_value:+.3f}ns {offset_direction})'
                
                ax2.axvline(estimated_lfex_peak_2_relative, color='gray', 
                           linestyle=':', linewidth=1, label=estimated_label)
                
                if results['max_value_2'] > 0 and not hide_lfex_2nd_peak_detected:
                    ax2.axvline(results['lfex_peak_2_relative'], color='purple', 
                               linestyle='--', linewidth=2, label=f'2nd Peak (Detected)')
            
            # 基準時間ラインを表示
            if reference_mode == 'gxii_peak':
                ax2.axvline(0, color='red', 
                           linestyle='-', linewidth=2, 
                           label=reference_label)
            elif reference_mode == 'gxii_rise':
                ax2.axvline(0, color='orange', 
                           linestyle='-', linewidth=2, 
                           label=reference_label)
                # GXIIピーク縦線を削除（GXII riseモード時）
            elif reference_mode == 'custom_t0':
                ax2.axvline(0, color='purple', linestyle='-', linewidth=2, label=reference_label)
                # custom_t0モード時はGXIIピーク縦線を削除
                # gxii_peak_relative = results.get('gxii_peak_relative', 0)
                # ax2.axvline(gxii_peak_relative, color='green', linestyle='--', linewidth=2, label=f'GXII Peak: {gxii_peak_relative:.3f} ns')
            else:
                # GXII以外の基準の場合、GXIIピーク縦線を削除
                # ax2.axvline(results['gxii_peak_relative'], color='green', 
                #            linestyle='--', linewidth=2, 
                #            label=f'GXII Peak: {results["gxii_peak_relative"]:.3f} ns')
                ax2.axvline(0, color='red', 
                           linestyle='-', linewidth=2, 
                           label=reference_label)
        
        # t=0 の線を追加
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.8, label='t=0')
        
        # x軸の範囲を調整してt=0を含むようにする
        min_x = min(relative_time.min(), 0)
        max_x = relative_time.max()
        ax1.set_xlim(min_x, max_x) # LFEXプロットのX軸範囲も設定
        ax2.set_xlim(min_x, max_x)
        
        ax2.set_title(f"{results['shotid']} GXII Normalized Profile")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Normalized Intensity")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def _create_space_lineout_plot(self, results: Dict) -> Figure:
        """スペースラインアウトプロット作成（PDFパネル4相当）"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 生データから空間プロファイルを生成
        data_array = results['data_array']
        
        if data_array is not None:
            # 全画像範囲での空間積分プロファイル（Y方向積分）
            full_space_profile = np.sum(data_array, axis=0)  # Y方向積分
            
            # ピクセル座標を空間座標に変換（Raw dataと同じ座標系）
            pixel_count = len(full_space_profile)
            spatial_coords = np.linspace(-(100/78)*1344/2, (100/78)*1344/2, pixel_count)
            
            # GXIIピーク位置を取得してX軸をセンタリング
            gxii_peak_time = results.get('gxii_peak', 0)
            
            # GXIIピーク位置に対応する空間座標を中心に設定
            # GXIIピーク時刻から対応する空間位置を推定
            gx_xmin = results.get('gx_xmin', 520)
            gx_xmax = results.get('gx_xmax', 600)
            gxii_center_pixel = (gx_xmin + gx_xmax) / 2
            gxii_spatial_center = spatial_coords[int(gxii_center_pixel)] if int(gxii_center_pixel) < len(spatial_coords) else 0
            
            # X軸をGXIIピーク中心にシフト
            centered_coords = spatial_coords - gxii_spatial_center
            
            # プロット
            ax.plot(centered_coords, full_space_profile, 'k-', linewidth=2, label='Full Image Space Profile')
            
            # GXII領域とLFEX領域をハイライト表示
            gx_xmin = results.get('gx_xmin', 520)
            gx_xmax = results.get('gx_xmax', 600)
            lfex_xmin = results.get('lfex_xmin', 700)
            lfex_xmax = results.get('lfex_xmax', 800)
            
            # 領域のプロファイルも表示
            if gx_xmin < len(full_space_profile) and gx_xmax < len(full_space_profile):
                gxii_mask = np.zeros_like(full_space_profile)
                gxii_mask[gx_xmin:gx_xmax] = full_space_profile[gx_xmin:gx_xmax]
                ax.fill_between(centered_coords, 0, gxii_mask, alpha=0.3, color='green', label='GXII Region')
            
            if lfex_xmin < len(full_space_profile) and lfex_xmax < len(full_space_profile):
                lfex_mask = np.zeros_like(full_space_profile)
                lfex_mask[lfex_xmin:lfex_xmax] = full_space_profile[lfex_xmin:lfex_xmax]
                ax.fill_between(centered_coords, 0, lfex_mask, alpha=0.3, color='blue', label='LFEX Region')
            
            # 全Y軸範囲を表示
            y_min = np.min(full_space_profile)
            y_max = np.max(full_space_profile)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
            
            ax.set_title(f"{results['shotid']} Space Lineout (Centered at GXII Peak)")
            ax.set_xlabel("On-target length (µm) - Centered at GXII")
            ax.set_ylabel("Integrated Intensity")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # ゼロライン追加
            ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='GXII Center')
            
        else:
            # データがない場合のフォールバック
            ax.text(0.5, 0.5, 'No Space Profile Data Available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_title(f"{results.get('shotid', 'Unknown')} Space Lineout")
            ax.set_xlabel("On-target length (µm)")
            ax.set_ylabel("Integrated Intensity")
        
        plt.tight_layout()
        return fig

    def _add_region_rectangles(self, ax: plt.Axes, regions: Dict[str, Dict[str, int]]):
        """領域矩形を追加（内部メソッド）"""
        region_colors = {'gxii': 'red', 'lfex': 'blue'}
        region_names = {'gxii': 'GXII', 'lfex': 'LFEX'}
        
        for region_key, params in regions.items():
            color = region_colors.get(region_key, 'gray')
            name = region_names.get(region_key, region_key.upper())
            
            width = params['xmax'] - params['xmin']
            height = params['ymax'] - params['ymin']
            
            rect = Rectangle((params['xmin'], params['ymin']), width, height,
                           linewidth=3, edgecolor=color, facecolor='none',
                           label=f'{name} Region ({width}×{height})')
            ax.add_patch(rect)
    
    def save_figure(self, fig: Figure, filepath: Union[str, Path], 
                   dpi: int = 300, format: str = 'png') -> bool:
        """
        図を保存
        
        Args:
            fig: matplotlib図オブジェクト
            filepath: 保存パス
            dpi: 解像度
            format: ファイル形式
            
        Returns:
            保存成功フラグ
        """
        try:
            fig.savefig(filepath, dpi=dpi, format=format, 
                       bbox_inches='tight', facecolor='white')
            return True
        except Exception as e:
            from .logger_manager import log_error
            log_error(f"図の保存に失敗: {e}", "plot_manager", exc_info=True)
            return False

    def generate_report_plots(self, results: Dict, waveform_library=None, hide_lfex_2nd_peak_detected: bool = False) -> List[bytes]:
        """
        レポート用のプロットを生成し、バイトデータとして返す
        
        Args:
            results: analyze_timing()の結果辞書
            waveform_library: 波形ライブラリ（カスタム波形保持用）
            hide_lfex_2nd_peak_detected: LFEX 2nd ピークの検出値を非表示にするか
            
        Returns:
            生成された画像ファイルのバイトデータのリスト
        """
        from src.utils.logger_manager import log_info, log_error
        import io

        saved_plot_bytes = []
        try:
            # 1. 生データヒートマップ
            fig1 = self._create_raw_data_heatmap(results)
            img_byte_arr1 = io.BytesIO()
            fig1.savefig(img_byte_arr1, format='PNG', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig1)
            saved_plot_bytes.append(img_byte_arr1.getvalue())
            log_info("生データヒートマップのバイトデータ生成完了", "plot_manager")

            # 2. 2パネル垂直レイアウト
            fig2 = self._create_two_panel_comparison(results, waveform_library, hide_lfex_2nd_peak_detected)
            img_byte_arr2 = io.BytesIO()
            fig2.savefig(img_byte_arr2, format='PNG', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig2)
            saved_plot_bytes.append(img_byte_arr2.getvalue())
            log_info("2パネル比較プロットのバイトデータ生成完了", "plot_manager")

            return saved_plot_bytes
        except Exception as e:
            log_error(f"レポートプロット生成エラー: {str(e)}", "plot_manager", exc_info=True)
            return []
