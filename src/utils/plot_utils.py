"""
プロットユーティリティモジュール
グラフ作成関連の共通機能を提供

⚠️ DEPRECATED: このモジュールは非推奨です
新しいコードでは src.utils.plot_manager.PlotManager を使用してください
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, Tuple, List, Dict, Any
import streamlit as st

from ..config.settings import settings

# 非推奨警告を発行
warnings.warn(
    "PlotUtils is deprecated. Use PlotManager instead.",
    DeprecationWarning,
    stacklevel=2
)


class PlotUtils:
    """プロット作成ユーティリティクラス"""
    
    @staticmethod
    def setup_matplotlib_style():
        """matplotlibのスタイルを設定"""
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = settings.DEFAULT_FIGURE_SIZE
        plt.rcParams['figure.dpi'] = settings.DEFAULT_DPI
        plt.rcParams['lines.linewidth'] = settings.DEFAULT_LINE_WIDTH
        plt.rcParams['lines.markersize'] = settings.DEFAULT_MARKER_SIZE
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
    
    @staticmethod
    def create_data_heatmap(data_array: np.ndarray, title: str = "Data Heatmap", 
                           cmap: str = 'hot', vmin: Optional[float] = None, 
                           vmax: Optional[float] = None) -> plt.Figure:
        """
        データのヒートマップを作成
        
        Args:
            data_array: データ配列
            title: タイトル
            cmap: カラーマップ
            vmin, vmax: カラーレンジ
            
        Returns:
            matplotlib図オブジェクト
        """
        PlotUtils.setup_matplotlib_style()
        fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGURE_SIZE)
        
        # 自動スケーリング
        if vmin is None:
            vmin = float(np.percentile(data_array, 5))
        if vmax is None:
            vmax = float(np.percentile(data_array, 95))
        
        # ヒートマップ作成
        im = ax.imshow(data_array, cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=[0, data_array.shape[1], data_array.shape[0], 0],
                      aspect='auto')
        
        # カラーバー追加
        cbar = plt.colorbar(im, ax=ax, label='Intensity')
        
        ax.set_title(title)
        ax.set_xlabel('X Pixel')
        ax.set_ylabel('Y Pixel')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_profile_plot(x_data: np.ndarray, y_data: np.ndarray, 
                           title: str = "Profile", xlabel: str = "Position", 
                           ylabel: str = "Intensity", peaks: Optional[np.ndarray] = None,
                           fitted_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> plt.Figure:
        """
        プロファイルプロットを作成
        
        Args:
            x_data: X軸データ
            y_data: Y軸データ
            title: タイトル
            xlabel: X軸ラベル
            ylabel: Y軸ラベル
            peaks: ピーク位置
            fitted_curve: フィッティング曲線 (x, y)
            
        Returns:
            matplotlib図オブジェクト
        """
        PlotUtils.setup_matplotlib_style()
        fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGURE_SIZE)
        
        # メインプロット
        ax.plot(x_data, y_data, color=settings.COLORS['primary'], 
               linewidth=settings.DEFAULT_LINE_WIDTH, label='Data')
        
        # ピークをハイライト
        if peaks is not None and len(peaks) > 0:
            ax.plot(x_data[peaks], y_data[peaks], 'ro', 
                   markersize=settings.DEFAULT_MARKER_SIZE + 2, label='Peaks')
            
            # ピーク位置にテキスト追加
            for i, peak in enumerate(peaks):
                ax.annotate(f'Peak {i+1}\n({x_data[peak]:.1f}, {y_data[peak]:.1f})',
                           xy=(x_data[peak], y_data[peak]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # フィッティング曲線
        if fitted_curve is not None:
            fit_x, fit_y = fitted_curve
            ax.plot(fit_x, fit_y, '--', color=settings.COLORS['secondary'], 
                   linewidth=settings.DEFAULT_LINE_WIDTH, label='Fitted Curve')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_comparison_plot(x_data: np.ndarray, gxii_data: np.ndarray, 
                              lfex_data: np.ndarray, title: str = "GXII vs LFEX Comparison",
                              gxii_peaks: Optional[np.ndarray] = None,
                              lfex_peaks: Optional[np.ndarray] = None) -> plt.Figure:
        """
        GXIIとLFEXの比較プロットを作成
        
        Args:
            x_data: X軸データ
            gxii_data: GXIIデータ
            lfex_data: LFEXデータ
            title: タイトル
            gxii_peaks: GXIIピーク位置
            lfex_peaks: LFEXピーク位置
            
        Returns:
            matplotlib図オブジェクト
        """
        PlotUtils.setup_matplotlib_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(settings.DEFAULT_FIGURE_SIZE[0], 
                                                     settings.DEFAULT_FIGURE_SIZE[1] * 1.5))
        
        # GXIIプロット
        ax1.plot(x_data, gxii_data, color='red', linewidth=settings.DEFAULT_LINE_WIDTH, label='GXII')
        if gxii_peaks is not None and len(gxii_peaks) > 0:
            ax1.plot(x_data[gxii_peaks], gxii_data[gxii_peaks], 'ro', 
                    markersize=settings.DEFAULT_MARKER_SIZE + 2, label='GXII Peaks')
        
        ax1.set_title('GXII Profile')
        ax1.set_ylabel('Intensity')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # LFEXプロット
        ax2.plot(x_data, lfex_data, color='blue', linewidth=settings.DEFAULT_LINE_WIDTH, label='LFEX')
        if lfex_peaks is not None and len(lfex_peaks) > 0:
            ax2.plot(x_data[lfex_peaks], lfex_data[lfex_peaks], 'bo', 
                    markersize=settings.DEFAULT_MARKER_SIZE + 2, label='LFEX Peaks')
        
        ax2.set_title('LFEX Profile')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Intensity')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_overlay_plot(x_data: np.ndarray, gxii_data: np.ndarray, 
                           lfex_data: np.ndarray, title: str = "GXII vs LFEX Overlay",
                           normalize: bool = True) -> plt.Figure:
        """
        GXIIとLFEXのオーバーレイプロットを作成
        
        Args:
            x_data: X軸データ
            gxii_data: GXIIデータ
            lfex_data: LFEXデータ
            title: タイトル
            normalize: 正規化するかどうか
            
        Returns:
            matplotlib図オブジェクト
        """
        PlotUtils.setup_matplotlib_style()
        fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGURE_SIZE)
        
        # データの正規化
        if normalize:
            gxii_norm = gxii_data / np.max(gxii_data) if np.max(gxii_data) > 0 else gxii_data
            lfex_norm = lfex_data / np.max(lfex_data) if np.max(lfex_data) > 0 else lfex_data
        else:
            gxii_norm = gxii_data
            lfex_norm = lfex_data
        
        # プロット
        ax.plot(x_data, gxii_norm, color='red', linewidth=settings.DEFAULT_LINE_WIDTH, 
               label='GXII' + (' (normalized)' if normalize else ''), alpha=0.8)
        ax.plot(x_data, lfex_norm, color='blue', linewidth=settings.DEFAULT_LINE_WIDTH, 
               label='LFEX' + (' (normalized)' if normalize else ''), alpha=0.8)
        
        ax.set_title(title)
        ax.set_xlabel('Position')
        ax.set_ylabel('Intensity' + (' (normalized)' if normalize else ''))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_timing_analysis_plot(timing_differences: List[float], shot_numbers: List[int],
                                   title: str = "Timing Analysis Results") -> plt.Figure:
        """
        タイミング解析結果のプロットを作成
        
        Args:
            timing_differences: タイミング差のリスト
            shot_numbers: ショット番号のリスト
            title: タイトル
            
        Returns:
            matplotlib図オブジェクト
        """
        PlotUtils.setup_matplotlib_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(settings.DEFAULT_FIGURE_SIZE[0], 
                                                     settings.DEFAULT_FIGURE_SIZE[1] * 1.5))
        
        # タイミング差のトレンド
        ax1.plot(shot_numbers, timing_differences, 'o-', color=settings.COLORS['primary'],
                linewidth=settings.DEFAULT_LINE_WIDTH, markersize=settings.DEFAULT_MARKER_SIZE)
        ax1.set_title('Timing Difference Trend')
        ax1.set_xlabel('Shot Number')
        ax1.set_ylabel('Timing Difference')
        ax1.grid(True, alpha=0.3)
        
        # 統計情報
        mean_timing = np.mean(timing_differences)
        std_timing = np.std(timing_differences)
        ax1.axhline(y=mean_timing, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_timing:.3f}')
        ax1.axhline(y=mean_timing + std_timing, color='orange', linestyle='--', alpha=0.7, 
                   label=f'+1σ: {mean_timing + std_timing:.3f}')
        ax1.axhline(y=mean_timing - std_timing, color='orange', linestyle='--', alpha=0.7, 
                   label=f'-1σ: {mean_timing - std_timing:.3f}')
        ax1.legend()
        
        # ヒストグラム
        ax2.hist(timing_differences, bins=20, color=settings.COLORS['info'], alpha=0.7, edgecolor='black')
        ax2.set_title('Timing Difference Distribution')
        ax2.set_xlabel('Timing Difference')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 統計情報をテキストで表示
        stats_text = f'Mean: {mean_timing:.3f}\nStd: {std_timing:.3f}\nCount: {len(timing_differences)}'
        ax2.text(0.7, 0.95, stats_text, transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def add_region_rectangles(ax: plt.Axes, regions: Dict[str, Dict[str, int]]):
        """
        プロットに領域の矩形を追加
        
        Args:
            ax: matplotlib軸オブジェクト
            regions: 領域定義の辞書
        """
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (name, region) in enumerate(regions.items()):
            color = colors[i % len(colors)]
            rect = Rectangle((region['xmin'], region['ymin']), 
                           region['xmax'] - region['xmin'], 
                           region['ymax'] - region['ymin'], 
                           linewidth=3, edgecolor=color, facecolor='none', 
                           label=f'{name} Region')
            ax.add_patch(rect)
    
    @staticmethod
    def save_plot_to_streamlit(fig: plt.Figure, filename: str = None) -> bytes:
        """
        プロットをStreamlitで表示し、バイトデータとして保存
        
        Args:
            fig: matplotlib図オブジェクト
            filename: ファイル名（オプション）
            
        Returns:
            画像のバイトデータ
        """
        import io
        
        # Streamlitで表示
        st.pyplot(fig)
        
        # バイトデータとして保存
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=settings.DEFAULT_DPI, bbox_inches='tight')
        img_buffer.seek(0)
        
        # ダウンロードボタンを提供
        if filename:
            st.download_button(
                label=f"📥 {filename}をダウンロード",
                data=img_buffer.getvalue(),
                file_name=f"{filename}.png",
                mime="image/png"
            )
        
        return img_buffer.getvalue()