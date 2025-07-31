"""
XSC (X-ray Streak Camera) ショット・サマリー可視化モジュール
1ショット = 1PDFレポート生成システム

ChatGPT o3提案仕様:
- A4横向きレイアウト (297mm x 210mm)
- 5つのパネル配置:
  ① 左上: 生画像 (Raw Image)
  ② 右上: 処理画像 + オーバーレイ (Processed Image + Overlays) 
  ③ 左下: 時間ラインアウト (Time Lineout)
  ④ 中下: 空間ラインアウト (Space Lineout)  
  ⑤ 右下: アノテーションボックス (Annotation Box)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from datetime import datetime

from .plot_manager import PlotManager, PlotTheme
from ..config.settings import settings
from .logger_manager import log_info, log_warning, log_error


class XSCShotSummaryGenerator:
    """XSCショット・サマリー生成クラス"""
    
    def __init__(self, plot_manager: Optional[PlotManager] = None):
        """
        初期化
        
        Args:
            plot_manager: 既存のPlotManagerインスタンス（オプション）
        """
        self.plot_manager = plot_manager or PlotManager()
        
        # A4横向きサイズ設定 (297mm x 210mm)
        self.page_width = 11.69  # 297mm in inches
        self.page_height = 8.27  # 210mm in inches
        
        # パネル配置設定
        self.panel_configs = {
            'raw_image': {'position': [0.05, 0.55, 0.4, 0.4]},      # 左上
            'processed_image': {'position': [0.55, 0.55, 0.4, 0.4]}, # 右上  
            'time_lineout': {'position': [0.05, 0.05, 0.25, 0.4]},   # 左下
            'space_lineout': {'position': [0.35, 0.05, 0.25, 0.4]},  # 中下
            'annotation_box': {'position': [0.65, 0.05, 0.3, 0.4]}   # 右下
        }
        
        log_info("XSCショット・サマリー生成器を初期化", "xsc_shot_summary")
    
    def generate_shot_summary_pdf(self, shot_data: Dict[str, Any], 
                                 output_path: Union[str, Path],
                                 shot_id: str = None,
                                 dpi: int = 300) -> bool:
        """
        XSCショット・サマリーPDFを生成
        
        Args:
            shot_data: ショットデータ辞書
                - raw_data: 生データ配列 (np.ndarray)
                - processed_data: 処理済みデータ配列 (np.ndarray) 
                - time_profile: 時間プロファイル {'x': np.ndarray, 'y': np.ndarray}
                - space_profile: 空間プロファイル {'x': np.ndarray, 'y': np.ndarray}
                - analysis_results: 解析結果辞書
                - metadata: メタデータ辞書
            output_path: 出力PDFファイルパス
            shot_id: ショットID（オプション）
            dpi: PDF解像度
            
        Returns:
            bool: 生成成功フラグ
        """
        try:
            output_path = Path(output_path)
            shot_id = shot_id or shot_data.get('shot_id', f"shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            log_info(f"XSCショット・サマリーPDF生成開始: {shot_id}", "xsc_shot_summary")
            
            with PdfPages(output_path) as pdf:
                # メイン図を作成
                fig = self._create_shot_summary_figure(shot_data, shot_id)
                
                # PDFに保存
                pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
                plt.close(fig)
                
                # メタデータページ（オプション）
                if shot_data.get('metadata') and shot_data['metadata'].get('include_metadata_page', False):
                    metadata_fig = self._create_metadata_page(shot_data, shot_id)
                    pdf.savefig(metadata_fig, bbox_inches='tight', dpi=dpi)
                    plt.close(metadata_fig)
            
            log_info(f"XSCショット・サマリーPDF生成完了: {output_path}", "xsc_shot_summary")
            return True
            
        except Exception as e:
            log_error(f"XSCショット・サマリーPDF生成失敗: {e}", "xsc_shot_summary", exc_info=True)
            return False
    
    def generate_shot_summary_pdf_bytes(self, shot_data: Dict[str, Any], 
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
        import io
        
        try:
            log_info(f"XSCショット・サマリーPDFバイト生成開始: shot_id={shot_id}", "xsc_shot_summary")
            
            # バイトバッファを作成
            pdf_buffer = io.BytesIO()
            
            # パラメータをshot_dataに設定
            shot_data['generation_params'] = {
                'dpi': dpi,
                'colormap': colormap,
                'annotation_level': annotation_level,
                'include_metadata_page': include_metadata_page
            }
            
            with PdfPages(pdf_buffer) as pdf:
                # メイン図を作成
                fig = self._create_shot_summary_figure(shot_data, shot_id)
                
                # PDFに保存
                pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
                plt.close(fig)
                
                # メタデータページ（オプション）
                if include_metadata_page:
                    metadata_fig = self._create_metadata_page(shot_data, shot_id)
                    pdf.savefig(metadata_fig, bbox_inches='tight', dpi=dpi)
                    plt.close(metadata_fig)
            
            pdf_bytes = pdf_buffer.getvalue()
            pdf_buffer.close()
            
            log_info(f"XSCショット・サマリーPDFバイト生成完了: size={len(pdf_bytes)}bytes", "xsc_shot_summary")
            return pdf_bytes
            
        except Exception as e:
            log_error(f"XSCショット・サマリーPDFバイト生成失敗: {e}", "xsc_shot_summary", exc_info=True)
            return None
    
    def _create_shot_summary_figure(self, shot_data: Dict[str, Any], shot_id: str) -> Figure:
        """
        ショット・サマリーメイン図を作成
        
        Args:
            shot_data: ショットデータ辞書
            shot_id: ショットID
            
        Returns:
            matplotlib図オブジェクト
        """
        # A4横向きサイズでフィギュア作成
        fig = plt.figure(figsize=(self.page_width, self.page_height))
        fig.suptitle(f'XSC Shot Summary: {shot_id}', fontsize=16, fontweight='bold')
        
        # ① 左上: 生画像パネル
        self._add_raw_image_panel(fig, shot_data)
        
        # ② 右上: 処理画像 + オーバーレイパネル
        self._add_processed_image_panel(fig, shot_data)
        
        # ③ 左下: 時間ラインアウトパネル
        self._add_time_lineout_panel(fig, shot_data)
        
        # ④ 中下: 空間ラインアウトパネル
        self._add_space_lineout_panel(fig, shot_data)
        
        # ⑤ 右下: アノテーションボックスパネル
        self._add_annotation_panel(fig, shot_data)
        
        # レイアウト調整
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95, 
                           hspace=0.3, wspace=0.3)
        
        return fig
    
    def _add_raw_image_panel(self, fig: Figure, shot_data: Dict[str, Any]) -> None:
        """① 生画像パネルを追加"""
        pos = self.panel_configs['raw_image']['position']
        ax = fig.add_axes(pos)
        
        raw_data = shot_data.get('raw_data')
        if raw_data is not None:
            # 自動スケーリング
            vmin = float(np.percentile(raw_data, 5))
            vmax = float(np.percentile(raw_data, 95))
            
            # ヒートマップ表示
            im = ax.imshow(raw_data, cmap='hot', vmin=vmin, vmax=vmax,
                          extent=[0, raw_data.shape[1], raw_data.shape[0], 0],
                          aspect='auto')
            
            # カラーバー追加
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Intensity', fontsize=10)
            
            ax.set_title('① Raw Image Data', fontsize=12, fontweight='bold')
            ax.set_xlabel('X Pixel', fontsize=10)
            ax.set_ylabel('Y Pixel', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Raw Data Available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_title('① Raw Image Data', fontsize=12, fontweight='bold')
    
    def _add_processed_image_panel(self, fig: Figure, shot_data: Dict[str, Any]) -> None:
        """② 処理画像 + オーバーレイパネルを追加"""
        pos = self.panel_configs['processed_image']['position']
        ax = fig.add_axes(pos)
        
        processed_data = shot_data.get('processed_data')
        if processed_data is not None:
            # 自動スケーリング
            vmin = float(np.percentile(processed_data, 5))
            vmax = float(np.percentile(processed_data, 95))
            
            # ヒートマップ表示
            im = ax.imshow(processed_data, cmap='hot', vmin=vmin, vmax=vmax,
                          extent=[0, processed_data.shape[1], processed_data.shape[0], 0],
                          aspect='auto')
            
            # オーバーレイ追加
            self._add_overlay_elements(ax, shot_data)
            
            # カラーバー追加
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Intensity', fontsize=10)
            
            ax.set_title('② Processed Image + Overlays', fontsize=12, fontweight='bold')
            ax.set_xlabel('X Pixel', fontsize=10)
            ax.set_ylabel('Y Pixel', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Processed Data Available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_title('② Processed Image + Overlays', fontsize=12, fontweight='bold')
    
    def _add_overlay_elements(self, ax: plt.Axes, shot_data: Dict[str, Any]) -> None:
        """オーバーレイ要素を追加"""
        # GXII/LFEX領域表示
        regions = shot_data.get('regions')
        if regions:
            for region_name, params in regions.items():
                if region_name == 'gxii':
                    color = 'red'
                    label = 'GXII Region'
                elif region_name == 'lfex':
                    color = 'blue'
                    label = 'LFEX Region'
                else:
                    color = 'gray'
                    label = f'{region_name.upper()} Region'
                
                width = params['xmax'] - params['xmin']
                height = params['ymax'] - params['ymin']
                
                rect = Rectangle((params['xmin'], params['ymin']), width, height,
                               linewidth=2, edgecolor=color, facecolor='none',
                               label=label, alpha=0.8)
                ax.add_patch(rect)
        
        # ピーク位置表示
        peaks = shot_data.get('peaks')
        if peaks:
            # peaks が辞書形式 {'gxii_peak': {'time': x, 'value': y}, ...} の場合
            if isinstance(peaks, dict):
                for i, (peak_name, peak_data) in enumerate(peaks.items()):
                    if isinstance(peak_data, dict) and 'time' in peak_data and 'value' in peak_data:
                        x, y = peak_data['time'], peak_data['value']
                        color = 'red' if 'gxii' in peak_name.lower() else 'cyan'
                        ax.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='black',
                               markeredgewidth=1, label=f'{peak_name}' if i < 3 else "")
            # peaks がタプルのリスト [(x1, y1), (x2, y2), ...] の場合
            elif isinstance(peaks, (list, tuple)):
                for i, peak in enumerate(peaks):
                    if len(peak) >= 2:
                        x, y = peak[0], peak[1]
                        ax.plot(x, y, 'yo', markersize=8, markeredgecolor='black',
                               markeredgewidth=1, label=f'Peak {i+1}' if i == 0 else "")
        
        # 凡例追加
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper right', fontsize=8,
                     bbox_to_anchor=(1.0, 1.0), framealpha=0.8)
    
    def _add_time_lineout_panel(self, fig: Figure, shot_data: Dict[str, Any]) -> None:
        """③ 時間ラインアウトパネルを追加"""
        pos = self.panel_configs['time_lineout']['position']
        ax = fig.add_axes(pos)
        
        time_profile = shot_data.get('time_profile')
        if time_profile and 'x' in time_profile and 'y' in time_profile:
            x_data = time_profile['x']
            y_data = time_profile['y']
            
            # メインプロット
            ax.plot(x_data, y_data, 'b-', linewidth=2, label='Time Profile')
            
            # ピーク表示
            peaks = time_profile.get('peaks')
            if peaks is not None:
                ax.plot(x_data[peaks], y_data[peaks], 'ro', 
                       markersize=6, label='Peaks')
            
            # フィッティング曲線
            fitted = time_profile.get('fitted_curve')
            if fitted is not None:
                fit_x, fit_y = fitted
                ax.plot(fit_x, fit_y, 'r--', linewidth=1.5, 
                       label='Fitted Curve', alpha=0.7)
            
            ax.set_title('③ Time Lineout', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (ns)', fontsize=10)
            ax.set_ylabel('Intensity', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Time Profile Data',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='red')
            ax.set_title('③ Time Lineout', fontsize=12, fontweight='bold')
    
    def _add_space_lineout_panel(self, fig: Figure, shot_data: Dict[str, Any]) -> None:
        """④ 空間ラインアウトパネルを追加"""
        pos = self.panel_configs['space_lineout']['position']
        ax = fig.add_axes(pos)
        
        space_profile = shot_data.get('space_profile')
        if space_profile and 'x' in space_profile and 'y' in space_profile:
            x_data = space_profile['x']
            y_data = space_profile['y']
            
            # メインプロット
            ax.plot(x_data, y_data, 'g-', linewidth=2, label='Space Profile')
            
            # ピーク表示
            peaks = space_profile.get('peaks')
            if peaks is not None:
                ax.plot(x_data[peaks], y_data[peaks], 'ro',
                       markersize=6, label='Peaks')
            
            ax.set_title('④ Space Lineout', fontsize=12, fontweight='bold')
            ax.set_xlabel('Position (pixel)', fontsize=10)
            ax.set_ylabel('Intensity', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Space Profile Data',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='red')
            ax.set_title('④ Space Lineout', fontsize=12, fontweight='bold')
    
    def _add_annotation_panel(self, fig: Figure, shot_data: Dict[str, Any]) -> None:
        """⑤ アノテーションボックスパネルを追加"""
        pos = self.panel_configs['annotation_box']['position']
        ax = fig.add_axes(pos)
        
        # 軸を非表示
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_title('⑤ Analysis Results', fontsize=12, fontweight='bold')
        
        # アノテーション情報を作成
        annotations = self._create_annotations(shot_data)
        
        # テキスト表示
        y_pos = 0.95
        for line in annotations:
            ax.text(0.05, y_pos, line, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   fontfamily='monospace')
            y_pos -= 0.08
    
    def _create_annotations(self, shot_data: Dict[str, Any]) -> List[str]:
        """アノテーション情報を作成"""
        annotations = []
        
        # ショット情報
        shot_id = shot_data.get('shot_id', 'Unknown')
        timestamp = shot_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        annotations.extend([
            f"Shot ID: {shot_id}",
            f"Timestamp: {timestamp}",
            ""
        ])
        
        # 解析結果
        results = shot_data.get('analysis_results', {})
        if results:
            annotations.append("Analysis Results:")
            
            # タイミング情報
            if 'timing_difference' in results:
                annotations.append(f"  Timing Diff: {results['timing_difference']:.3f} ns")
            
            if 'gxii_peak' in results:
                annotations.append(f"  GXII Peak: {results['gxii_peak']:.3f} ns")
            
            if 'lfex_peak' in results:
                annotations.append(f"  LFEX Peak: {results['lfex_peak']:.3f} ns")
            
            # 統計情報
            if 'statistics' in results:
                stats = results['statistics']
                annotations.append("")
                annotations.append("Statistics:")
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        annotations.append(f"  {key}: {value:.3f}")
                    else:
                        annotations.append(f"  {key}: {value}")
        
        # メタデータ
        metadata = shot_data.get('metadata', {})
        if metadata:
            annotations.append("")
            annotations.append("Metadata:")
            for key, value in metadata.items():
                if key != 'include_metadata_page':  # 内部フラグは除外
                    annotations.append(f"  {key}: {value}")
        
        return annotations
    
    def _create_metadata_page(self, shot_data: Dict[str, Any], shot_id: str) -> Figure:
        """メタデータページを作成"""
        fig = plt.figure(figsize=(self.page_width, self.page_height))
        fig.suptitle(f'XSC Shot Metadata: {shot_id}', fontsize=16, fontweight='bold')
        
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 詳細メタデータ表示
        metadata = shot_data.get('metadata', {})
        analysis_results = shot_data.get('analysis_results', {})
        
        all_info = {**metadata, **analysis_results}
        
        text_lines = []
        for key, value in all_info.items():
            if isinstance(value, dict):
                text_lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    text_lines.append(f"  {sub_key}: {sub_value}")
            else:
                text_lines.append(f"{key}: {value}")
        
        y_pos = 0.95
        for line in text_lines:
            ax.text(0.05, y_pos, line, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   fontfamily='monospace')
            y_pos -= 0.04
            if y_pos < 0.05:  # ページ下端に達した場合
                break
        
        return fig


def create_sample_shot_data() -> Dict[str, Any]:
    """サンプルショットデータを作成（テスト用）"""
    # サンプル生データ
    np.random.seed(42)
    raw_data = np.random.normal(100, 20, (100, 150)) + \
               50 * np.exp(-((np.arange(100)[:, None] - 50)**2 + 
                            (np.arange(150)[None, :] - 75)**2) / 200)
    
    # サンプル処理済みデータ
    processed_data = raw_data * 1.2
    
    # サンプル時間プロファイル
    time_x = np.linspace(0, 10, 100)
    time_y = 50 * np.exp(-(time_x - 5)**2 / 2) + np.random.normal(0, 2, 100)
    time_peaks = [np.argmax(time_y)]
    
    # サンプル空間プロファイル
    space_x = np.linspace(0, 150, 150)
    space_y = 40 * np.exp(-(space_x - 75)**2 / 300) + np.random.normal(0, 1.5, 150)
    
    return {
        'shot_id': 'TEST_SHOT_001',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'raw_data': raw_data,
        'processed_data': processed_data,
        'time_profile': {
            'x': time_x,
            'y': time_y,
            'peaks': time_peaks
        },
        'space_profile': {
            'x': space_x,
            'y': space_y
        },
        'regions': {
            'gxii': {'xmin': 20, 'xmax': 60, 'ymin': 10, 'ymax': 90},
            'lfex': {'xmin': 90, 'xmax': 130, 'ymin': 10, 'ymax': 90}
        },
        'peaks': [(75, 50), (125, 40)],
        'analysis_results': {
            'timing_difference': 2.345,
            'gxii_peak': 5.123,
            'lfex_peak': 7.468,
            'statistics': {
                'mean_intensity': 102.5,
                'max_intensity': 185.3,
                'snr': 12.8
            }
        },
        'metadata': {
            'laser_energy': '100 J',
            'exposure_time': '1.5 ns',
            'filter': 'None',
            'detector': 'XSC-1',
            'operator': 'GXII-LFEX Team'
        }
    }
