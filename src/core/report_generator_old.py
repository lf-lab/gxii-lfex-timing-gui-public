import streamlit as st
import pandas as pd
import io
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont # 追加

from src.config.settings import settings
from src.utils.logger_manager import log_info, log_error, log_warning
from src.utils.plot_manager import PlotManager
from src.core.waveform_library import WaveformLibrary # 必要に応じてインポート


class ReportGenerator:
    """レポート生成クラス"""

    def __init__(self, waveform_library: Optional[WaveformLibrary] = None):
        self.plot_manager = PlotManager()
        self.waveform_library = waveform_library if waveform_library is not None else WaveformLibrary()

    def generate_single_page_report(self, app_instance, report_author: str, sweep_speed_setting: str,
                                    laser_comments: str, app_version: str) -> Optional[bytes]:
        """単一ページのレポート（画像）を生成し、バイトデータを返す"""
        try:
            log_info("単一ページレポート生成開始", "report_generator")

            results = st.session_state.analysis_results
            if results is None:
                log_warning("解析結果がありません。レポートを生成できません。", "report_generator")
                return None

            shot_id = results.get('shotid', 'Unknown')
            
            # プロットを生成し、バイトデータとして取得
            plot_bytes_list = self.plot_manager.generate_report_plots(results, waveform_library=self.waveform_library)
            if not plot_bytes_list or len(plot_bytes_list) < 2: # スペースラインアウトが不要になったため2枚に
                log_error("プロット画像の生成に失敗しました。必要な画像が揃っていません。", "report_generator")
                return None

            # レポート画像のサイズ設定 (PowerPointワイドスクリーン 16:9 を想定)
            # 例: 1920x1080 (Full HD) または 1280x720
            report_width = 1920
            report_height = 1080
            background_color = (255, 255, 255) # 白

            report_image = Image.new('RGB', (report_width, report_height), background_color)
            draw = ImageDraw.Draw(report_image)

            # フォント設定
            try:
                # 日本語対応フォントのパスを試す
                font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc" # macOSの例
                if not Path(font_path).exists():
                    font_path = "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc" # 別のmacOSフォント
                if not Path(font_path).exists():
                    font_path = "/Library/Fonts/Arial.ttf" # 一般的なフォント（日本語非対応）
                
                title_font = ImageFont.truetype(font_path, 40)
                header_font = ImageFont.truetype(font_path, 28)
                text_font = ImageFont.truetype(font_path, 24)
                small_font = ImageFont.truetype(font_path, 18)
            except Exception:
                log_warning("指定されたフォントが見つかりません。デフォルトフォントを使用します。", "report_generator")
                title_font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
                small_font = ImageFont.load_default()

            # タイトル
            title_text = f"GXII-LFEX Timing Analysis Report - Shot ID: {shot_id}"
            draw.text((50, 30), title_text, font=title_font, fill=(0, 0, 0))

            # レポート情報 (左上)
            info_y_start = 100
            draw.text((50, info_y_start), f"Sweep Speed Setting: {sweep_speed_setting}", font=text_font, fill=(0, 0, 0))
            draw.text((50, info_y_start + 40), f"Laser Comments: {laser_comments}", font=text_font, fill=(0, 0, 0))

            # 解析結果の表示 (左側)
            results_y_start = info_y_start + 100
            draw.text((50, results_y_start), "Analysis Results:", font=header_font, fill=(0, 0, 0))
            
            # 基準値に対するGXII, LFEXのピークタイミング
            gxii_peak_relative = results.get('gxii_peak_relative', 0.0)
            lfex_peak_1_relative = results.get('lfex_peak_1_relative', 0.0)
            lfex_peak_2_relative = results.get('lfex_peak_2_relative', 0.0)
            reference_time_mode = results.get('reference_time_mode', 'Unknown')

            # Reference Modeの表示を改善
            reference_mode_display = {
                'gxii_peak': 'GXIIピーク基準',
                'lfex_peak': 'LFEXピーク基準',
                'absolute': '絶対時間基準',
                'manual': '手動設定基準',
                'gxii_rise': 'GXII立ち上がり基準',
                'streak_time': 'ストリーク画像時間基準',
                'custom_t0': 'カスタム波形t0基準'
            }.get(reference_time_mode, f'不明な基準 ({reference_time_mode})')

            draw.text((50, results_y_start + 40), f"基準モード: {reference_mode_display}", font=text_font, fill=(0, 0, 0))
            draw.text((50, results_y_start + 80), f"GXIIピーク (基準相対): {gxii_peak_relative:.3f} ns", font=text_font, fill=(0, 0, 0))
            draw.text((50, results_y_start + 120), f"LFEX 1stピーク (基準相対): {lfex_peak_1_relative:.3f} ns", font=text_font, fill=(0, 0, 0))
            if lfex_peak_2_relative != 0.0:
                draw.text((50, results_y_start + 160), f"LFEX 2ndピーク (基準相対): {lfex_peak_2_relative:.3f} ns", font=text_font, fill=(0, 0, 0))

            # プロットの配置
            # Raw Data Heatmap は左側、GXII & LFEX Profiles は右側に配置

            def _resize_and_paste(target_image, source_image_bytes, target_width, target_height, position):
                img = Image.open(io.BytesIO(source_image_bytes))
                img_width, img_height = img.size

                # アスペクト比を維持してリサイズ
                aspect_ratio = img_width / img_height
                if target_width / target_height > aspect_ratio:
                    # ターゲットの高さに合わせて幅を調整
                    new_height = target_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # ターゲットの幅に合わせて高さを調整
                    new_width = target_width
                    new_height = int(new_width / aspect_ratio)
                
                img = img.resize((new_width, new_height), Image.LANCZOS)

                # 中央に配置するためのオフセットを計算
                offset_x = position[0] + (target_width - new_width) // 2
                offset_y = position[1] + (target_height - new_height) // 2

                target_image.paste(img, (offset_x, offset_y))

            # 左側のプロット領域
            left_plot_area_x = 50
            left_plot_area_y = results_y_start + 200
            left_plot_area_width = report_width // 2 - 100 # 左側の余白を考慮
            left_plot_area_height = report_height - left_plot_area_y - 50 # テキストの下から下部余白まで

            # プロット1: 生データヒートマップ (左側)
            _resize_and_paste(report_image, plot_bytes_list[0], left_plot_area_width, left_plot_area_height, (left_plot_area_x, left_plot_area_y))

            # 右側のプロット領域
            right_plot_area_x = report_width // 2 + 50
            right_plot_area_y = 50
            right_plot_area_width = report_width // 2 - 100 # 右側の余白を考慮
            right_plot_area_height = report_height - 100 # 上下余白を考慮

            # プロット2: 2パネル比較 (右側)
            _resize_and_paste(report_image, plot_bytes_list[1], right_plot_area_width, right_plot_area_height, (right_plot_area_x, right_plot_area_y))

            # AuthorとApp Versionを右下に配置
            footer_text_author = f"Author: {report_author}"
            footer_text_version = f"App Version: {app_version}"
            
            # テキストのバウンディングボックスを取得
            author_bbox = draw.textbbox((0, 0), footer_text_author, font=small_font)
            version_bbox = draw.textbbox((0, 0), footer_text_version, font=small_font)

            author_text_height = author_bbox[3] - author_bbox[1]
            version_text_height = version_bbox[3] - version_bbox[1]

            # 右下からのオフセット
            margin = 20
            draw.text((report_width - author_bbox[2] - margin, report_height - version_text_height - author_text_height - margin), footer_text_author, font=small_font, fill=(0, 0, 0))
            draw.text((report_width - version_bbox[2] - margin, report_height - version_text_height - margin), footer_text_version, font=small_font, fill=(0, 0, 0))

            # バイトデータとして保存
            img_byte_arr = io.BytesIO()
            report_image.save(img_byte_arr, format='PNG')
            report_bytes = img_byte_arr.getvalue()
            
            log_info(f"単一ページレポート生成完了: Shot ID={shot_id}", "report_generator")
            return report_bytes
            
            log_info(f"単一ページレポート生成完了: Shot ID={shot_id}", "report_generator")
            return report_bytes

        except Exception as e:
            log_error(f"単一ページレポート生成エラー: {str(e)}", "report_generator", exc_info=True)
            return None


# Streamlitのセッション状態にReportGeneratorインスタンスを保持
# これにより、アプリの再実行時に毎回初期化されるのを防ぐ
if 'report_generator' not in st.session_state:
    st.session_state.report_generator = ReportGenerator()

def get_report_generator() -> ReportGenerator:
    return st.session_state.report_generator
