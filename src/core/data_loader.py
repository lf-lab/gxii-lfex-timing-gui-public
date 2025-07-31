"""
データローダーモジュール
各種データファイルの読み込み機能を提供
"""
import numpy as np
import struct
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from ..config.settings import settings
from ..utils.logger_manager import log_info, log_error, log_debug, log_warning
from .img_parser import ImgParser


class DataLoader:
    """データファイルローダークラス"""
    
    @staticmethod
    def determine_file_type(filepath: str) -> str:
        """
        ファイル拡張子に基づいてファイルタイプを判定
        
        Args:
            filepath: ファイルパス
            
        Returns:
            ファイルタイプ ('img', 'txt', 'unknown')
        """
        if filepath.lower().endswith('.img'):
            return 'img'
        elif filepath.lower().endswith('.txt'):
            return 'txt'
        else:
            return 'unknown'
    
    @staticmethod
    def load_data_file(filepath: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        ファイルタイプに応じてデータを読み込む
        
        Args:
            filepath: ファイルパス
            
        Returns:
            データ配列とエラーメッセージのタプル
        """
        log_info(f"データファイル読み込み開始: {filepath}", "data_loader")
        
        file_type = DataLoader.determine_file_type(filepath)
        log_debug(f"ファイルタイプ判定: {file_type}", "data_loader")
        
        if file_type == 'img':
            data_array, metadata, error = ImgParser.load_img_file(filepath)
            if error:
                log_error(f"IMGファイル読み込みエラー: {error}", "data_loader")
            else:
                log_info(f"IMGファイル読み込み成功: shape={data_array.shape if data_array is not None else 'None'}", "data_loader")
            return data_array, error
        elif file_type == 'txt':
            data_array, error = DataLoader.load_txt_file(filepath)
            if error:
                log_error(f"TXTファイル読み込みエラー: {error}", "data_loader")
            else:
                log_info(f"TXTファイル読み込み成功: shape={data_array.shape if data_array is not None else 'None'}", "data_loader")
            return data_array, error
        else:
            error_msg = f"Unsupported file type: {filepath}"
            log_error(error_msg, "data_loader")
            return None, error_msg
    
    @staticmethod
    def load_txt_file(filepath: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        .txtファイルを読み込む
        
        Args:
            filepath: ファイルパス
            
        Returns:
            データ配列とエラーメッセージのタプル
        """
        log_debug(f"TXTファイル解析開始: {filepath}", "data_loader")
        
        try:
            data = []
            with open(filepath) as f:
                lines = f.readlines()
                log_debug(f"TXTファイル総行数: {len(lines)}", "data_loader")
                
                for i, line in enumerate(lines):
                    if i <= 1:  # ヘッダー行をスキップ
                        continue
                    row_str = line.strip().split('\t')
                    row_int = [int(c.strip()) for c in row_str]
                    row_int.pop(0)  # 最初の列を除去
                    data.append(row_int)
            
            data_array = np.array(data)
            log_info(f"TXTファイル読み込み完了: shape={data_array.shape}, データ行数={len(data)}", "data_loader")
            return data_array, None
        except Exception as e:
            error_msg = str(e)
            log_error(f"TXTファイル読み込みエラー: {error_msg}", "data_loader", exc_info=True)
            return None, error_msg