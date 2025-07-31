import numpy as np
import struct
from typing import Tuple, Optional, Dict, Any

from ..config.settings import get_config
from ..utils.logger_manager import log_info, log_error, log_debug, log_warning


class ImgParser:
    """IMGファイル解析ユーティリティクラス"""

    @staticmethod
    def _detect_metadata_end(f, file_size: int) -> int:
        """
        IMGファイルのメタデータ終了位置を検出
        最後のCRLF (0d0a) パターンを探し、その後のnullバイトやパディングも考慮
        
        Args:
            f: ファイルオブジェクト
            file_size: ファイルサイズ
            
        Returns:
            メタデータ終了位置（バイナリデータ開始位置）
        """
        original_pos = f.tell()
        f.seek(0)
        
        # ファイル全体を読み込んでCRLFパターンを検索
        # 大きなファイルの場合は最初の10KB程度をチェック
        search_size = min(10240, file_size)
        data = f.read(search_size)
        
        # CRLF (0x0d0a) パターンの最後の出現位置を検索
        crlf_pattern = b'\x0d\x0a'
        last_pos = -1
        pos = 0
        
        # 全てのCRLFパターンを検索し、最後のものを記録
        while True:
            pos = data.find(crlf_pattern, pos)
            if pos == -1:
                break
            last_pos = pos
            pos += len(crlf_pattern)
        
        if last_pos != -1:
            # 最後のCRLFパターンが見つかった場合
            base_pos = last_pos + len(crlf_pattern)
            
            # 次の数バイトをチェックして実際のバイナリデータ開始位置を特定
            check_start = base_pos
            check_end = min(check_start + 20, len(data))  # 最大20バイト先まで確認
            
            # nullバイト（0x00）をスキップしつつ、明らかにバイナリデータと思われる位置を探す
            for i in range(check_start, check_end):
                if i + 1 < len(data):
                    # 連続する2バイトが両方とも1バイト値の範囲（0-255）内で、
                    # かつnullバイトではない場合、バイナリデータの開始と判定
                    byte1, byte2 = data[i], data[i + 1]
                    if byte1 != 0 and byte2 == 0 and byte1 < 256:  # リトルエンディアンの2バイト値
                        metadata_end = i
                        break
            else:
                # 明確なパターンが見つからない場合、CRLF直後から開始
                metadata_end = base_pos
        else:
            # CRLFが見つからない場合、デフォルトで717バイトを使用
            log_warning("CRLFパターンが見つかりません。デフォルト値(717)を使用します。", "img_parser")
            metadata_end = 717
        
        f.seek(original_pos)
        return metadata_end

    @staticmethod
    def _parse_metadata(metadata_string: str) -> Dict[str, Any]:
        """メタデータ文字列をパースして辞書に変換"""
        metadata_lines = metadata_string.splitlines()
        metadata_dict = {}
        
        for lines in metadata_lines:
            dict_temp = {}
            split_temp = lines.split(',')
            string_split = []
            
            for string in split_temp:
                if string.find('=') != -1:
                    string_split.append(string)
                elif len(string_split) == 0:
                    string_split.append(string)
                else:
                    string_split[-1] += ',' + string
            
            dict_temp = {'fieldname': string_split[0].strip("[]")}
            
            for string in string_split[1:]:
                element_split = string.split('=', 1)
                dict_temp[element_split[0]] = element_split[1]
            
            if len(dict_temp) > 1:
                metadata_dict[dict_temp['fieldname']] = dict_temp
        
        return metadata_dict

    @staticmethod
    def _read_image_data(f, metadata_end: int, data_x: int, data_y: int, length: int) -> np.ndarray:
        """
        画像データを読み込んでnumpy配列に変換
        """
        f.seek(metadata_end)
        
        # 利用可能なファイルサイズを確認
        current_pos = f.tell()
        f.seek(0, 2)
        file_end = f.tell()
        available_bytes = file_end - current_pos
        f.seek(current_pos)
        
        # 全画像データを一度に読み込み
        total_pixels = data_x * data_y
        expected_bytes = total_pixels * length
        bytes_to_read = min(expected_bytes, (available_bytes // length) * length)
        
        raw_data = f.read(bytes_to_read)
        
        if len(raw_data) != bytes_to_read:
            raise ValueError(f"Data read error: expected {bytes_to_read}, got {len(raw_data)}")
        
        # 設定からエンディアン設定を取得
        byte_order_setting = get_config('files.img_settings.byte_order', 'little')
        log_debug(f"IMGファイルエンディアン設定: {byte_order_setting}", "img_parser")
        
        # エンディアンを決定（デフォルトはリトルエンディアン）
        if byte_order_setting == 'big':
            endian = '>'
            log_info("ビッグエンディアンを使用", "img_parser")
        else:
            endian = '<'
            log_info("リトルエンディアンを使用", "img_parser")
        
        # XSCVa48452分析結果に基づき、符号付き16ビットリトルエンディアン形式でデータを読み込み
        if length == 1:
            # 1バイト: 符号なし整数
            data_array = np.frombuffer(raw_data, dtype=f'{endian}u1')
        elif length == 2:
            # 2バイト: 符号付き16ビット（XSCVa48452分析で確認済み）
            data_array = np.frombuffer(raw_data, dtype=f'{endian}i2')
            log_info("符号付き16ビットリトルエンディアン形式でデータを読み込み", "img_parser")
        elif length == 4:
            # 4バイト: 符号付き32ビット
            data_array = np.frombuffer(raw_data, dtype=f'{endian}i4')
        elif length == 8:
            # 8バイト: 符号付き64ビット
            data_array = np.frombuffer(raw_data, dtype=f'{endian}i8')
        else:
            raise ValueError(f"Unsupported bytes per pixel: {length}")
        
        # 実際に読み込めたピクセル数に基づいて形状を調整
        pixels_available = len(data_array)
        if pixels_available < total_pixels:
            # 完全な行数を計算
            complete_rows = pixels_available // data_x
            data_array = data_array[:complete_rows * data_x]
            data_array = data_array.reshape(complete_rows, data_x)
        else:
            data_array = data_array.reshape(data_y, data_x)
        
        log_debug(f"データ配列形状: {data_array.shape}, dtype: {data_array.dtype}", "img_parser")
        return data_array

    @staticmethod
    def load_img_file(filepath: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[str]]:
        """
        .imgファイルを読み込んでnumpy配列として返す
        
        Args:
            filepath: ファイルパス
            
        Returns:
            データ配列、メタデータ、エラーメッセージのタプル
        """
        log_debug(f"IMGファイル解析開始: {filepath}", "img_parser")
        
        try:
            with open(filepath, "rb") as f:
                # ファイルサイズを取得
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(0)
                log_debug(f"IMGファイルサイズ: {file_size} bytes", "img_parser")
                
                # メタデータの終了位置を正確に検出
                metadata_end = ImgParser._detect_metadata_end(f, file_size)
                log_debug(f"メタデータ終了位置: {metadata_end}", "img_parser")
                
                # ヘッダー部分（最初の64バイト）をスキップしてメタデータを読み込み
                header_end = 64
                f.seek(header_end)
                bdata = f.read(metadata_end - header_end)
                metadata_string = bdata.decode('utf-8', errors='ignore')
                metadata_dict = ImgParser._parse_metadata(metadata_string)
                
                # 画像データのサイズを取得
                areGRBScan_str = metadata_dict['Acquisition']['areGRBScan']
                areGRBScan = areGRBScan_str.strip('"').split(',')
                
                data_x = int(areGRBScan[2]) - int(areGRBScan[0])
                data_y = int(areGRBScan[3]) - int(areGRBScan[1])
                length = int(metadata_dict['Acquisition']['BytesPerPixel'])
                log_debug(f"画像データサイズ: {data_x}x{data_y}, {length} bytes/pixel", "img_parser")
                
                # 画像データの読み込み
                data_array = ImgParser._read_image_data(f, metadata_end, data_x, data_y, length)
                log_info(f"IMGファイル読み込み完了: shape={data_array.shape}", "img_parser")
                
                return data_array, metadata_dict, None
                
        except Exception as e:
            error_msg = str(e)
            log_error(f"IMGファイル読み込みエラー: {error_msg}", "img_parser", exc_info=True)
            return None, None, error_msg
