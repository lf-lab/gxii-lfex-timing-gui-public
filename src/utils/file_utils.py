"""
ファイルユーティリティモジュール
ファイル操作関連の共通機能を提供
"""
import os
import glob
from pathlib import Path
from typing import List, Optional

from ..config.settings import settings


class FileUtils:
    """ファイル操作ユーティリティクラス"""
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        ファイルサイズを人間が読みやすい形式にフォーマット
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    @staticmethod
    def get_supported_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
        """
        指定されたディレクトリからサポートされているファイルを取得
        
        Args:
            directory: 検索ディレクトリ
            extensions: サポートする拡張子のリスト
            
        Returns:
            見つかったファイルのリスト
        """
        if extensions is None:
            extensions = settings.SUPPORTED_FORMATS
        
        files = []
        for ext in extensions:
            pattern = os.path.join(directory, f"*{ext}")
            files.extend(glob.glob(pattern))
        
        return sorted(files)
    
    @staticmethod
    def create_output_directory(base_path: str = None) -> Path:
        """
        出力ディレクトリを作成
        
        Args:
            base_path: ベースパス
            
        Returns:
            作成された出力ディレクトリのパス
        """
        if base_path is None:
            base_path = Path.cwd()
        else:
            base_path = Path(base_path)
        
        output_dir = base_path / settings.OUTPUT_DIR
        output_dir.mkdir(exist_ok=True)
        
        return output_dir
    
    @staticmethod
    def validate_file_path(filepath: str) -> bool:
        """
        ファイルパスの妥当性をチェック
        
        Args:
            filepath: チェックするファイルパス
            
        Returns:
            妥当性（True/False）
        """
        try:
            path = Path(filepath)
            return path.exists() and path.is_file()
        except Exception:
            return False
    
    @staticmethod
    def get_file_info(filepath: str) -> dict:
        """
        ファイル情報を取得
        
        Args:
            filepath: ファイルパス
            
        Returns:
            ファイル情報の辞書
        """
        try:
            path = Path(filepath)
            stat = path.stat()
            
            return {
                'name': path.name,
                'size': stat.st_size,
                'size_formatted': FileUtils.format_file_size(stat.st_size),
                'extension': path.suffix,
                'modified': stat.st_mtime,
                'created': stat.st_ctime,
                'exists': True
            }
        except Exception as e:
            return {
                'name': '',
                'size': 0,
                'size_formatted': '0 B',
                'extension': '',
                'modified': 0,
                'created': 0,
                'exists': False,
                'error': str(e)
            }

    @staticmethod
    def get_version_from_file(filepath: str) -> str:
        """
        指定されたファイルからバージョン文字列を読み込む
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return "N/A"
        except Exception:
            return "Error reading version"

    @staticmethod
    def generate_shot_id_from_filepath(filepath: str) -> str:
        """
        ファイルパスからショットIDを生成
        
        Args:
            filepath: ファイルパス
            
        Returns:
            生成されたショットID
        """
        filename_base = Path(filepath).name.rsplit('.', 1)[0]
        # _ 以降の数字部分を削除
        if '_' in filename_base:
            parts = filename_base.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                filename_base = '_'.join(parts[:-1])
        
        shotid = "G" + filename_base.split('Va', 1)[1] if 'Va' in filename_base else filename_base
        return shotid