import tempfile
import time
from pathlib import Path
from typing import Optional


def save_uploaded_file(file_content: bytes, file_name: str) -> Optional[str]:
    """
    アップロードされたファイルのバイトデータを一時ディレクトリに保存
    
    Args:
        file_content: ファイルのバイトデータ
        file_name: 元のファイル名
        
    Returns:
        保存されたファイルのパス（失敗時はNone）
    """
    try:
        # 一時ディレクトリを作成
        temp_dir = Path(tempfile.gettempdir()) / "gxii_lfex_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        # ファイル名が重複しないように一意のファイル名を作成
        timestamp = str(int(time.time()))
        file_extension = Path(file_name).suffix
        safe_filename = f"{Path(file_name).stem}_{timestamp}{file_extension}"
        temp_file = temp_dir / safe_filename
        
        # ファイルを保存
        with open(temp_file, "wb") as f:
            f.write(file_content)
        
        return str(temp_file)
    except Exception:
        return None
