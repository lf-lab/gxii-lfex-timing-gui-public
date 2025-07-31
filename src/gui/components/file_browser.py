import streamlit as st
import os
from pathlib import Path
from typing import List, Optional


def file_browser(label: str, default_path: str = "", file_types: Optional[List[str]] = None, 
                is_directory: bool = False) -> Optional[str]:
    """
    ファイル/ディレクトリブラウザ機能
    
    Args:
        label: ブラウザのラベル
        default_path: デフォルトパス
        file_types: 対象ファイルタイプのリスト
        is_directory: ディレクトリ選択モード
        
    Returns:
        選択されたパス
    """
    if is_directory:
        # ディレクトリ選択
        current_path = st.session_state.get(f"{label}_current_dir", 
                                           str(Path(default_path).parent) if default_path else str(Path.home()))
    else:
        # ファイル選択
        current_path = st.session_state.get(f"{label}_current_dir", 
                                           str(Path(default_path).parent) if default_path else str(Path.home()))
    
    # クイックアクセスボタン
    st.write("🚀 クイックアクセス:")
    quick_access_col1, quick_access_col2, quick_access_col3 = st.columns(3)
    
    with quick_access_col1:
        if st.button("🏠 ホーム", key=f"{label}_home"):
            st.session_state[f"{label}_current_dir"] = str(Path.home())
            st.rerun()
    
    with quick_access_col2:
        if st.button("🗂️ デスクトップ", key=f"{label}_desktop"):
            desktop_path = Path.home() / "Desktop"
            if desktop_path.exists():
                st.session_state[f"{label}_current_dir"] = str(desktop_path)
                st.rerun()
    
    with quick_access_col3:
        if st.button("💾 ボリューム", key=f"{label}_volumes"):
            volumes_path = Path("/Volumes")
            if volumes_path.exists():
                st.session_state[f"{label}_current_dir"] = str(volumes_path)
                st.rerun()
    
    # 現在のパス表示と入力
    st.write(f"📁 現在のディレクトリ: `{current_path}`")
    
    # パス手動入力
    manual_path = st.text_input("📝 パスを直接入力:", value=current_path, key=f"{label}_manual_path")
    if st.button("移動", key=f"{label}_manual_go"):
        if Path(manual_path).exists():
            st.session_state[f"{label}_current_dir"] = manual_path
            st.rerun()
        else:
            st.error("指定されたパスが存在しません")
    
    # ディレクトリ内容を表示
    try:
        items = _list_directory_items(current_path, file_types, is_directory)
        
        if not items:
            st.info("表示できるアイテムがありません")
            return None
        
        # アイテム選択
        selected_item = st.selectbox(
            "📂 アイテムを選択:",
            options=[""] + items,
            key=f"{label}_item_select"
        )
        
        if selected_item:
            full_path = Path(current_path) / selected_item
            
            if full_path.is_dir():
                # ディレクトリの場合
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📁 開く", key=f"{label}_open_dir"):
                        st.session_state[f"{label}_current_dir"] = str(full_path)
                        st.rerun()
                
                if is_directory:
                    with col2:
                        if st.button("✅ 選択", key=f"{label}_select_dir"):
                            return str(full_path)
            else:
                # ファイルの場合
                if not is_directory:
                    if st.button("✅ 選択", key=f"{label}_select_file"):
                        return str(full_path)
                    
                    # ファイル情報表示
                    file_size = full_path.stat().st_size
                    st.info(f"📄 ファイルサイズ: {format_file_size(file_size)}")
        
    except Exception as e:
        st.error(f"ディレクトリの読み込みエラー: {e}")
    
    return None

def _list_directory_items(directory_path: str, file_types: Optional[List[str]] = None, 
                         is_directory: bool = False) -> List[str]:
    """ディレクトリ内のアイテムをリスト"""
    try:
        path = Path(directory_path)
        items = []
        
        # 親ディレクトリへの移動オプション
        if path.parent != path:
            items.append(".. (親ディレクトリ)")
        
        # ディレクトリを最初に追加
        for item in sorted(path.iterdir()):
            if item.is_dir():
                items.append(f"📁 {item.name}")
        
        # ファイルを追加（ディレクトリ選択モードでない場合）
        if not is_directory:
            for item in sorted(path.iterdir()):
                if item.is_file():
                    if file_types is None or any(item.suffix.lower() in ft for ft in file_types):
                        items.append(f"📄 {item.name}")
        
        return items
    except Exception:
        return []

def format_file_size(size_bytes: int) -> str:
    """ファイルサイズを人間が読みやすい形式にフォーマット"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"
