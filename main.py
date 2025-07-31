#!/usr/bin/env python3
"""
GXII-LFEX Timing Analysis GUI - メインエントリーポイント
新しいモジュール構造を使用
"""

import sys
import os
from pathlib import Path

# プロジェクトルートを設定
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

def main():
    """メイン関数"""
    try:
        # Streamlitセッション状態を使用して初回のみログ出力
        # まずstreamlitをインポート
        import streamlit as st
        
        # 初回起動時のみログを出力
        if 'main_initialized' not in st.session_state:
            # ロガーを初期化
            from src.utils.logger_manager import log_info, log_error, log_critical, initialize_gui_logger_queue
            initialize_gui_logger_queue() # 追加
            log_info("🚀 GXII-LFEX Timing Analysis GUI 起動中...", "main")
            st.session_state.main_initialized = True
        
        # 新しいGUIアプリケーションを起動
        from src.gui.app import main as run_app
        run_app()
        
    except ImportError as e:
        # ロガーが初期化される前にエラーが発生した場合はprint使用
        try:
            from src.utils.logger_manager import log_error
            log_error(f"❌ モジュールのインポートエラー: {e}", "main", exc_info=True)
            log_error("📦 必要な依存関係がインストールされているか確認してください: pip install -r requirements.txt", "main")
        except ImportError:
            # フォールバック: ロガーが利用できない場合
            print(f"❌ モジュールのインポートエラー: {e}")
            print("📦 必要な依存関係がインストールされているか確認してください:")
            print("   pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        try:
            from src.utils.logger_manager import log_critical
            log_critical(f"❌ アプリケーション起動エラー: {e}", "main", exc_info=True)
        except ImportError:
            print(f"❌ アプリケーション起動エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
