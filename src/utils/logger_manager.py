import logging
import logging.handlers
import streamlit as st
from collections import deque
from typing import Optional, Dict, Any

# GUI表示用のログを保持するキュー
# Streamlitの再実行でログが消えないようにセッション状態に保存
def initialize_gui_logger_queue():
    if 'gui_log_queue' not in st.session_state:
        st.session_state.gui_log_queue = deque(maxlen=500)

class GUILogHandler(logging.Handler):
    """Streamlit GUIにログを表示するためのカスタムハンドラ"""
    def emit(self, record):
        try:
            initialize_gui_logger_queue() # 追加
            msg = self.format(record)
            st.session_state.gui_log_queue.append({
                "timestamp": record.created,
                "level": record.levelname,
                "module": record.name,
                "message": msg,
                "exc_info": self.format_exception(record.exc_info) if record.exc_info else None
            })
        except Exception:
            self.handleError(record)

    def format_exception(self, exc_info) -> Optional[str]:
        """例外情報をフォーマットする"""
        if exc_info:
            import traceback
            return ''.join(traceback.format_exception(*exc_info))
        return None

class LoggerManager:
    """アプリケーション全体のロギングを管理するクラス"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.loggers = {}
        self.default_log_level = logging.INFO
        self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        
        # ルートロガーの設定
        logging.basicConfig(level=self.default_log_level, format=self.log_format, datefmt=self.date_format)
        
        # GUIハンドラの設定
        self.gui_handler = GUILogHandler()
        self.gui_handler.setFormatter(logging.Formatter(self.log_format, datefmt=self.date_format))
        
        # 既存のルートロガーにGUIハンドラを追加（重複防止）
        if not any(isinstance(handler, GUILogHandler) for handler in logging.getLogger().handlers):
            logging.getLogger().addHandler(self.gui_handler)

        self._initialized = True
        self.get_logger("logger_manager").info("LoggerManager initialized.")

    def get_logger(self, name: str) -> logging.Logger:
        """指定された名前のロガーを取得または作成"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.default_log_level)
            # ルートロガーにハンドラが設定されているため、ここではハンドラを追加しない
            # これにより、ログが重複して出力されるのを防ぐ
            logger.propagate = True  # ルートロガーに伝播させる
            self.loggers[name] = logger
        return self.loggers[name]

    def set_log_level(self, level: int, name: str = None):
        """ロガーのログレベルを設定"""
        if name:
            logger = self.get_logger(name)
            logger.setLevel(level)
        else:
            # 全てのロガーのレベルを設定
            for logger_name in self.loggers:
                self.loggers[logger_name].setLevel(level)
            logging.getLogger().setLevel(level) # ルートロガーも設定

    def get_gui_logs(self, level_filter: Optional[str] = None, limit: Optional[int] = None) -> list:
        """GUI表示用のログを取得"""
        initialize_gui_logger_queue() # 追加
        logs = list(st.session_state.gui_log_queue)
        
        if level_filter and level_filter != "ALL":
            logs = [log for log in logs if log["level"] == level_filter]
            
        if limit:
            logs = logs[-limit:] # 最新のログを返す
            
        return logs

    def clear_gui_logs(self):
        """GUI表示用のログをクリア"""
        initialize_gui_logger_queue() # 追加
        st.session_state.gui_log_queue.clear()
        self.get_logger("logger_manager").info("GUI logs cleared.")

    def get_log_statistics(self) -> Dict[str, Any]:
        """ログの統計情報を取得"""
        initialize_gui_logger_queue() # 追加
        logs = list(st.session_state.gui_log_queue)
        stats = {
            "total": len(logs),
            "level_distribution": {},
            "modules": set()
        }
        for log in logs:
            level = log["level"]
            module = log["module"]
            stats["level_distribution"][level] = stats["level_distribution"].get(level, 0) + 1
            stats["modules"].add(module)
        stats["modules"] = list(stats["modules"])
        return stats

    def export_logs_json(self, limit: Optional[int] = None) -> str:
        """
        GUIログをJSON形式でエクスポート
        """
        import json
        logs_to_export = self.get_gui_logs(limit=limit)
        # timestampを文字列に変換
        for log in logs_to_export:
            log["timestamp"] = pd.Timestamp(log["timestamp"], unit='s').strftime('%Y-%m-%d %H:%M:%S')
        return json.dumps(logs_to_export, indent=2, ensure_ascii=False)

# グローバルなLoggerManagerインスタンス
_logger_manager_instance = LoggerManager()

def get_logger_manager() -> LoggerManager:
    """LoggerManagerのシングルトンインスタンスを取得"""
    return _logger_manager_instance

def log_info(message: str, module_name: str = "app"):
    """INFOレベルのログを出力"""
    get_logger_manager().get_logger(module_name).info(message)

def log_warning(message: str, module_name: str = "app"):
    """WARNINGレベルのログを出力"""
    get_logger_manager().get_logger(module_name).warning(message)

def log_error(message: str, module_name: str = "app", exc_info=False):
    """ERRORレベルのログを出力"""
    get_logger_manager().get_logger(module_name).error(message, exc_info=exc_info)

def log_critical(message: str, module_name: str = "app", exc_info=False):
    """CRITICALレベルのログを出力"""
    get_logger_manager().get_logger(module_name).critical(message, exc_info=exc_info)

def log_debug(message: str, module_name: str = "app"):
    """
    DEBUGレベルのログを出力
    """
    get_logger_manager().get_logger(module_name).debug(message)