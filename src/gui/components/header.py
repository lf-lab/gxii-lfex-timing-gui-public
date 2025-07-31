import streamlit as st
from src.config.settings import settings, get_config

def render_header():
    """ヘッダーを描画（ConfigManagerベース）"""
    st.title(f"{get_config('app.name', 'GXII-LFEX Timing Analysis GUI')} v{settings.get_version()}")
    st.markdown(f"*{get_config('app.description', 'GXII-LFEX実験のタイミング解析GUI')}*")
    st.markdown("---")
