import streamlit as st
import pandas as pd

def render_log_display_panel(logger_manager):
    """ログ表示パネルを描画"""
    st.header("📋 アプリケーションログ")

    # ログレベル選択
    col_level, col_clear = st.columns([3, 1])
    with col_level:
        log_level_filter = st.selectbox(
            "表示レベル",
            ["ALL", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=0,
            key="log_level_filter"
        )

    with col_clear:
        if st.button("🗑️", help="ログをクリア", key="clear_logs_button"):
            logger_manager.clear_gui_logs()
            st.rerun()

    # ログ表示件数
    log_limit = st.slider("表示件数", min_value=10, max_value=100, value=30, key="log_limit_slider")

    # ログ取得
    level_filter = None if log_level_filter == "ALL" else log_level_filter
    try:
        logs = logger_manager.get_gui_logs(level_filter=level_filter, limit=log_limit)

        if logs:
            # ログコンテナ
            log_container = st.container()
            with log_container:
                # ログ統計情報
                with st.expander("📊 ログ統計", expanded=False):
                    stats = logger_manager.get_log_statistics()
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("総ログ数", stats.get('total', 0))
                    with col_stat2:
                        error_count = stats.get('level_distribution', {}).get('ERROR', 0)
                        st.metric("エラー数", error_count, delta_color="inverse" if error_count > 0 else "normal")

                # ログ表示
                st.subheader("📝 最新ログ")

                # ログレコードを逆順で表示（新しいものが上）
                for i, log in enumerate(reversed(logs)):
                    # レベルに応じて色分け
                    if log["level"] == "ERROR":
                        st.error(f"**{pd.Timestamp(log["timestamp"], unit='s').strftime('%H:%M:%S')}** [{log["module"]}] {log["message"]}")
                    elif log["level"] == "WARNING":
                        st.warning(f"**{pd.Timestamp(log["timestamp"], unit='s').strftime('%H:%M:%S')}** [{log["module"]}] {log["message"]}")
                    elif log["level"] == "CRITICAL":
                        st.error(f"🚨 **{pd.Timestamp(log["timestamp"], unit='s').strftime('%H:%M:%S')}** [{log["module"]}] {log["message"]}")
                    else:
                        st.info(f"**{pd.Timestamp(log["timestamp"], unit='s').strftime('%H:%M:%S')}** [{log["module"]}] {log["message"]}")

                    # 例外情報がある場合は詳細表示
                    if log["exc_info"]:
                        with st.expander("🔍 例外詳細"):
                            st.code(log["exc_info"], language="python")

            # ログエクスポート
            with st.expander("💾 ログエクスポート"):
                if st.button("📄 JSON形式でエクスポート"):
                    json_data = logger_manager.export_logs_json(limit=log_limit)
                    st.download_button(
                        label="📥 JSONファイルをダウンロード",
                        data=json_data,
                        file_name=f"app_logs_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        else:
            st.info("📝 ログがありません")
            st.caption("アプリケーションの動作に応じてここにログが表示されます")

    except Exception as e:
        st.error(f"❌ ログ表示エラー: {str(e)}")
        # log_error(f"ログ表示エラー: {str(e)}", "gui.app", exc_info=True) # 循環参照になるためコメントアウト
