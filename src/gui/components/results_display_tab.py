import streamlit as st
import numpy as np

def render_results_display_tab(app_instance):
    """結果表示タブを描画"""
    st.header("📈 解析結果")

    # エンディアン設定が変更された場合のチェック
    if hasattr(st.session_state, 'img_byte_order_changed') and st.session_state.img_byte_order_changed:
        st.warning("⚠️ エンディアン設定が変更されました。データプレビュータブでファイルを再読み込みしてから解析を再実行してください。")
        return

    if st.session_state.analysis_results is None:
        st.warning("⚠️ 先に解析を実行してください")
        return

    results = st.session_state.analysis_results

    # 結果サマリー
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔴 GXII ピーク", f"{results['gxii_peak_relative']:.3f} ns")

    with col2:
        st.metric("🔵 LFEX ピーク 1", f"{results['lfex_peak_1_relative']:.3f} ns")

    with col3:
        if results['max_value_2'] > 0:
            st.metric("🔵 LFEX ピーク 2", f"{results['lfex_peak_2_relative']:.3f} ns")
        else:
            st.metric("🔵 LFEX ピーク 2", "N/A")

    with col4:
        # タイミング差の意味を基準時間モードに応じて表示
        if results['reference_time_mode'] == 'gxii_rise':
            timing_label = "⏱️ 立ち上がり→LFEX差"
            timing_help = f"GXII立ち上がり({results['gxii_rise_percentage']:.1f}%)からLFEXピークまでの時間差"
        elif results['reference_time_mode'] == 'lfex_peak':
            timing_label = "⏱️ GXII→LFEX差"
            timing_help = "GXIIピークからLFEXピークまでの時間差（LFEX基準）"
        else:
            timing_label = "⏱️ ピーク間差"
            timing_help = "GXIIピークとLFEXピークの時間差"

        st.metric(timing_label, f"{results['time_diff']:.3f} ns", help=timing_help)

    # 詳細情報
    with st.expander("📊 詳細解析情報"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**GXII 解析結果:**")
            st.write(f"- ピーク位置: {results['gxii_peak_relative']:.3f} ns")
            st.write(f"- 立ち上がり時間: {results['gxii_rise_time']:.3f} ns")
            st.write(f"- フィッティングパラメータ: {results['fitting_params']}")

        with col2:
            st.write("**LFEX 解析結果:**")
            st.write(f"- 第1ピーク: {results['lfex_peak_1_relative']:.3f} ns (強度: {results['max_value_1']:.2f})")
            if results['max_value_2'] > 0:
                st.write(f"- 第2ピーク: {results['lfex_peak_2_relative']:.3f} ns (強度: {results['max_value_2']:.2f})")
            st.write(f"- ショットID: {results['shotid']}")

        with col3:
            st.write("**基準時間・校正設定:**")
            # 基準時間情報を追加
            reference_mode_labels = {
                "gxii_peak": "GXIIピークタイミング",
                "streak_time": "ストリーク画像時間（t=0基準）",
                "gxii_rise": "GXIIの立ち上がり（n%）",
                "custom_t0": "カスタム波形t0基準"
            }
            st.write(f"- 基準時間モード: {reference_mode_labels.get(results['reference_time_mode'], results['reference_time_mode'])}")
            st.write(f"- 基準時間: {results['reference_time']:.3f} ns")
            if results['reference_time_mode'] == 'gxii_rise':
                st.write(f"- 立ち上がり閾値: {results['gxii_rise_percentage']:.1f}%")
            st.write(f"- 校正モード: {results['time_calibration_mode']}")
            st.write(f"- 全幅時間: {results['full_width_time']:.3f} ns")
            st.write(f"- 1pixel時間: {results['time_per_pixel']:.6f} ns/pixel")

        col4_dummy, col5_dummy, col6_dummy = st.columns(3) # 3列目を空にするためのダミー
        with col4_dummy:
            st.write("**波形フィッティング設定:**")
            # 波形タイプの表示
            waveform_labels = {
                "gaussian": "ガウシアン",
                "custom_pulse": "カスタムパルス",
                "custom_file": "カスタムファイル"
            }
            wf_type = results.get('actual_waveform_type', results.get('waveform_type'))
            wf_label = waveform_labels.get(wf_type, wf_type)
            if results.get('waveform_name'):
                wf_label += f" ({results['waveform_name']})"
            st.write(f"- 波形タイプ: {wf_label}")

            # フィッティング成功/失敗の表示
            if results.get('fitting_success', False):
                st.write("- フィッティング: ✅ 成功")
            else:
                st.write("- フィッティング: ❌ 失敗 (最大値検出)")
            if results.get('waveform_r_squared') is not None:
                st.write(f"- R²: {results['waveform_r_squared']:.4f}")

            # フィッティングパラメータ
            if results.get('fitting_params'):
                params = results['fitting_params']
                if len(params) >= 2:
                    st.write(f"- 振幅: {params[0]:.3f}")
                    st.write(f"- ピーク位置: {params[1]:.3f} ns")
                    if len(params) > 2 and wf_type == 'gaussian':
                        st.write(f"- パラメータ3: {params[2]:.3f}")

    # プロット表示
    st.subheader("📊 解析結果プロット")

    try:
        # XSC対応の3つのプロットを作成（PDF生成と同じ構成）
        figs = app_instance.plot_manager.create_xsc_result_display_plots(
            results,
            app_instance.timing_analyzer.waveform_library,
        )

        # プロットタイトルを追加
        plot_titles = [
            "① Raw Data with Overlays",
            "② Vertical 2-Panel Plot", 
            "③ Space Lineout (PDF Panel 4)"
        ]

        for i, (fig, title) in enumerate(zip(figs, plot_titles)):
            st.markdown(f"**{title}**")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"プロット作成エラー: {str(e)}")
        # フォールバック：元のプロット作成機能を使用
        try:
            figs = app_instance.timing_analyzer.create_plots(results)
            for i, fig in enumerate(figs):
                st.pyplot(fig)
        except Exception as e2:
            st.error(f"フォールバックプロット作成エラー: {str(e2)}")
            _create_basic_plots(app_instance.plot_manager, results)

def _create_basic_plots(plot_manager, results):
    """基本的なプロット作成（PlotManagerを使用）"""
    # LFEXプロファイルプロット
    lfex_fig = plot_manager.create_profile_plot(
        x_data=results['streak_time'],
        y_data=results['lfex_time'],
        title='LFEX Time Profile',
        xlabel='Time (ns)',
        ylabel='Intensity',
        color=plot_manager.theme.COLORS['lfex']
    )
    st.pyplot(lfex_fig)

    # GXIIプロファイルプロット
    gxii_fig = plot_manager.create_profile_plot(
        x_data=results['streak_time'],
        y_data=results['gxii_norm'],
        title='GXII Normalized Profile',
        xlabel='Time (ns)',
        ylabel='Normalized Intensity',
        color=plot_manager.theme.COLORS['gxii']
    )
    st.pyplot(gxii_fig)
