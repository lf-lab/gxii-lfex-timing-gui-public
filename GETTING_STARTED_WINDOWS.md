# 🚀 GXII-LFEX Timing Analysis GUI - Windows スタートガイド v1.6.0

**Windows初めての方向け完全ガイド** - zipファイルの解凍から実際の使用まで

---

## 📦 STEP 1: ファイルの準備

### 1.1 ダウンロードしたファイルの確認
ダウンロードしたファイルを確認してください：
```cmd
dir gxii-lfex-timing-gui-v1.6.0*.tar.gz
```

通常、以下のようなファイル名になります：
- `gxii-lfex-timing-gui-v1.6.0.tar.gz`
- `gxii-lfex-timing-gui-v1.6.0-windows.tar.gz`

### 1.2 適切な場所への移動
作業しやすい場所（例：デスクトップまたはドキュメント）に移動：
```cmd
REM デスクトップの場合
cd %USERPROFILE%\Desktop

REM ドキュメントの場合
cd %USERPROFILE%\Documents

REM または専用フォルダを作成
mkdir %USERPROFILE%\Documents\gxii-lfex-workspace
cd %USERPROFILE%\Documents\gxii-lfex-workspace
```

---

## 🔓 STEP 2: アーカイブの解凍

### 2.1 解凍方法（いずれかを選択）

#### 方法A: Windows標準機能を使用
1. tarファイルを右クリック
2. 「すべて展開...」を選択
3. 展開先を選択して「展開」をクリック

#### 方法B: コマンドプロンプトを使用
```cmd
REM ファイル名は実際のものに置き換えてください
tar -xzf gxii-lfex-timing-gui-v1.6.0.tar.gz
```

#### 方法C: PowerShellを使用
```powershell
# PowerShellの場合
Expand-Archive -Path gxii-lfex-timing-gui-v1.6.0.tar.gz -DestinationPath .
```

### 2.2 解凍結果の確認
```cmd
REM 解凍されたフォルダを確認
dir
REM LFEXTimingフォルダが作成されているはずです

REM フォルダの内容を確認
dir LFEXTiming
```

### 2.3 プロジェクトディレクトリに移動
```cmd
cd LFEXTiming
```

---

## 🔧 STEP 3: 実行環境の確認

### 3.1 Python環境の確認
```cmd
REM Pythonがインストールされているか確認
python --version

REM または
python3 --version

REM pipが利用可能か確認
python -m pip --version
```

**注意**: Pythonがインストールされていない場合は、[Python公式サイト](https://www.python.org/downloads/)からダウンロードしてインストールしてください。

### 3.2 必要なファイルの確認
```cmd
REM 重要なファイルが存在するか確認
dir start_app.bat
dir cleanup_processes.bat
dir main.py
dir requirements.txt
```

---

## 🚀 STEP 4: アプリケーションの起動

### 4.1 最初の起動

#### コマンドプロンプトの場合：
```cmd
start_app.bat
```

#### PowerShellまたはGit Bashの場合：
```bash
./start_app.sh
```

**期待される動作**：
1. 仮想環境の自動設定（初回は時間がかかる場合があります）
2. 必要なパッケージの自動インストール
3. Streamlitサーバーの起動
4. ブラウザの自動オープン

### 4.2 正常起動の確認
以下のメッセージが表示されれば成功です：
```
🚀 GXII-LFEX Timing Analysis GUI
=================================
🔍 Environment check...
✅ Created venv virtual environment
🔄 Activating venv environment...
📦 Installing dependencies...
✅ Dependencies installation complete
✅ Streamlit is available
🚀 Starting GUI...
💡 Access via browser: http://localhost:8502

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8502
  Network URL: http://192.168.x.x:8502
```

### 4.3 ブラウザでのアクセス
- ブラウザが自動で開かない場合：`http://localhost:8502` を手動で開く
- ポートが異なる場合：起動時に表示されたURLを使用

---

## 🎯 STEP 5: 基本的な使い方

### 5.1 サンプルデータでテスト
1. **ファイルアップロード**：
   - 左サイドバーの「📁 データファイルの選択」
   - `testdata\` フォルダ内のサンプルファイルを選択

2. **領域設定**：
   - GXII領域とLFEX領域をスライダーで調整
   - プリセット設定の利用も可能

3. **解析実行**：
   - パラメーターを確認
   - 「解析実行」ボタンをクリック

4. **結果確認**：
   - ヒートマップの表示
   - タイミング差の計算結果

---

## 🔧 STEP 6: Windows固有のトラブルシューティング

### 6.1 起動時の問題

#### 問題: "Python not found" エラー
**解決策**：
```cmd
REM Python環境の確認
where python
where python3

REM 環境変数PATHの確認
echo %PATH%
```

**対処法**：
1. Python公式サイトからPythonをダウンロード・インストール
2. インストール時に「Add Python to PATH」を必ずチェック
3. コマンドプロンプトを再起動

#### 問題: "Access denied" エラー
**解決策**：
```cmd
REM 管理者権限でコマンドプロンプトを開く
REM スタートメニュー → cmd → 右クリック → 管理者として実行

REM または、ユーザーフォルダに移動
cd %USERPROFILE%\Documents\gxii-lfex-workspace\LFEXTiming
start_app.bat
```

#### 問題: Windows Defenderによるブロック
**解決策**：
1. Windows Defender セキュリティセンターを開く
2. 「ウイルスと脅威の防止」→「除外」
3. LFEXTimingフォルダを除外リストに追加

### 6.2 ポート関連の問題

#### 問題: ポート8502が使用中
**解決策**：
```cmd
REM 自動クリーンアップ
start_app.bat --cleanup

REM または詳細なクリーンアップ
cleanup_processes.bat

REM 手動でポート確認
netstat -ano | findstr :8502
```

#### 問題: プロセスが残っている
**解決策**：
```cmd
REM 状態確認
start_app.bat --status

REM 完全クリーンアップ
cleanup_processes.bat

REM 再起動
start_app.bat
```

### 6.3 ファイアウォール関連の問題

#### 問題: ブラウザでアクセスできない
**解決策**：
1. **Windows ファイアウォール設定**：
   ```cmd
   REM ファイアウォール設定を確認
   netsh advfirewall show allprofiles
   ```

2. **手動でファイアウォール例外を追加**：
   - コントロールパネル → システムとセキュリティ → Windows Defender ファイアウォール
   - 「アプリまたは機能をWindows Defender ファイアウォール経由で許可」
   - Pythonを許可リストに追加

3. **一時的な解決（テスト用）**：
   ```cmd
   REM 別のポートで起動
   python -m streamlit run main.py --server.port=8503
   ```

### 6.4 依存関係の問題

#### 問題: NumPy 2.0互換性エラー
**解決策**：
```cmd
REM 仮想環境をアクティベート
venv\Scripts\activate.bat

REM NumPyのダウングレード
pip install "numpy>=1.21.0,<2.0.0"

REM 仮想環境を無効化
deactivate

REM 再起動
start_app.bat
```

#### 問題: その他のパッケージエラー
**解決策**：
```cmd
REM 仮想環境の削除
rmdir /s venv

REM 再作成（start_app.batが自動で作成）
start_app.bat
```

---

## 🔧 STEP 7: 便利なコマンドとショートカット

### 7.1 利用可能なコマンド
```cmd
REM ヘルプ表示
start_app.bat --help

REM 状態確認
start_app.bat --status

REM プロセスクリーンアップ
start_app.bat --cleanup

REM 専用クリーンアップツール
cleanup_processes.bat
```

### 7.2 デスクトップショートカット作成
1. `start_app.bat`を右クリック
2. 「ショートカットの作成」を選択
3. 作成されたショートカットをデスクトップに移動
4. 次回からはダブルクリックで起動可能

### 7.3 バッチファイルのカスタマイズ
高度な設定が必要な場合：
```cmd
REM start_app.batをテキストエディタで開く
notepad start_app.bat

REM ポート番号の変更例（8502 → 8503）
REM set PORT=8503
```

---

## 📖 STEP 8: 詳細情報とヘルプ

### 8.1 詳細ドキュメント
- **基本的な使い方**: `QUICKSTART.md`
- **詳細セットアップ**: `README.md`
- **Windows対応情報**: `docs\WINDOWS_CROSS_PLATFORM_SUPPORT.md`
- **リリース情報**: リリースノート

### 8.2 プロジェクト構造（Windows）
主要なファイルとディレクトリ：
```
LFEXTiming\
├── start_app.bat           # Windows起動スクリプト
├── cleanup_processes.bat   # Windowsプロセス管理ツール
├── start_app.sh           # Unix/Git Bash用起動スクリプト
├── main.py               # GUIアプリケーション本体
├── requirements.txt      # Python依存関係
├── testdata\            # サンプルデータ
├── docs\               # 詳細ドキュメント
├── scripts\            # ツールスクリプト
│   └── system_check.bat   # Windowsシステム診断
└── legacy\             # 下位互換サポート
```

### 8.3 システム要件
- **OS**: Windows 10/11 (64-bit推奨)
- **Python**: 3.8以上 (3.9-3.11推奨)
- **RAM**: 4GB以上
- **ストレージ**: 1GB以上の空き容量
- **ブラウザ**: Chrome, Firefox, Edge (最新版)

---

## ✅ STEP 9: 成功確認のチェックリスト

以下が全て完了していれば、正常にセットアップできています：

- [ ] tarファイルが正常に解凍された
- [ ] `LFEXTiming` フォルダに移動した
- [ ] Pythonが正しくインストールされている
- [ ] `start_app.bat` が正常に実行された
- [ ] ブラウザで `http://localhost:8502` にアクセスできる
- [ ] GUIが表示されている
- [ ] サンプルデータが読み込める

---

## 🆘 Windows固有のよくある質問

### Q: 文字化けが発生する
**A**: コマンドプロンプトの文字コードを確認：
```cmd
REM 文字コードをUTF-8に設定
chcp 65001

REM 再起動
start_app.bat
```

### Q: "システムによって無効化されました" エラー
**A**: PowerShell実行ポリシーの変更：
```powershell
# PowerShellを管理者として実行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q: 日本語ファイル名のデータが読み込めない
**A**: ファイル名を英数字に変更するか、UTF-8対応エディタで保存し直してください。

### Q: アンチウイルスソフトが誤検知する
**A**: LFEXTimingフォルダを除外リストに追加してください。

### Q: ネットワークエラーが発生する
**A**: 企業ネットワークの場合、IT部門にStreamlitポート（8502）の許可を依頼してください。

---

## 🎯 STEP 10: 追加のWindows最適化

### 10.1 パフォーマンス向上
```cmd
REM 高性能モードに設定（ノートPCの場合）
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

REM 仮想メモリの最適化（必要に応じて）
REM システム → 詳細設定 → パフォーマンス → 設定
```

### 10.2 自動起動設定
1. `win + R` → `shell:startup`
2. LFEXTimingショートカットをStartupフォルダにコピー

### 10.3 バックアップ作成
```cmd
REM プロジェクト全体のバックアップ
xcopy LFEXTiming LFEXTiming_backup /E /I /H /Y
```

---

## 🎉 Windows版セットアップ完了！

**おめでとうございます！** GXII-LFEX Timing Analysis GUI v1.6.0 Windows版が使用可能になりました。

今後は簡単に起動できます：
```cmd
cd %USERPROFILE%\Documents\gxii-lfex-workspace\LFEXTiming
start_app.bat
```

**または、デスクトップショートカットをダブルクリック！**

**快適なWindows環境での解析作業を！** 🚀

---

## 📞 Windows版サポート情報

### トラブル時の連絡先
- Windows固有の問題: `docs\WINDOWS_CROSS_PLATFORM_SUPPORT.md`
- 一般的な問題: `README.md`
- システム診断: `scripts\system_check.bat`

### 有用なWindowsコマンド
```cmd
REM システム情報確認
systeminfo

REM ネットワーク設定確認
ipconfig /all

REM プロセス確認
tasklist | findstr python

REM ポート使用状況
netstat -ano | findstr :8502
```

---

*GXII-LFEX Timing Analysis GUI v1.6.0 - Windows Cross-Platform Edition*  
*作成日: 2025年5月29日*  
*対応OS: Windows 10/11*
