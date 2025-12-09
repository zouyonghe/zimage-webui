# Z-Image WebUI

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-CUDA%20GPU-green.svg)](https://developer.nvidia.com/cuda-zone)

[中文](README.md) | [English](README_en.md) | [日本語](README_jp.md)

**ローカルモデルベースの軽量AI画像生成Webインターフェース**

</div>

## 📖 プロジェクト概要

Z-Image WebUIはローカルAIモデルベースの軽量画像生成インターフェースで、直感的なWeb操作体験を提供します。インターネット接続不要で完全にローカルで実行され、創作のプライバシーを保護します。

### ✨ コア機能

- 🎨 **直感的なWebインターフェース** - Vue 3ベースのモダンなシングルページアプリケーション
- 🌍 **多言語サポート** - 中国語、英語、日本語のインターフェース切り替え内蔵
- 🖼️ **スマートアスペクトプリセット** - 一般的な解像度比率をサポート、16ピクセルステップに自動整列
- ⚡ **バッチ生成** - 1クリックで1-10枚の画像を生成、ランダムまたは固定シードをサポート
- 💾 **自動保存** - 生成結果を完全なメタデータ付きでローカルに自動保存
- 🔍 **HDアップスケーリング** - Real-ESRGAN超解像度技術内蔵、1-4倍拡大をサポート
- 🔎 **拡大鏡機能** - プレビュー時の詳細表示、パフォーマンスを節約
- 🎯 **すぐに使える** - 初回ロード時にサンプルプロンプトを自動入力

## 🚀 クイックスタート

### 動作環境

- **Python**: 3.10以上
- **GPU**: CUDA対応NVIDIAグラフィックカード
- **メモリ**: 推奨8GB以上VRAM
- **システム**: Linux / Windows / macOS

### インストール手順

1. **プロジェクトをクローン**
   ```bash
   git clone https://github.com/zouyonghe/zimage-webui.git
   cd zimage-webui
   ```

2. **PyTorchをインストール**（CUDAバージョンに応じて選択）
   ```bash
   # CUDA 12.1の例
   pip install torch==2.5.1+cu121 torchvision==0.20.1 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. **プロジェクト依存関係をインストール**
   ```bash
   pip install -r requirements.txt
   ```

4. **モデル重みをダウンロード**
   ```bash
   cd scripts && bash download_models.sh && cd ..
   ```

### サービス起動

```bash
python webui_server.py
```

サービスはデフォルトで `http://localhost:9000` で実行されます。環境変数 `ZIMAGE_PORT` でポートを変更できます。

### コマンドライン使用

```bash
# デフォルトプロンプトを使用
python zimage.py

# カスタムプロンプトを使用
python zimage.py "美しい山岳風景"
```

## 📁 プロジェクト構造

```
zimage-webui/
├── webui/                    # フロントエンドリソース
│   ├── index.html           # メインページ（Vue 3 SPA）
│   └── favicon-*.png        # アイコンファイル
├── webui_server.py          # WebサーバーとAPI
├── zimage.py               # コマンドラインツール
├── zimage-model/           # AIモデル重みディレクトリ
├── weights/                # アップスケーリングモデル重み
├── outputs/                # 生成結果保存ディレクトリ
├── scripts/                # ヘルパースクリプト
│   └── download_models.sh  # モデルダウンロードスクリプト
└── requirements.txt        # Python依存関係
```

## 🎯 機能詳細

### 画像生成パラメータ

| パラメータ | 説明 | 範囲 |
|-----------|-------------|-------|
| プロンプト | 生成したい内容を記述 | 任意のテキスト |
| ネガティブ | 望ましくない内容を記述 | 任意のテキスト |
| ステップ数 | 生成品質を制御 | 1-50 |
| 誘導強度 | プロンプトへの従順度を制御 | 1.0-20.0 |
| シード | 生成のランダム性を制御 | 任意の整数または空欄 |

### アスペクトプリセット

- **正方形**: 512×512, 768×768, 1024×1024
- **横向き**: 768×512, 1024×768, 1024×576
- **縦向き**: 512×768, 768×1024, 576×1024
- **ワイドスクリーン**: 1024×512, 1152×648
- **垂直**: 512×1024, 648×1152

### HDアップスケーリング

- **拡大倍率**: 1x, 2x, 3x, 4x
- **モデル**: Real-ESRGAN_x4plus
- **自動アップスケール**: 新しく生成された画像の自動アップスケーリングオプション
- **品質最適化**: ディテールを保持しながら解像度を向上

## ⚙️ 設定

### 環境変数

| 変数名 | デフォルト値 | 説明 |
|----------|---------|-------------|
| `ZIMAGE_PORT` | 9000 | Webサービスポート |
| `ZIMAGE_UPSCALE_MODEL` | weights/RealESRGAN_x4plus.pth | アップスケーリングモデルパス |

### カスタム設定

解像度制限を変更する場合（フロントエンドとバックエンドの両方を変更が必要）：
- フロントエンド設定: `webui/index.html`
- バックエンド設定: `webui_server.py`

デフォルト制限：
- 最小解像度: 512×512
- 最大解像度: 1024×1024
- ステップ: 16ピクセル

## 🔧 トラブルシューティング

### 一般的な問題

**Q: CUDAが利用できない場合**
```bash
# CUDAインストールを確認
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: Real-ESRGANインストール警告**
```bash
# 警告は無視できます。警告を消去するにはtensorboardをインストール
pip install tensorboard
pip install realesrgan --no-deps
```

**Q: VRAMが不足している場合**
- 生成解像度を下げる
- バッチ生成数を減らす
- 自動アップスケーリングを無効にする

**Q: モデルダウンロードに失敗した場合**
```bash
# aria2cがインストールされているか確認
aria2c --version

# 手動でモデルを対応するディレクトリにダウンロード
```

### パフォーマンス最適化

1. **メモリ最適化**
   - xformersメモリ効率の良い注意を有効にする
   - 拡張可能メモリモードを使用
   - 注意スライシングを有効にする

2. **生成速度最適化**
   - 適切な精度を使用（BF16/FP16）
   - 生成ステップ数を調整
   - 合理的なバッチサイズを設定

## 📊 生成結果

### ファイル命名形式

生成された画像は `outputs/` ディレクトリに自動保存され、ファイル名形式：
```
{タイムスタンプ}_{幅}x{高さ}_{シード}.png
```

例: `20240614_153045_768x768_rand.png`

### メタデータ情報

各画像には完全な生成パラメータが含まれます：
- プロンプトとネガティブプロンプト
- 生成パラメータ（ステップ、誘導、シード）
- 生成タイムスタンプ
- 拡大倍率（該当する場合）

## 🤝 貢献

Issueやプルリクエストの投稿を歓迎します！

### 開発環境セットアップ

```bash
# プロジェクトをクローン
git clone https://github.com/zouyonghe/zimage-webui.git
cd zimage-webui

# 開発依存関係をインストール
pip install -r requirements.txt

# テストを実行
python scripts/test_cuda.py
```

### コード規約

- PEP 8 Pythonコード規約に従う
- 意味的なGitコミットメッセージを使用
- 新機能には適切なドキュメントを追加

## 📄 ライセンス

このプロジェクトは[MITライセンス](LICENSE)の下で公開されています。

## 🙏 謝辞

- [Diffusers](https://github.com/huggingface/diffusers) - 強力な拡散モデルライブラリ
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - 優れた超解像度モデル
- [Vue.js](https://vuejs.org/) - モダンなフロントエンドフレームワーク
- [Element Plus](https://element-plus.org/) - 優れたVue 3コンポーネントライブラリ

## 📞 お問い合わせ

質問や提案がある場合は、以下の方法でお問い合わせください：

- [GitHub Issue](https://github.com/zouyonghe/zimage-webui/issues)を投稿
- プロジェクトホームページ: [https://github.com/zouyonghe/zimage-webui](https://github.com/zouyonghe/zimage-webui)

---

<div align="center">

**⭐ このプロジェクトが役立った場合は、Starをください！**

</div>