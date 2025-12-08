# Z-Image WebUI

[中文](README.md) | [English](README_en.md) | [日本語](README_jp.md)

アスペクト比プリセット、連続生成、自動保存、高解像度アップスケール、ルーペ機能を備えた Z-Image の軽量ローカル Web UI です。

## 機能概要
- Vue SPA：プロンプト / ネガティブ、ステップ、ガイダンス、シード。
- 言語切替（中文 / English / 日本語）。
- アスペクト比プリセット（512/768/1024 の正方形＋一般的な比率）、入力は 16px ステップで 512–1024 にクランプ。
- 1–10 枚の連続生成；シード未入力で毎回ランダム、固定シードで再現可能。
- `outputs/` に自動保存（タイムスタンプ・サイズ・シード付き）、API 返り値 `meta.saved_path` で保存パス取得。
- 初回ロード時にランダムなプロンプトを自動入力。
- Real-ESRGAN による高解像度アップスケール（1–4x）、プレビュー内の「高分放大」ボタンで実行、後続生成に自動適用するオプションあり。結果とメタデータを履歴に記録。
- ルーペ（放大鏡）トグルはプレビュー内に配置、デフォルト無効で高解像度時の負荷を回避。

## 必要環境
- Python 3.10+
- CUDA GPU（対応する PyTorch ビルド）
- `zimage-model/` に配置した Z-Image 重み（ローカル読み込みのみ）
- 任意: Real-ESRGAN 重み `weights/RealESRGAN_x4plus.pth`（環境変数 `ZIMAGE_UPSCALE_MODEL` で上書き可）

## インストール
```bash
# CUDA に合った torch/torchvision をインストール（例: CUDA 12.1）
# pip install torch==2.5.1+cu121 torchvision==0.20.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

# 重みをダウンロード（aria2c が必要）: メインモデルは zimage-model/、RealESRGAN_x4plus は weights/ に保存
cd scripts && bash download_models.sh && cd ..
```

## 実行
```bash
python webui_server.py
# デフォルト 0.0.0.0:9000、環境変数 ZIMAGE_PORT で変更可

# コマンドラインで即生成（ローカルモデル使用）
python zimage.py                     # デフォルトプロンプト
python zimage.py "a scenic mountain" # 任意のプロンプト
```

ブラウザで `http://localhost:9000` を開いてください。

生成画像は `outputs/` に `20240614_153045_768x768_rand.png` のようなファイル名で保存されます。

## ディレクトリ
- `webui/`: フロントエンド（Vue 3 ESM）
- `webui_server.py`: HTTP サーバーと生成 API
- `zimage-model/`: モデル重み
- `outputs/`: 自動保存先（実行時に作成）
- その他: `requirements.txt`、補助/テストスクリプト
- ダウンロードスクリプト: `scripts/download_models.sh`（hf-mirror を利用）
- CUDA テスト: `scripts/test_cuda.py`

## 補足
- 解像度制限: 最小 512 / 最大 1024 / ステップ 16。変更する場合は `webui/index.html` と `webui_server.py` の両方を修正してください。
- サーバーログに生成パラメータとエラーが出るのでデバッグしやすいです。
- Real-ESRGAN インストール時の `tb-nightly` 警告は無視可能。静かにしたい場合は `tensorboard` を入れてから `pip install realesrgan --no-deps` を実行。実行環境で `import realesrgan` できれば問題ありません。
- 高解像度アップスケール結果は結果リストと履歴に追加され、倍率が表示されます。「自動アップスケール」はデフォルト無効。
- ルーペのオン/オフはプレビュー操作バーにあります。オフのままなら描画されず、パフォーマンスに影響しません。

## ライセンス
本プロジェクトは MIT ライセンスで配布しています。詳細は `LICENSE` を参照してください。
