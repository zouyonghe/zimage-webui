# Z-Image WebUI

Z-Image の軽量ローカル Web UI。アスペクト比プリセット、連続生成、自動保存、高解像度アップスケール、ルーペ機能を備えています。

## 機能
- Vue SPA：プロンプト / ネガティブ、ステップ、ガイダンス、シード。
- アスペクト比プリセット（512/768/1024 の正方形と一般的な比率）、16px ステップで 512–1024 にクランプ。
- 1–10 枚の連続生成；シード未入力で毎回ランダム、固定シードで再現可能。
- `outputs/` に自動保存（タイムスタンプ・サイズ・シード付き）、API 返り値 `meta.saved_path` で保存パスを取得。
- Real-ESRGAN による高解像度アップスケール（1–4x）、自動アップスケールのトグルあり。結果とメタデータを履歴に記録。
- ルーペ（放大鏡）トグルはデフォルト無効（高解像度時の負荷軽減）。
- UI 言語切替：zh / en / ja。

## 必要環境
- Python 3.10+
- CUDA GPU（対応する PyTorch ビルド）
- `zimage-model/` に配置した Z-Image 重み（ローカル読み込みのみ）
- 任意：Real-ESRGAN 重み `weights/RealESRGAN_x4plus.pth`（環境変数 `ZIMAGE_UPSCALE_MODEL` で上書き可）

## インストール
```bash
# CUDA に合った torch/torchvision をインストール（例: CUDA 12.1）
# pip install torch==2.5.1+cu121 torchvision==0.20.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

# 重みをダウンロード（aria2c が必要）
cd scripts && bash download_models.sh && cd ..
```

## 実行
```bash
python webui_server.py
# デフォルト 0.0.0.0:9000、環境変数 ZIMAGE_PORT で変更可
```

ブラウザで `http://localhost:9000` を開いてください。

生成画像は `outputs/` に保存され、ファイル名は `20240614_153045_768x768_rand.png` のようになります。

## 補足
- 解像度制限：最小 512 / 最大 1024 / ステップ 16。変更する場合は `webui/index.html` と `webui_server.py` を両方修正してください。
- Real-ESRGAN インストール時の `tb-nightly` 警告は無視可能。静かにしたい場合は `tensorboard` を入れてから `pip install realesrgan --no-deps` を実行。実行環境で `import realesrgan` できれば問題ありません。
