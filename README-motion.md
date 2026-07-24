# GoPro + MediaPipe 動作解析

`gopro_motion_analysis.py` は、GoProのRGB動画から次を記録します。

- 頭揺れ: 肩中心に対する頭部中心の移動（肩幅で正規化）
- 上体前傾: MediaPipeの3D world landmarksによる肩中心—腰中心の角度
- 拍同期: 頭・上体の動作ピークと音楽の拍との時間差、100 ms以内の割合、0〜1の同期スコア

フレーム別データは `motion_metrics.csv`、全体集計は
`motion_summary.json` に保存されます。

## 1. インストール

MediaPipeを利用できるPython 3.11環境を推奨します。現在このMacの
`python3` は3.14なので、別の仮想環境を作成してください。

```bash
python3.11 -m venv .venv-motion
source .venv-motion/bin/activate
python -m pip install -r requirements-motion.txt
```

初回起動時は、Google公式のPose Landmarker Fullモデルを
`models/pose_landmarker_full.task` にダウンロードします。手動配置したモデルは
`--model` で指定できます。

## 2. GoPro入力

GoProを三脚に固定し、可能ならレンズを `Linear`、解像度を1080p、フレーム
レートを30または60 fpsにします。HyperSmoothやカメラの移動は、身体の動きと
映像の動きを混同するため避けてください。頭から腰まで常に画角に入れます。

### USBウェブカメラ（カメラ番号0）

GoProをWebcamモードにしてから実行します。別のカメラとして認識された場合は
`--source 1`、`--source 2` のように番号を変更します。

```bash
python gopro_motion_analysis.py --source 0 --view front --bpm 120
```

60秒だけ収集する例:

```bash
python gopro_motion_analysis.py --source 0 --duration 60 --bpm 120
```

Open GoProで開始したRTSP/UDPストリームもURLをそのまま指定できます。

```bash
python gopro_motion_analysis.py --source "rtsp://GoProのURL"
```

### 保存済みGoPro MP4

```bash
python gopro_motion_analysis.py \
  --source GX010001.MP4 \
  --music GX010001.MP4 \
  --view front \
  --no-preview
```

環境によってMP4内の音声を`librosa`が直接読めない場合は、音声をWAVに変換し、
`--music music.wav` として指定します。

## 3. 拍の指定方法

次の3方式のうち1つだけを指定します。

- `--music FILE`: 音声から拍とテンポを自動検出（保存動画向け）
- `--bpm 120 --beat-offset 0.25`: 既知BPMと第1拍の時刻（ライブ向け）
- `--beat-times beats.csv`: 1列目に拍時刻（秒）を並べたCSV（最も正確）

GoProのウェブカメラ映像はOpenCVから音声を取得できないため、ライブ解析では
既知BPMか、DAW等から書き出した拍時刻CSVを使います。音楽再生開始と映像開始の
ずれを`--beat-offset`で補正してください。

## 4. カメラ方向と値の解釈

- `--view front`: 被写体がカメラを向く。カメラ方向への前傾を正とする
- `--view side-right`: 横向きで、被写体が画面右を向く
- `--view side-left`: 横向きで、被写体が画面左を向く

前傾判定の既定値は15度です。変更例は `--forward-threshold 10` です。
開始後2秒間は自然な基準姿勢を保ってください。この区間の頭位置中央値が、
頭揺れの基準になります。

主な集計値:

- `head_sway_rms_shoulder_width`: 頭揺れのRMS。0.1なら肩幅の10%
- `forward_lean_max_deg`: 最大前傾角
- `forward_lean_fraction`: 前傾しきい値を超えた時間割合
- `mean_abs_beat_error_ms`: 動作ピークから最寄りの拍までの平均絶対時間差
- `on_beat_fraction_100ms`: 拍の前後100 ms以内に入った動作ピークの割合
- `beat_sync_score`: 0〜1。1に近いほど拍に同期

単眼RGBによる3D奥行きは推定値です。研究用途では同じカメラ位置・画角・照明・
被写体距離を維持し、数回の既知動作で前傾角と同期スコアを校正してください。
