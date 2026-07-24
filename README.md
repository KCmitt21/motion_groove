# GoPro + MediaPipe 動作解析

`gopro_motion_analysis.py` は、GoProのRGB動画から次を記録します。

- 頭揺れ: 肩中心に対する頭部中心の移動（肩幅で正規化）
- 上体前傾: MediaPipeの3D world landmarksによる肩中心—腰中心の角度
- 拍同期: 頭・上体の動作ピークと音楽の拍との時間差、100 ms以内の割合、0〜1の同期スコア

フレーム別データは `motion_metrics.csv`、全体集計は
`motion_summary.json` に保存されます。

## 1. インストール

MediaPipeとOpen GoPro SDKの両方を利用できるPython 3.11環境を使います。
現在の`.venv-motion`はPython 3.14なので、その環境は使わず、新しい環境を
作成してください。

Open GoProはPyPIで取得できる`0.22.0`に固定しています。公式ドキュメントには
`0.23.0`も掲載されていますが、PyPIにはまだ公開されていません。

```bash
brew install python@3.11
python3.11 -m venv .venv-motion311
source .venv-motion311/bin/activate
python -m pip install -r requirements-motion.txt
```

初回起動時は、Google公式のPose Landmarker Fullモデルを
`models/pose_landmarker_full.task` にダウンロードします。手動配置したモデルは
`--model` で指定できます。

## 2. GoPro入力

GoProを三脚に固定し、可能ならレンズを `Linear`、解像度を1080p、フレーム
レートを30または60 fpsにします。HyperSmoothやカメラの移動は、身体の動きと
映像の動きを混同するため避けてください。頭から腰まで常に画角に入れます。

### USB直結でリアルタイム解析（推奨）

この方法では、プログラムがOpen GoPro公式SDKを使ってGoProを検出し、
Webcamストリームの開始と終了を自動で行います。USBストリーミングは基本的に
HERO11以降が対象です。

1. GoProのファームウェアを最新版にする
2. データ通信対応のUSB-CケーブルでGoProとMacを直接つなぐ
3. GoProの電源を手動で入れる
4. macOSにネットワーク接続の確認が出たら許可する
5. GoPro Webcamアプリ、VLC、Zoomなど、GoProを使う他のアプリを終了する
6. 次のコマンドを実行する

```bash
cd /Users/keishi-mac/code/motion_groove
source .venv-motion311/bin/activate
python gopro_motion_analysis.py \
  --gopro-usb \
  --view front \
  --duration 60 \
  --bpm 120
```

`--gopro-usb`を指定した場合、`--source 0`は不要です。既定では互換性の高い
MPEG-TS/UDPを使います。対応機種ではRTSPも選択できます。

```bash
python gopro_motion_analysis.py \
  --gopro-usb \
  --gopro-protocol RTSP \
  --view front \
  --bpm 120
```

RTSPで失敗した場合は`--gopro-protocol RTSP`を外してください。

解析中の映像も保存する場合は`--record-video`を追加します。このMP4は映像のみで
音声は入りません。

```bash
python gopro_motion_analysis.py \
  --gopro-usb \
  --view front \
  --duration 60 \
  --bpm 120 \
  --record-video recordings/session01.mp4 \
  --output results/session01.csv \
  --summary results/session01.json
```

プレビュー画面で`Q`または`Esc`を押すと終了します。終了時にGoProのWebcam
ストリームを停止し、CSVとJSONを書き出します。

#### USB接続が動かないとき

- 充電専用ではなく、データ通信対応のUSB-Cケーブルを使う
- USBハブを避け、Macへ直接接続する
- GoProを一度再起動する
- GoProの「接続をリセット」後、USBをつなぎ直す
- 他のカメラ／配信アプリを終了する
- HERO13の一部ファームウェアにはWebcam APIの既知問題があるため、
  GoProのファームウェアを更新する
- 複数台接続時だけ`--gopro-id`で対象を指定する

接続だけを公式デモで確認する場合:

```bash
gopro-webcam
```

このデモが映るのに解析プログラムが映らない場合は、両方を同時起動せず、
`gopro-webcam`を終了してから解析プログラムを起動してください。

### 仮想Webcamまたは既存ストリームを使う場合

GoPro Webcam UtilityなどによってGoProがmacOSのカメラとして既に認識されて
いる場合は、従来どおりカメラ番号を指定できます。

```bash
python gopro_motion_analysis.py --source 0 --view front --bpm 120
```

別のカメラとして認識された場合は`--source 1`、`--source 2`を試します。
すでに開始済みのRTSP/UDP URLも`--source`へ指定できます。

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

GoProのUSB Webcam映像はこのプログラムから音声を取得しないため、ライブ解析では
既知BPMか、DAW等から書き出した拍時刻CSVを使います。`--bpm`方式では、解析開始後
の最初の拍の時刻を`--beat-offset`で補正します。たとえば映像開始から0.4秒後が
最初の拍なら`--beat-offset 0.4`とします。

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
