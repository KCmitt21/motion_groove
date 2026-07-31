# GoPro + MediaPipe 動作解析

GoProのRGB映像からMediaPipe Pose Landmarkerで上半身姿勢を推定し、次の指標を
CSVとJSONへ保存する研究用ツールです。

- 頭部の揺れ：肩中心に対する頭部中心の移動量
- 上体の前傾：肩中心と腰中心を結ぶ3Dベクトルの角度
- 拍同期：頭部・上体の動作ピークと音楽の拍との時間差

主な利用方法は、USB接続したGoProからのリアルタイム解析と、保存済みMP4の
オフライン解析です。このプログラムは解析結果を保存しますが、USBストリームの
映像や音声そのものは保存しません。

## 動作確認済み環境

- macOS（Apple Silicon）
- Python 3.11
- GoPro HERO13 Black
- USB-Cによる有線接続
- RTSP Webcamストリーム

このREADMEはmacOSでのセットアップを基準にしています。Open GoPro公式SDKは
Python 3.11以上3.14未満を対象としていますが、このリポジトリでは依存関係を
揃えるためPython 3.11を標準環境とします。

Open GoProのUSB Webcamストリーミングは公式にはHERO11以降が対象です。対応機種と
最小ファームウェアは[Open GoPro公式Compatibility](https://gopro.github.io/OpenGoPro/)
を確認してください。

## リポジトリ構成

```text
motion_groove/
├── gopro_motion_analysis.py       # 解析プログラム
├── requirements-motion.txt        # Python依存パッケージ
├── test_gopro_motion_analysis.py  # 単体テスト
├── README.md
├── models/                        # MediaPipeモデル（初回実行時に作成）
└── out/                           # CSV・JSON出力（初回実行時に作成）
```

`models/`と`out/`はGit管理対象外です。研究データを共有するときは、リポジトリとは
別の安全な保存場所を使用してください。

## 1. 必要なもの

### ハードウェア

- HERO11以降の対応GoPro
- データ通信対応USB-Cケーブル
- macOSを搭載したMac
- 三脚または固定具
- 十分な照明

充電専用USBケーブルではGoProを検出できません。最初のセットアップではUSBハブを
使用せず、GoProをMacへ直接接続してください。

### ソフトウェア

- Git
- Homebrew
- Python 3.11

Homebrewが未導入の場合は、[Homebrew公式サイト](https://brew.sh/)の手順で
インストールしてください。

## 2. リポジトリとPython環境のセットアップ

### 2.1 リポジトリを取得

```bash
git clone https://github.com/KCmitt21/motion_groove.git
cd motion_groove
```

すでにリポジトリを取得済みの場合は、その`motion_groove`ディレクトリへ移動して
ください。

### 2.2 Python 3.11をインストール

```bash
brew install python@3.11
python3.11 --version
```

`Python 3.11.x`と表示されることを確認します。

### 2.3 仮想環境を作成

```bash
python3.11 -m venv .venv-motion311
source .venv-motion311/bin/activate
python --version
```

プロンプトの先頭に`(.venv-motion311)`が表示され、`python --version`が
Python 3.11になっていることを確認します。

次回以降は、リポジトリへ移動したあと次のコマンドだけで仮想環境を有効化できます。

```bash
source .venv-motion311/bin/activate
```

### 2.4 依存パッケージをインストール

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-motion.txt
```

インストール確認：

```bash
python -c "import cv2, mediapipe, open_gopro; print('dependencies: OK')"
```

`dependencies: OK`と表示されれば完了です。

### 2.5 テストを実行

```bash
python -m unittest -v test_gopro_motion_analysis.py
python gopro_motion_analysis.py --help
```

すべてのテストが`ok`になり、ヘルプが表示されることを確認してください。

## 3. GoProと撮影環境の準備

1. GoProのファームウェアを最新版に更新する
2. GoProを三脚へ固定する
3. レンズを可能なら`Linear`にする
4. 頭・両肩・両腰が常に画角内へ入るよう調整する
5. カメラの高さ、被写体との距離、照明を固定する
6. HyperSmoothを無効にするか、すべての実験で同じ設定にする
7. データ通信対応USB-CケーブルでGoProとMacを直接接続する
8. GoProの電源を手動で入れる
9. macOSからローカルネットワーク接続の許可を求められたら許可する
10. GoPro Webcam、Zoom、VLCなど、カメラを使う他のアプリを終了する

Open GoProの有線接続では、GoProの電源をUSBだけでプログラムからON/OFFすることは
できません。必要に応じて本体ボタンで電源を入れてください。

### 撮影方向を決める

撮影方向は自動検出されません。実際の配置に対応する`--view`を必ず指定します。

| 指定 | 被写体の向き | 前傾として正になる方向 |
|---|---|---|
| `front` | カメラを向く | カメラへ近づく方向 |
| `side-right` | 画面右を向く | 画面右方向 |
| `side-left` | 画面左を向く | 画面左方向 |

前傾角を重視する場合は横向きが測りやすく、頭揺れを他の条件と比較する場合は
正面撮影が適しています。撮影方向をまたいだ数値の直接比較は避けてください。

## 4. 初回の接続確認

まず10秒間だけ解析します。拍情報はまだ指定しません。

```bash
python gopro_motion_analysis.py \
  --gopro-usb \
  --view front \
  --duration 10
```

初回だけMediaPipe Pose Landmarker Fullモデルが
`models/pose_landmarker_full.task`へ自動ダウンロードされます。

成功時はプレビューが開き、終了後に次のような保存先が表示されます。

```text
CSV: out/motion_metrics.csv
集計: out/motion_summary_YYYYMMDD_HHMMSS.json
```

プレビュー上で骨格が安定して表示され、JSONの`pose_detection_fraction`が
できるだけ1.0に近いことを確認してください。

Open GoPro公式SDK単体で接続を確認したい場合は、次を実行できます。

```bash
gopro-webcam
```

公式デモと解析プログラムは同時に実行しないでください。デモを完全に終了してから
解析プログラムを起動します。公式デモについては
[Open GoPro Python SDK QuickStart](https://gopro.github.io/OpenGoPro/python_sdk/quickstart.html)
も参照してください。

## 5. 実際にデータを収集する

### 5.1 動きだけを収集

```bash
python gopro_motion_analysis.py \
  --gopro-usb \
  --view front \
  --duration 60
```

拍情報を指定しない場合、頭部揺れと前傾は計算されますが、拍同期関連のJSON値は
`null`になります。

### 5.2 既知BPMを使って拍同期も収集

```bash
python gopro_motion_analysis.py \
  --gopro-usb \
  --view front \
  --duration 60 \
  --bpm 120 \
  --beat-offset 0.0
```

`--beat-offset`は解析開始後、最初の拍が来るまでの秒数です。最初の拍が解析開始から
0.4秒後なら`--beat-offset 0.4`を指定します。

USBライブストリームから音声は取得しません。実際の音楽と厳密に同期させる場合は、
解析開始と音楽開始を揃えるか、次の拍時刻CSVを使用してください。

### 5.3 拍時刻CSVを使う

1列目へ解析開始からの拍時刻を秒単位で記録します。ヘッダーは省略可能です。

```csv
beat_time_s
0.40
0.90
1.40
1.90
```

実行例：

```bash
python gopro_motion_analysis.py \
  --gopro-usb \
  --view front \
  --duration 60 \
  --beat-times beats.csv
```

`--music`、`--beat-times`、`--bpm`は同時に指定できません。

### 5.4 研究用のファイル名を指定

CSVの既定ファイル名は常に`out/motion_metrics.csv`なので、次の実行で上書き
されます。実験では被験者・条件・試行を含む一意な名前を必ず指定してください。

```bash
python gopro_motion_analysis.py \
  --gopro-usb \
  --view side-right \
  --duration 60 \
  --bpm 120 \
  --output out/P001_side_right_trial01_metrics.csv \
  --summary out/P001_side_right_trial01_summary.json
```

ファイル名には氏名ではなく匿名化した被験者IDを使用してください。

### 5.5 保存済みGoPro MP4を解析

GoProのSDカードからコピーした動画は、USB Webcamを使わず解析できます。

```bash
python gopro_motion_analysis.py \
  --source GX010001.MP4 \
  --music GX010001.MP4 \
  --view front \
  --output out/P001_trial01_metrics.csv \
  --summary out/P001_trial01_summary.json \
  --no-preview
```

MP4内の音声を`librosa`が直接読めない場合はFFmpegを導入し、音声をWAVへ変換して
指定します。

```bash
brew install ffmpeg
ffmpeg -i GX010001.MP4 -vn music.wav
```

その後、`--music music.wav`を使用してください。

## 6. 収録中と終了時の操作

- プレビューの`Q`または`Esc`：正常終了
- ターミナルの`Control+C`：正常終了してCSV・JSONを保存
- `--duration`：指定秒数で自動終了
- `--no-preview`：プレビューを表示しない

安全な終了手順：

1. `Q`、`Esc`、または`Control+C`で解析を終了する
2. ターミナルに`CSV:`と`集計:`が表示されるまで待つ
3. プロンプトへ戻ったことを確認する
4. USB-Cケーブルを抜く
5. 必要ならGoProの電源を切る

解析中にUSBケーブルを抜かないでください。正常終了時はGoProのWebcamストリームを
自動停止します。

## 7. 出力データ

### CSV：フレーム単位

| 列 | 内容 |
|---|---|
| `frame` | フレーム番号 |
| `time_s` | 解析開始からの時刻（秒） |
| `pose_detected` | 姿勢を解析できた場合1、できなかった場合0 |
| `head_x_shoulder_width` | 肩中心に対する頭部X位置（肩幅単位） |
| `head_y_shoulder_width` | 肩中心に対する頭部Y位置（肩幅単位） |
| `head_sway_from_baseline` | 最初の基準姿勢からの頭部距離 |
| `head_frame_displacement` | 直前の有効フレームからの頭部移動量 |
| `forward_lean_deg` | 平滑化した符号付き前傾角（度） |
| `total_torso_tilt_deg` | 前後左右を区別しない上体傾斜角（度） |
| `is_forward_leaning` | 前傾閾値以上なら1 |
| `motion_energy` | 頭部移動と前傾変化を合成した動作量 |
| `motion_peak` | 動作ピークなら1 |
| `nearest_beat_error_ms` | 最寄りの拍との時間差（ms） |
| `on_beat` | 拍の前後100 ms以内なら1 |

姿勢未検出フレームでは、`pose_detected`以外の解析列が空欄になります。

### JSON：セッション集計

| キー | 内容 |
|---|---|
| `duration_s` | 解析時間 |
| `total_frames` | 全フレーム数 |
| `pose_detected_frames` | 姿勢検出成功フレーム数 |
| `pose_detection_fraction` | 姿勢検出率 |
| `detected_music_tempo_bpm` | 指定または音声から推定したBPM |
| `motion_peak_count` | 動作エネルギーのピーク数 |
| `beat_count` | 解析対象の拍数 |
| `mean_abs_beat_error_ms` | 動作ピークと最寄り拍の平均時間差 |
| `on_beat_fraction_100ms` | 拍の前後100 ms以内に入ったピーク割合 |
| `beat_sync_score` | 0〜1の同期スコア。1に近いほど拍に近い |
| `head_sway_rms_shoulder_width` | 頭揺れのRMS（肩幅単位） |
| `head_sway_p95_shoulder_width` | 頭揺れの95パーセンタイル |
| `head_path_length_shoulder_width` | 頭部の累積移動量 |
| `forward_lean_mean_deg` | 平均前傾角 |
| `forward_lean_max_deg` | 最大前傾角 |
| `forward_lean_fraction` | 前傾閾値以上だった有効フレーム割合 |
| `forward_lean_event_count` | 閾値未満から閾値以上になった回数 |
| `source` | 映像入力 |
| `camera_view` | 指定した撮影方向 |
| `forward_lean_threshold_deg` | 前傾判定閾値 |
| `beat_method` | `audio`、`beat_times`、`bpm`、`none` |

## 8. 研究用の撮影プロトコル

比較可能なデータを得るため、全試行で次を統一してください。

- カメラ機種とファームウェア
- `--view`
- カメラの高さ、角度、被写体との距離
- 画角、照明、背景
- `--duration`
- `--forward-threshold`
- `--baseline-seconds`
- `--smoothing-seconds`
- BPM、拍開始位置、または拍時刻CSV

各試行の開始後、既定では最初の2秒間が頭部基準位置の計算に使われます。この間は
正面または指定した横向きの自然な基準姿勢を保ってください。頭、両肩、両腰を
常に画角内へ入れ、原則として画面内は1人にしてください。

記録しておくべきメタデータ：

- 匿名化した被験者ID
- 条件名と試行番号
- 撮影日時
- GoPro機種とファームウェア
- カメラ方向と距離
- 音楽・BPM・拍オフセット
- 使用したGitコミット

使用コードのコミットは次で確認できます。

```bash
git rev-parse HEAD
```

### 解釈上の制限

- 単眼RGB映像から得る3D奥行きは推定値であり、実測角度ではありません。
- 頭部移動は画面上の肩幅で正規化されます。横向きでは肩が重なり、値が正面撮影
  より大きくなりやすいため、異なる`--view`間で直接比較しないでください。
- `head_path_length_shoulder_width`は往復運動と検出揺らぎも累積します。
- `forward_lean_max_deg`は一瞬の誤検出の影響を受けるため、平均値、割合、
  イベント数、CSV時系列と併せて判断してください。
- `--bpm`による同期評価は解析開始時刻を基準にするため、実際の第1拍とのずれが
  スコアへ影響します。
- `beat_sync_score`だけで統計的な同期の有無を断定しないでください。

## 9. 主なオプション

| オプション | 用途 | 既定値 |
|---|---|---|
| `--gopro-usb` | USB GoProを自動検出して解析 | 無効 |
| `--gopro-id ID` | 複数台接続時のGoPro識別子 | 自動検出 |
| `--gopro-protocol` | `RTSP`または`TS` | `RTSP` |
| `--source` | カメラ番号、MP4、RTSP/UDP URL | `0` |
| `--view` | `front`、`side-right`、`side-left` | `front` |
| `--duration` | 解析秒数 | 手動終了まで |
| `--output` | フレーム別CSVの保存先 | `out/motion_metrics.csv` |
| `--summary` | 集計JSONの保存先 | 日時付きファイル |
| `--bpm` | 既知のBPM | なし |
| `--beat-offset` | 解析開始から最初の拍までの秒数 | `0.0` |
| `--beat-times` | 拍時刻CSV | なし |
| `--music` | 拍検出する音声またはMP4 | なし |
| `--forward-threshold` | 前傾判定角度 | `15.0` |
| `--visibility-threshold` | ランドマーク可視性閾値 | `0.5` |
| `--baseline-seconds` | 頭部基準位置の収集時間 | `2.0` |
| `--smoothing-seconds` | 指標の平滑化時定数 | `0.12` |
| `--width`、`--height` | 要求する入力解像度 | `1920 × 1080` |
| `--fps` | 要求FPS・取得失敗時の代替FPS | `30` |
| `--no-preview` | プレビューを非表示 | 無効 |
| `--model` | MediaPipeモデルのパス | `models/...task` |
| `--no-download` | モデル自動取得を禁止 | 無効 |

完全な一覧：

```bash
python gopro_motion_analysis.py --help
```

## 10. トラブルシューティング

### `No matching distribution found`または依存関係をインストールできない

Python 3.11の仮想環境を使用しているか確認します。

```bash
which python
python --version
python -m pip --version
```

別バージョンの仮想環境を作ってしまった場合は、その環境を無理に修正せず、
Python 3.11で`.venv-motion311`を作り直してください。

### GoProのUSB検出がタイムアウトする

次を順番に確認します。

1. GoProの電源が入っている
2. データ通信対応ケーブルを使っている
3. USBハブではなくMacへ直接接続している
4. GoPro Webcam、VLC、Zoom、ブラウザ等がカメラを使用していない
5. macOSのローカルネットワークアクセスが許可されている
6. GoProとMacを再起動してから接続し直す
7. GoProのファームウェアを更新する

それでも動かない場合：

```bash
gopro-webcam
```

公式デモも失敗する場合は、コードではなくUSB接続、macOS権限、GoPro側の状態を
先に確認してください。

### `ReadTimeout`またはGoProがWebcam状態のままになる

異常終了後にストリームが残っている場合、GoProを接続したまま次を実行します。

```bash
curl --noproxy '*' \
  'http://172.23.147.51:8080/gopro/webcam/stop'
```

状態確認：

```bash
curl --noproxy '*' \
  'http://172.23.147.51:8080/gopro/webcam/status'
```

`status`が`0`または`1`なら停止中、`2`ならストリーム動作中です。改善しない場合は
GoProの電源を切り、USBを接続し直してください。

### 映像入力を開けない

このリポジトリで確認済みのHERO13 BlackではRTSPが動作したため、既定値はRTSPです。
まず他のカメラアプリと`gopro-webcam`を終了してください。

別機種でRTSPが動かない場合だけTSを試します。

```bash
python gopro_motion_analysis.py \
  --gopro-usb \
  --gopro-protocol TS \
  --view front \
  --duration 10
```

### プレビューは出るが姿勢を検出できない

- 頭、肩、腰がすべて画角内にあるか確認する
- 被写体を大きく映しすぎない
- 逆光を避け、正面から照明を当てる
- 身体と背景のコントラストを確保する
- 画面内を1人にする
- `pose_detection_fraction`で検出率を確認する

検出率が低い試行は、頭揺れ、前傾、拍同期の集計値をそのまま比較しないでください。

### MediaPipeモデルをダウンロードできない

初回実行にはGoogle Storageへのネットワーク接続が必要です。別の端末で
Pose Landmarker Fullモデルを取得した場合は、次のように指定できます。

```bash
python gopro_motion_analysis.py \
  --gopro-usb \
  --model /path/to/pose_landmarker_full.task \
  --no-download
```

## 11. 参考資料

- [Open GoPro公式仕様・対応機種](https://gopro.github.io/OpenGoPro/)
- [Open GoPro Python SDK](https://gopro.github.io/OpenGoPro/python_sdk/)
- [Open GoPro Python SDK QuickStart](https://gopro.github.io/OpenGoPro/python_sdk/quickstart.html)
- [Open GoPro FAQ / Known Issues](https://gopro.github.io/OpenGoPro/docs/faq/)
- [MediaPipe PoseLandmarker Python API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker)
