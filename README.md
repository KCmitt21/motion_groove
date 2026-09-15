# 3視点・奏者インタラクション解析MVP

同期済み3視点映像から、ギタリスト／ベーシストの姿勢、頭部姿勢、粗い注視先、楽器点、
音響特徴と相互作用を解析する研究用Pythonパイプラインです。既定設定は先頭60秒を処理します。

> 注視先はFace Landmarkerの頭部姿勢による粗い代理指標で、眼球注視計測ではありません。
> `mock`姿勢は配線テスト専用の合成値で、研究結果に使用できません。

## 確認済み入力

| camera | file | view | frames |
|---|---|---|---:|
| `cam_guitar_close` | `GX010235.mov` | ギター寄り | 4037 |
| `cam_bass_close` | `GX010262.mov` | ベース寄り | 4037 |
| `cam_wide` | `GX010340.mov` | 2人全景（左bass、右guitar） | 4035 |

3本とも1920×1080、`30000/1001` fps、開始タイムコード`14:03:53:01`、48 kHz
モノラル音声です。パイプラインはタイムコードとfpsを開始時に検証し、同じ`frame_idx`を
ロックステップで読みます。全尺では最短の4035フレームで停止します。内蔵音声の先頭PTSは
映像0秒より約4.7秒遅いため、そのオフセットを`ffprobe`で取得して動画時刻へ戻します。各表の時刻は
コンテナPTSに依存せず、常に次式です。

```text
time_sec = frame_idx * 1001 / 30000
```

60秒設定では`frame_idx=0..1798`の1799フレームを対象にします。動画音声が映像より短い場合、
音声範囲外のフレーム特徴は欠損になります。

## 構成

```text
motion_groove/
├── configs/
│   ├── mvp.yaml                 # 実推論（RTMPose + Face Landmarker）
│   ├── mvp_mock.yaml            # 数秒の配線テスト
│   └── anipose.toml             # Anipose設定雛形
├── musician_interaction/
│   ├── video.py                 # fps/TC検証・3本ロックステップ読込
│   ├── pose.py                  # RTMPose WholeBody・人物ID固定
│   ├── face.py / gaze.py        # 頭部Euler角・粗い注視分類
│   ├── instruments.py           # 楽器5点CSVアダプタ
│   ├── audio.py                 # onset/RMS/spectral flux
│   ├── features.py              # 速度・相互相関・音/動作lag・イベント加算
│   ├── qc.py / outputs.py       # QC・表・動画・グラフ
│   ├── calibration.py           # ChArUco内部/外部校正
│   ├── triangulation.py         # 多視点DLT・再投影誤差
│   └── pipeline.py / cli.py
├── dlc/
│   ├── manage_project.py        # DLC作成/ラベル/学習/評価/推論
│   ├── export_interchange.py    # DLC CSV→本パイプライン形式
│   └── project_template.yaml    # 5点×guitar/bass
├── instrument_detector/
│   └── manage.py                # 5点姿勢推定の準備・学習・CSV出力
├── tests/
├── run_analysis.py
├── requirements-core.txt
├── requirements-pose.txt
├── requirements-instruments.txt
└── requirements-dlc.txt
```

## インストール

MediaPipe/OpenMMLabの対応範囲に合わせPython 3.11を使用します。既存の`.venv-motion311`
を再利用する場合も、次のコア依存を追加してください。

```bash
cd /Users/keishi-mac/code/motion_groove
python3.11 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-core.txt
```

`mediapipe`はApple Silicon用wheelがある0.10.35へ固定しています。macOSのヘッドレス実行では
GPUサービスを作れない場合があるため、Face Landmarkerは別プロセスで実行し、ネイティブ障害が
解析本体を終了させないようにしています。

Face Landmarkerモデルを配置します（URLはMediaPipe公式配布物）。

```bash
curl -L \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
  -o models/face_landmarker.task
```

RTMPose環境はOpenMMLabのバイナリ互換性があるため、使用するPyTorchに合わせて分離することを
推奨します。Apple SiliconでMPS演算が非対応ならYAMLの`device: cpu`を維持してください。

```bash
python -m pip install "torch==2.2.2" "torchvision==0.17.2" openmim
PIP_CONSTRAINT=constraints-openmmlab.txt mim install "mmengine==0.10.7"
python -m pip install "setuptools==60.2.0" ninja
MMCV_WITH_OPS=1 PIP_CONSTRAINT=constraints-openmmlab.txt \
  python -m pip install --no-build-isolation "mmcv==2.1.0"
PIP_CONSTRAINT=constraints-openmmlab.txt mim install "mmdet==3.2.0"
# MMPose 1.3.2が依存する古いchumpyのPEP 517不具合を修正したcommit。
python -m pip install --no-build-isolation \
  "git+https://github.com/mattloper/chumpy.git@4228d703b622e172e843438fe0fada102979361a"
PIP_CONSTRAINT=constraints-openmmlab.txt \
  python -m pip install --no-build-isolation "mmpose==1.3.2"
python -c "from mmpose.apis import MMPoseInferencer; print('MMPose OK')"
```

`configs/mvp.yaml`はMMPose 1.3.2公式の`wholebody`別名を使用します。これは
RTMPose-m WholeBodyと既定のRTMDet-m検出器を選択します。別モデルを使う場合は、その版の
model-indexに登録された設定名または設定ファイルを指定します。

## 実行

まず入力だけ検証します。

```bash
python run_analysis.py validate --config configs/mvp.yaml
python run_analysis.py face-check --config configs/mvp.yaml --image /path/to/frame.jpg
```

依存や出力配線を短時間で確認します。これは合成姿勢であり、解析値ではありません。

```bash
python run_analysis.py run --config configs/mvp_mock.yaml --max-seconds 2
```

実推論の1分MVP:

```bash
python run_analysis.py run --config configs/mvp.yaml --max-seconds 60
```

全尺はYAMLの`max_seconds: null`にするか、十分大きな値を指定します。どの場合も最短動画の
終了で停止します。別WAVを使う場合は`audio.wav_path`へ指定し、動画0秒に対するWAV先頭時刻を
`audio.wav_offset_sec`へ指定します。

## 人物IDと注視分類

実画像を確認した初期条件を`visible_performers`と`initial_left_to_right`に記載済みです。
最初の有効フレームを左右順で意味IDへ割り当て、以後は有効な身体点の中心と直前位置の距離が
最小になる対応を選びます。閉塞後の再登場や大きな交差は重畳動画で必ず確認してください。

全画面では顔が小さすぎるため、MMPose WholeBodyの顔68点から人物別ROIを作り、余白を付けて
`256x256`へ拡大した画像だけをFace Landmarkerへ入力します。検出ランドマークは元フレーム座標へ
戻してから、6顔点をPnPへ入力してyaw/pitch/rollを推定します。ROI作成時点で人物IDが確定して
いるため、検出顔はその人物へ直接割り当てます。MMPoseが短時間欠損した場合は直前ROIを再利用し、
検出失敗時は別の余白倍率で再試行します。

```yaml
face:
  roi:
    enabled: true
    scale: 2.5
    retry_scales: [3.5]
    min_size_px: 96
    output_size: 256
    reuse_frames: 5
```

### LAEO連続スコア

相互注視の主解析には、カメラ基準のyaw分類ではなく`laeo_by_time`を使います。PnPの回転から
顔前方の単位ベクトル`g_A`, `g_B`、PnPの並進から同一カメラ座標系の顔位置を得て、顔AからB、
BからAへのベクトルとの角度`theta_A`, `theta_B`（度）を計算します。各人物のスコアと相互スコアは
次式です。

```text
p_A = exp(-(theta_A ** 2) / (2 * sigma_deg ** 2))
p_B = exp(-(theta_B ** 2) / (2 * sigma_deg ** 2))
laeo_score = p_A * p_B
```

```yaml
laeo:
  enabled: true
  cameras: [cam_wide]
  performers: [bassist, guitarist]  # 順にA, B
  sigma_deg: 25.0
  smoothing:
    method: gaussian                # moving_averageも指定可能
    radius_frames: 10               # 前後10フレーム
    gaussian_sigma_frames: 4.0
```

`laeo_by_time.csv`には`timestamp`, `theta_A`, `theta_B`, `p_A`, `p_B`, `laeo_score`と、
平滑化後の`laeo_score_smoothed`を保存します。平滑化は中央窓なので前後のフレームを使い、顔検出の
欠損値を補間したり欠損区間をまたいだりしません。`head_pose_gaze`にも監査用としてPnPの顔位置と
顔方向ベクトルを保存します。

PnP位置は標準顔モデルと近似焦点距離から求める単眼3D推定であり、厳密な実空間座標ではありません。
そのため`sigma_deg`は目視ラベルを用いて調整し、奥行き差が大きい配置ではカメラ内部パラメータを
使った評価も検討してください。

従来の可視化・レポートとの互換性のため、頭向き、相手の2D方向、
自楽器点から`partner / own_instrument / forward / downward / unknown`へ分類します。閾値は
`gaze`節にありますが、これはLAEOの主スコアではありません。DLC点が未指定の場合、自楽器判定に
必要な点は欠損になり、無理に補いません。

固定配置の`cam_wide`では、画像上の人物間ベクトルではなく、相手を見ていることが既知のフレーム
から求めた人物別のyaw/pitch中央値を使えます。既知フレームを`head_pose_gaze.csv`から選び、
中央値を次のように設定します。許容角は既知フレームのばらつきを確認して調整してください。

```yaml
gaze:
  reference_angles:
    cam_wide:
      bassist:
        partner:
          yaw_deg: -40.0       # 既知フレームの中央値に置き換える
          pitch_deg: 2.0       # 既知フレームの中央値に置き換える
          yaw_tolerance_deg: 20.0
          pitch_tolerance_deg: 15.0
      guitarist:
        partner:
          yaw_deg: 40.0        # 既知フレームの中央値に置き換える
          pitch_deg: 2.0       # 既知フレームの中央値に置き換える
          yaw_tolerance_deg: 20.0
          pitch_tolerance_deg: 15.0
```

同じ人物の下へ`forward`、`downward`、`own_instrument`も同じ形式で追加できます。複数の参照状態の
許容範囲が重なった場合は、yaw/pitchの正規化距離が最も小さい状態を採用します。

`reference_angles`にカメラ名がある場合、そのカメラでは従来の画像ベクトル方式へフォールバック
しません。人物の参照値が未設定なら`unknown_reason=reference_profile_missing`、参照楕円の外なら
`outside_reference_tolerance`になります。また、`head_pose_quality`で設定した絶対角上限を超えた
PnP結果は`head_pose_angle_outlier`として分類から除外します。

## 楽器5点姿勢推定

楽器の移動、向き、ネックの上下動を得るため、`guitar` / `bass`の2クラスと、各楽器に共通する
`body_center, bridge, neck_joint, nut, head_tip`の5点をYOLO Poseで推定します。各クラスについて
1フレームに複数インスタンスがある場合、矩形confidenceが最大の1つを採用します。

UltralyticsとPyTorchは専用環境へ分離します。Apple Siliconでは`mps`を試せますが、Pose学習で
問題が発生する場合は各コマンドの`--device mps`を`--device cpu`へ変更してください。

```bash
cd /Users/keishi-mac/code/motion_groove
python3.11 -m venv .venv-instruments
source .venv-instruments/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-instruments.txt
```

データセット構造を作り、3本の動画から1秒間隔でアノテーション画像を抽出します。

```bash
python -m instrument_detector.manage init-dataset --output datasets/instruments_pose
python -m instrument_detector.manage extract \
  --video cam_guitar_close=movie/exp_2026_08_29/GX010235.mov \
  --video cam_bass_close=movie/exp_2026_08_29/GX010262.mov \
  --video cam_wide=movie/exp_2026_08_29/GX010340.mov \
  --output datasets/instruments_pose/raw/images \
  --interval-sec 1.0
```

CVAT、Ultralytics Platform等で矩形と5点を付け、YOLO pose形式で
`datasets/instruments_pose/raw/labels`へ保存します。クラス番号は`0=guitar`, `1=bass`です。
点の順序は以下で固定します。

```text
0 body_center
1 bridge
2 neck_joint
3 nut
4 head_tip
```

ラベル1行は次の形式で、矩形と点座標は画像サイズで0〜1へ正規化します。各点の`v`は
`0=未指定`, `1=遮蔽されているが位置を指定`, `2=可視`です。

```text
class_id box_x box_y box_w box_h \
  p0_x p0_y p0_v p1_x p1_y p1_v p2_x p2_y p2_v \
  p3_x p3_y p3_v p4_x p4_y p4_v
```

対象楽器がない画像にも空の`.txt`を置きます。close画像では写っている自楽器だけ、wide画像では
ギターとベースの両方を囲みます。照明変化、遮蔽、モーションブラー、画面端を含めてください。
ラベル後、80%/20%へ分割します。

```bash
python -m instrument_detector.manage split \
  --dataset datasets/instruments_pose \
  --validation-fraction 0.2 \
  --seed 29
```

COCO keypoints事前学習済みの軽量YOLO26n-poseを初期値として、2クラス・5点へ追加学習します。
初回は`yolo26n-pose.pt`が自動ダウンロードされます。

```bash
python -m instrument_detector.manage train \
  --data datasets/instruments_pose/dataset.yaml \
  --model yolo26n-pose.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device mps
```

既定では最良weightが`models/instrument_detector/yolo26n_pose/weights/best.pt`へ保存されます。
検証データで矩形mAPに加えてpose mAPを確認します。

```bash
python -m instrument_detector.manage validate \
  --model models/instrument_detector/yolo26n_pose/weights/best.pt \
  --data datasets/instruments_pose/dataset.yaml \
  --device mps
```

3カメラすべてを推論し、カメラ別CSVと確認用MP4を作ります。`--ema-alpha 0.35`は連続検出中の
全5点を指数移動平均で平滑化します。生の推定点を保存する場合は既定値の`1.0`を指定します。

```bash
python -m instrument_detector.manage infer-config \
  --config configs/mvp.yaml \
  --model models/instrument_detector/yolo26n_pose/weights/best.pt \
  --output-dir out/dlc \
  --preview-dir out/dlc/previews \
  --confidence 0.50 \
  --keypoint-confidence 0.50 \
  --ema-alpha 0.35 \
  --device mps
```

生成CSVは`frame_idx,instrument,keypoint,x,y,score`形式で、検出フレームごとに5点を出力します。
`configs/mvp.yaml`を次のように変更してから本解析を再実行します。

```yaml
instruments:
  keypoint_csv:
    cam_guitar_close: out/dlc/cam_guitar_close.csv
    cam_bass_close: out/dlc/cam_bass_close.csv
    cam_wide: out/dlc/cam_wide.csv
  score_threshold: 0.50
```

```bash
source .venv-analysis/bin/activate
python run_analysis.py run --config configs/mvp.yaml --max-seconds 60
```

採用前に`out/dlc/previews/*.mp4`で矩形と5点、楽器の取り違え、遮蔽直後の飛びを確認し、
`qc_instruments.csv`の各点の欠損率・低信頼度率を確認してください。なお現在の
`own_instrument`は「頭部pitchが下向きで、同フレームに自楽器中心が存在する」という粗い分類で、
眼球が実際に楽器を注視したことまでは保証しません。

本解析を再実行すると、追加で以下を生成します。

- `tables/instrument_motion_features.csv`: フレームごとの楽器軸角、ネック角、先端高さ、上向き速度、角速度
- `tables/instrument_motion_events.csv`: 上昇区間と周期揺れ区間。`near_end=True`は動画末尾5秒以内
- `graphs/instrument_orientation_and_height.png`: 楽器軸角とヘッド先端高さの時系列

主な列の定義は次のとおりです。

- `axis_angle_deg`: `body_center`から`head_tip`への画像平面上の角度。上向きを正とする
- `neck_angle_deg`: `neck_joint`から`head_tip`への角度
- `head_tip_height_norm`: 画面下端0、上端1としたヘッド先端高さ
- `head_tip_vertical_speed_norm_s`: 画像対角長で正規化した上向き速度
- `instrument_rise`: 平滑化した上向き速度が閾値以上で0.2秒以上継続した区間
- `visual_oscillation`: ネック角の3〜9 Hz帯パワー比と角度RMSが閾値以上の区間

閾値は`configs/mvp.yaml`の`analysis.instrument_motion`で変更します。`visual_oscillation`は
ビブラート候補となる視覚的な周期揺れであり、音高ビブラートそのものの判定ではありません。
演奏上のビブラートを確定する場合は、音声から基本周波数を推定し、周期的な音高変動と同時刻かを
照合する必要があります。

## DeepLabCut楽器5点

楽器ごとの点は`body_center, bridge, neck_joint, nut, head_tip`です。DLCはGUI・学習依存が
大きいため別環境を推奨します。

```bash
python3.11 -m venv .venv-dlc
source .venv-dlc/bin/activate
python -m pip install -r requirements-dlc.txt
python dlc/manage_project.py create --videos movie/exp_2026_08_29/*.mov
# 生成config.yamlへ dlc/project_template.yaml のindividuals/bodyparts/skeletonを反映
python dlc/manage_project.py extract --config dlc/projects/.../config.yaml
python dlc/manage_project.py label --config dlc/projects/.../config.yaml
python dlc/manage_project.py train --config dlc/projects/.../config.yaml
python dlc/manage_project.py evaluate --config dlc/projects/.../config.yaml
python dlc/manage_project.py infer --config dlc/projects/.../config.yaml --videos movie/exp_2026_08_29/*.mov
python dlc/export_interchange.py DLC_OUTPUT.csv out/dlc/cam_wide.csv
```

変換後CSVを`instruments.keypoint_csv.cam_wide`等へ指定します。学習前に照明、閉塞、左右端、
高速運動を含むフレームを各カメラから抽出し、評価誤差とlikelihood分布を確認してください。

## 校正と3D

現時点では校正画像がないため、通常実行では3Dを`skipped`としてマニフェストへ記録します。
同じ解像度・レンズ設定・フォーカスでChArUco板を全3台に同時提示し、画像を同じファイル名で
次のように置きます。板の実寸はYAMLと一致させてください。

```text
calibration/images/
├── cam_guitar_close/000001.png ...
├── cam_bass_close/000001.png ...
└── cam_wide/000001.png ...
```

```bash
python run_analysis.py calibrate --config configs/mvp.yaml
python run_analysis.py triangulate --config configs/mvp.yaml \
  --keypoints out/mvp/tables/pose_keypoints_2d.csv \
  --output out/mvp/tables/keypoints_3d.csv
```

内部校正は各カメラ8枚以上、外部校正は各ペア5組以上の有効検出を要求します。OpenCVの多視点
DLTは2ビュー以上かつ信頼度閾値以上のみ三角測量し、`n_views`とカメラ別再投影誤差を保存します。
Aniposeを使う場合は`configs/anipose.toml`を実際のフォルダ構成へ合わせてください。

## 出力と品質管理

`out/mvp/`以下へCSV（`pyarrow`があればParquetも）、カメラ別重畳MP4、PNGグラフ、
`manifest.json`を保存します。主な表は以下です。

- `pose_keypoints_2d`, `head_pose_gaze`, `instrument_keypoints_2d`
- `laeo_by_time`: 相手方向との角度、人物別確率、元のLAEOスコア、時間平滑化後スコア
- `audio_features`, `audio_onsets`, `motion_features`
- `performer_cross_correlation`: 正lagは「bassがguitarに遅れる」
- `audio_motion_lag`: 正lagは「動作が音に遅れる」
- `onset_triggered_average`: 発音時刻0秒基準のイベント同期加算
- `qc_keypoints`, `qc_faces`, `qc_instruments`: 欠損率・低信頼度率
- `qc_sync`: 全体/序盤/終盤の音声オフセットと推定ドリフト
- `qc_reprojection`: 3D実行時の再投影誤差と警告

低信頼度座標は生値表には残しますが、速度・三角測量には使用しません。欠損区間をまたぐ速度は
計算せず、相関には両系列が有効な標本だけを使います。最終的な採用前に重畳動画、ID交差、
顔角の符号、DLC点、同期ドリフト、再投影誤差を試行ごとに確認してください。

解析済みの`head_pose_gaze.csv`から、頭向きの時系列、分類別集計、ワイド映像で両者が同時に
`partner`となる区間を別表へ整理できます。

```bash
python run_analysis.py head-summary \
  --input out/mvp/tables/head_pose_gaze.csv \
  --output-dir output/tables \
  --camera cam_wide
```

`mutual_facing`は、同一フレームでベーシストとギタリストの両方が`partner`に分類された場合のみ
`True`です。片方だけの場合は`one_sided_partner=True`として区別します。
`head_direction_by_time`にはフレーム別の`unknown_reason`、`unknown_reason_summary`にはカメラ・人物・
理由別の件数、全フレーム比率、unknown内比率を保存します。古い`head_pose_gaze.csv`も読み込めますが、
当時の詳細理由は`reason_not_recorded`として扱います。

## テスト

```bash
python -m unittest discover -s tests -v
python -m compileall -q musician_interaction instrument_detector dlc run_analysis.py
```
