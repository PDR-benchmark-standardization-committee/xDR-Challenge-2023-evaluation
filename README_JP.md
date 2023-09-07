# xDR-Challenge-2023-evaluation
xDR Challenge 2023コンペティションで使われる指数の計算を行えるツールです。

なおxDR Challenge 2023のサンプルデータは本ツールには含まれておりません。
コンペティション参加登録済みの方にのみサンプルが公開されております。
各自でgt, gisデータをそれぞれ/xDR-Challenge-2023-evaluation/dataset/のgt, gisフォルダー下にコピーして使用してください。

以下が、計算可能な指数の概要と、対応する指標と必要条件になります。


| **指数** | **対応する指標と必要条件** | **概要** |
 ---       | ---                     |---
| I_ce        | 誤差絶対量 : CE (Circular Error)               　 | 正解座標と時間的最近傍の軌跡の距離が近いか評価　 　         |
| I_ca        | 誤差分偏移 : CA (Circular Accuracy)        　     | 正解座標との誤差偏移をyaw方向をy軸とした相対座標系で評価 　         |
| I_eag       | 誤差累積速度 : EAG (Error Accumulation Gradient)  | 位置補正のための座標からの誤差の累積スピードを評価          |
| I_ve        | 速度誤差絶対量 : (Velocity Error)                 | 正解歩行速度と時間的最近傍の軌跡の歩行速度が近いか評価       |
| I_obstacle  | 軌跡経路基準 : Requirement for Obstacle Avoidance | 地図上の軌跡が人間が侵入できない障害物を通過していないか評価 |

### 各指標の算出式
```
指数         上限(100)    下限(0)      算出式
I_ce       | ce < 1.0   | 30 < ce   | 100 - (100 * (ce - 1.))/29
I_ca       | ca = 0.0   | 10 < ca   | 100 - (10 * ca)
I_eag      | eag < 0.05 | 2.0 < eag | 100 - (100 * (eag - 0.05))/1.95
I_ve       | ve < 0.1   | 2.0 < ve  | 100 - (100 * (ve - 0.1))/1.9
I_obstacle | obs = 1.0  | obs = 0.0 | 100 * obs
```

### 評価周波数
評価周波数は正解データの周波数に依存しており、xDR Challenge 2023で使用する正解データの周波数は100Hzになります。
推定軌跡周波数が100Hzを下回る場合、正確な評価ができない可能性があります。
その為、100Hzで軌跡推定を行うか、もしくは100Hzへアップサンプリングすることを推奨しております。
ただし、速度評価のみ局所的な値変動を抑えるため1Hz分の平滑化後の値を評価しております。

## 必要条件
```
python==3.8.5
numpy==1.23.4
pandas==1.5.0
scipy==1.8.1
matplotlib==3.3.2
seaborn==0.10.1
```

## ファイルの概要

| **ファイル名** | **概要** |
 ---            |---
| do_evaluation_XC2023.py | 評価実行用スクリプト |
| requirements.txt        | Pythonの必要ライブラリのバージョンをまとめたファイル |
<!--
| **evtools**             |---
| bitmap_tools.py         | bitmapを扱う為の関数をまとめたスクリプト |
| EV_converter.py         | 評価値をxDR Challenge 2023コンペティションで使う指標へ変換する処理をまとめたスクリプト |
| evaluate_CA.py          | CA評価を行う為のスクリプト |
| evaluate_CE.py          | CE評価を行う為のスクリプト |
| evaluate_EAG.py         | EAG評価を行う為のスクリプト |
| evaluate_OE.py          | obstacle評価を行う為のスクリプト |
| evaluate_VE.py          | Velocity評価を行う為のスクリプト |
-->
## 使用方法
### Step.1  インストール
```
git clone --recursive https://github.com/PDR-benchmark-standardization-committee/xDR-Challenge-2023-evaluation
cd xDR-Challenge-2023-evaluation
pip install -r requirements.txt
```

### Step.2 推定ファイルを配置
推定軌跡ファイルをBLE情報あり(_est), BLE情報なし(_pdr_est)両方とも[dataset]/[traj]/に配置してください。
```
xDR-Challenge-2023-evaluation/
├ dataset/
|   ├ gis/
|   |  ├ beacon_list.csv
|   |  ├ FLD01_0.01_0.01.bmp
|   |  ├ FLU01_0.01_0.01.bmp
|   |  └ FLU02_0.01_0.01.bmp
|   |
|   ├ gt/
|   |  ├ *_*_gt.csv
|   |  └ *_*_gt.csv
|   |
|   └ traj/
|      ├ *_*_est.csv [**estimation with BLE files**]
|      ├ *_*_pdr_est.csv [**estimation files**]_pdr.csv
|
├ evtools/
├ output/
├ do_evaluation_XC2023.py
├ requirements.txt
└ README.md
```

#### \*\_\*\_est.csv, \*\_\*\_pdr_est.csvの構成
軌跡ファイルの中身はカンマ区切りで以下のような構成となります。
※軌跡ファイル内にヘッダーは含みません。
| timestamp (s) | x(m) | y(m) | floor |
| ---           | ---  | ---  | ---   

### Step.3 評価の実行
評価を行う推定軌跡のフォルダーパスをコマンドライン引数に入力する必要があります。
```
python do_evaluation_XC2023.py -t [estimation_path]
```
デモデータの評価を実行したい場合は、以下のスクリプトを実行してください。
```
python do_evaluation_XC2023.py -t dataset/traj/
```

各軌跡の評価指数と総合スコアおよび全軌跡の平均評価指数と平均総合スコアが保存されたcsvファイルが[output]フォルダー内に出力されます。

## コマンドライン引数
以下のコマンドライン引数を追加することが可能となっています。

### 1. 一部評価のgraph, imageを出力
--draw オプションを追加することで、CA評価ヒストグラム・EAG評価グラフ・Obstacle評価画像が保存された[軌跡名]フォルダーが[output]内に作成されます。
```
python do_evaluation_XC2023.py -t [estimation_folder] --draw
```

### 2. 結果出力先フォルダーの指定
--output_path * オプションを指定することで、評価結果ファイルおよびgraph, imageが保存されるフォルダーを指定することができます。
```
python do_evaluation_XC2023.py -t [estimation_folder] --output_path new_output_folder/
```

### 3. スコア計算のための各指数の重みを選択
--est_weight * * * * * オプションを追加することで、コンペティションスコアを求めるための各指数の重みを指定することができます。
デフォルトの各指数の重みは以下のようになっています。
```
I_ce = 0.25
I_ca = 0.20
I_eag = 0.25
I_ve = 0.15
I_obstacle = 0.15
```
コマンドライン引数として指定する場合は、I_ce, I_ca, I_eag, I_ve, I_obstacleの順にスペース区切りで指定します。
```
python do_evaluation_XC2023.py -t [estimation_folder] --est_weight 0.25 0.2 0.25 0.15 0.15
```
