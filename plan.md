# 研究計画書：7自由度ロボットマニピュレータにおける説明可能な多目的軌道・制御最適化フレームワーク

## 1. 研究概要と背景

### 1.1 研究タイトル案
- 英文: Explainable Multi-Objective Trajectory and Control Optimization for 7-DOF Robotic Manipulators Using Evolutionary Algorithms and SHAP Sensitivity Analysis
- 和文: 進化計算とSHAP感度分析を用いた7自由度ロボットマニピュレータの説明可能な多目的軌道・制御最適化

### 1.2 背景と課題
近年、ロボットマニピュレータの制御において深層強化学習（DDPG、TD3、SACなど）や敵対的生成模倣学習（GAIL）が広く研究されている。しかし、これらの手法は数万個のパラメータを持つニューラルネットワークをブラックボックスな方策として用いるため、以下の重大な課題を抱えている。
1. 説明可能性（Explainability）の欠如: エージェントがなぜ特定の軌道や制御出力を選択したのかを人間が物理的に解釈できず、実機導入時の安全性保証が困難である。
2. 多目的トレードオフの調整困難性: 位置決め精度、消費エネルギー、動作の滑らかさ（躍度：Jerk）といった複数の相反する目的を単一のスカラ報酬に統合する必要があり、重み係数の手動調整（Reward Engineering）に膨大な試行錯誤を要する。
3. サンプル効率と再現性の問題: 学習に数万から数十万エピソードを要し、乱数シードによる性能分散が大きい。

### 1.3 本研究のアプローチと学術的貢献（Contributions）
本研究では、ブラックボックスな強化学習や模倣学習に依存せず、物理的意味が明確な少数パラメータによる軌道・制御器の構造化（ホワイトボックス化）と、進化計算・多目的最適化アルゴリズム、および説明可能AI（XAI）手法であるSHAP（SHapley Additive exPlanations）を統合した新しい最適化フレームワークを提案する。

1. 物理的に解釈可能なパラメトリック軌道・制御器の定式化:
   3次B-スプライン軌道の中間制御点、時間スケーリング係数、およびタスク空間の可変インピーダンス（PD）ゲインからなる少数の設計変数ベクトル $\theta \in \mathbb{R}^D$ ($D \approx 12 \sim 15$) を定義し、高い制御精度と解釈性を両立する。
2. 単目的・多目的メタヒューリスティクスの包括的ベンチマーク:
   7自由度Franka Emika Panda環境（panda-gym）において、CMA-ES、PSO、DE、ベイズ最適化、および多目的進化計算（NSGA-II、NSGA-III）を統一された条件下で比較し、位置精度・消費エネルギー・滑らかさの3次元パレート最適フロントを導出する。
3. SHAPおよび大域的感度分析によるパラメータ寄与度の解明:
   最適化過程で得られる評価データに対してサロゲートモデルとSHAP分析（およびSobol感度分析）を適用し、どの物理パラメータがどの目的関数（精度、省エネ、滑らかさ）にどれだけ寄与しているかを定量的に可視化する。
4. SHAP誘導型次元削減（SHAP-Guided Parameter Pruning）による高速化:
   SHAP分析によって特定された低感度（冗長）パラメータを固定し、重要パラメータのみに探索空間を絞り込むことで、制御精度を損なうことなく最適化の収束速度を大幅に向上させる閉ループ改善手法を実証する。
5. 完全な再現性を持つオープンソースソフトウェア（OSS）の公開:
   誰もがワンコマンドで全実験・パレートフロント描画・SHAP分析を再現できるモジュール化されたPythonパッケージをMITライセンスで公開する。

---

## 2. 数学的定式化（Problem Formulation）

### 2.1 設計変数（Decision Variables）
ロボットの軌道形状と制御応答を決定する設計変数ベクトルを $\theta \in \mathbb{R}^D$ とする。
$$\theta = [\Delta p_1^T, \Delta p_2^T, \alpha_t^T, K_p^T, K_d^T]^T$$

- $\Delta p_1, \Delta p_2 \in \mathbb{R}^3$: 初期位置 $p_0$ と目標位置 $g$ を結ぶ直線からの3次元中間ウェイポイント（スプライン制御点）のオフセット（計6変数）
- $\alpha_t \in \mathbb{R}^3$: 軌道の序盤・中盤・終盤における時間進行（速度プロファイル）のスケーリング係数（計3変数）
- $K_p, K_d \in \mathbb{R}^3$: 各軸（X, Y, Z）における追従制御の比例ゲインおよび微分（減衰）ゲイン（計6変数）

### 2.2 目的関数（Objective Functions）と制約条件
エピソード長を $T$ とし、以下の3つの目的関数ベクトル $\mathbf{F}(\theta) = [f_1(\theta), f_2(\theta), f_3(\theta)]^T$ を最小化する。

1. 終端位置誤差および保持精度（Positioning Error）:
   $$f_1(\theta) = \| p(T) - g \|_2 + \lambda_{\text{hold}} \frac{1}{T_{\text{hold}}} \sum_{t=T - T_{\text{hold}}}^{T} \| p(t) - g \|_2$$
2. 制御エネルギーコスト（Control Effort / Energy）:
   $$f_2(\theta) = \frac{1}{T} \sum_{t=0}^{T} \| u(t) \|_2^2$$
3. 動作の滑らかさ / 躍度コスト（Jerk / Smoothness Penalty）:
   $$f_3(\theta) = \frac{1}{T} \sum_{t=1}^{T} \| \ddot{p}(t) - \ddot{p}(t-1) \|_2^2$$
4. 障害物回避制約（Obstacle Penalty / Constraint）:
   障害物中心 $p_{\text{obs}}$、安全半径 $r_{\text{safe}}$ に対し、最小距離 $d_{\min}(\theta) = \min_{t, i} \| p_{\text{link}, i}(t) - p_{\text{obs}} \|_2$ を計算し、ペナルティまたは制約条件 $g_{\text{obs}}(\theta) = \max(0, r_{\text{safe}} - d_{\min}(\theta)) \le 0$ として組み込む。

---

## 3. 検証タスク（3つの実験シナリオ）

単なる逆運動学（IK）で解ける単純な問題設定を避けるため、以下の3段階のタスクで評価を行う。

1. Scenario 1: High-Precision & Minimum-Jerk Reaching（高精度・最小躍度到達タスク）
   - 障害物のない空間において、ミリメートル単位の終端位置精度（$f_1 \le 10^{-3}\text{ m}$）を達成しつつ、消費エネルギー $f_2$ と躍度 $f_3$ を最小化するタスク。
2. Scenario 2: 3D Obstacle-Avoidance Trajectory Optimization（3次元障害物回避タスク）
   - 初期位置と目標位置を結ぶ直線上に球体・直方体の障害物を配置し、手先およびロボットアームのリンク全体が障害物に衝突せずに目標へ到達する軌道を生成するタスク。
3. Scenario 3: Contact-Rich Object Pushing（接触を伴う物体押しタスク：PandaPush-v3）
   - テーブル上のオブジェクトを目標座標まで押して移動させるタスク。接触ダイナミクスにより解析的な勾配計算が不可能であるため、勾配フリーな進化計算アルゴリズムの有効性を実証する。

---

## 4. 評価するアルゴリズムと分析手法

### 4.1 最適化アルゴリズム
- 単目的最適化（重み付き和 $J(\theta) = w_1 f_1 + w_2 f_2 + w_3 f_3 + w_{\text{obs}} g_{\text{obs}}$）:
  1. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
  2. PSO (Particle Swarm Optimization)
  3. DE (Differential Evolution)
  4. BO (Bayesian Optimization with TPE / Gaussian Process)
  5. Baseline RL (TD3 + HER): ブラックボックス手法との比較用ベースライン
- 多目的最適化（パレートフロント探索）:
  1. NSGA-II (Non-dominated Sorting Genetic Algorithm II)
  2. NSGA-III (Reference-point-based NSGA)
  3. MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)

### 4.2 説明可能性（XAI）と感度分析モジュール
1. SHAP分析（TreeSHAP / KernelSHAP）:
   最適化の探索履歴データ $\{(\theta^{(i)}, \mathbf{F}(\theta^{(i)}))\}_{i=1}^N$ を用いて勾配ブースティング決定木（XGBoost / LightGBM）のサロゲートモデルを学習し、各設計変数 $\theta_j$ が各目的関数 $f_1, f_2, f_3$ に与えるSHAP値 $\phi_j$ を算出する。
2. Sobol大域的感度分析（Variance-based Global Sensitivity Analysis）:
   1次感度指数 $S_i$（単独効果）および全感度指数 $S_{Ti}$（パラメータ間の相互作用を含む効果）を算出し、SHAPの結果とクロスバリデーションを行う。
3. SHAP誘導型次元削減（SHAP-Guided Parameter Pruning）:
   平均絶対SHAP値 $\mathbb{E}[|\phi_j|]$ が閾値以下の変数をデフォルト値に固定して探索空間の次元を削減し、再最適化を行った際の収束速度と最終精度を比較検証する。

---

## 5. OSS（GitHubリポジトリ）の設計構成

誰でも実験を再現できる高品質なOSSとして、以下のディレクトリ構造で新規リポジトリを構築する。

    explainable-manipulator-opt/
    ├── README.md                  # プロジェクト概要、インストール方法、ワンコマンド実行手順
    ├── LICENSE                    # MIT License
    ├── pyproject.toml             # 依存関係定義 (panda-gym, pymoo, cma, optuna, shap, SALib, xgboost)
    ├── configs/                   # シナリオ別・アルゴリズム別のハイパーパラメータ設定 (YAML)
    │   ├── scenario1_reach.yaml
    │   ├── scenario2_obstacle.yaml
    │   └── scenario3_push.yaml
    ├── src/
    │   ├── __init__.py
    │   ├── environments/          # panda-gym ラッパー、障害物配置、目的関数 (f1, f2, f3) 計測器
    │   ├── controllers/           # 3次スプライン軌道生成器および可変ゲインPD制御器
    │   ├── optimizers/            # CMA-ES, PSO, DE, BO, NSGA-II, NSGA-III の統一ラッパークラス
    │   └── explainability/        # サロゲート学習、SHAP算出、Sobol感度分析、次元削減モジュール
    ├── scripts/
    │   ├── run_single_obj.py      # 単目的最適化のベンチマーク実行スクリプト (複数シード対応)
    │   ├── run_multi_obj.py       # 多目的最適化およびパレートフロント算出スクリプト
    │   ├── run_shap_pruning.py    # SHAP分析および次元削減（Pruning）による高速化検証スクリプト
    │   └── generate_plots.py      # 論文掲載用の全グラフ・統計表を自動生成するスクリプト
    └── notebooks/
        └── tutorial_demo.ipynb    # 軌道可視化・パレートフロント・SHAPプロットの対話的デモ

---

## 6. 研究・実装の進行スケジュール（全6フェーズ）

### Phase 1: 環境構築とパラメトリック制御器の実装（第1〜2週）
- 新規GitHubリポジトリの作成とPython仮想環境（panda-gym, pybullet, pymoo, cma, optuna, shap, SALib）の構築。
- 設計変数 $\theta$ を受け取り、スプライン軌道を生成してPD制御でロボットを動かすコントローラクラスの実装。
- 3つの目的関数（位置誤差 $f_1$、エネルギー $f_2$、躍度 $f_3$）および障害物との最小距離 $d_{\min}$ を正確に計測・返却する評価関数（Fitness Function）の実装。

### Phase 2: 単目的最適化アルゴリズムの実装とベースライン比較（第3〜4週）
- CMA-ES、PSO、DE、ベイズ最適化の統一インターフェースを実装。
- Scenario 1（Reach）、Scenario 2（Obstacle Avoidance）、Scenario 3（Push）の3環境において、各アルゴリズムを10種類の乱数シードで実行。
- 収束曲線（評価回数 vs 目的関数値）、最終位置誤差（mm）、成功率、計算時間を集計し、統計的検定（Wilcoxon順位和検定）を実施。

### Phase 3: 多目的最適化によるパレートフロントの導出（第5〜6週）
- pymoo を用いて NSGA-II および NSGA-III を実装し、$(f_1, f_2, f_3)$ の3目的同時最適化を実行。
- 得られた非劣解集合（パレートフロント）を3D散布図および2D投影図としてプロット。
- 多目的最適化の評価指標である Hypervolume (HV) および Inverted Generational Distance (IGD) を算出してアルゴリズム間の性能を比較。

### Phase 4: SHAP・Sobol感度分析と次元削減（Pruning）実験（第7〜8週）
- 最適化ログデータを用いてXGBoostサロゲートモデルを訓練し、各目的関数に対するSHAP Summary Plot、Bar Plot、およびDependence Plotを生成。
- 「どの制御点・どのゲインが精度や滑らかさに支配的か」の物理的考察をまとめる。
- SHAP重要度の上位パラメータのみ（例：15変数中8変数）を最適化する「SHAP-Guided CMA-ES / NSGA-II」を実行し、全変数最適化と比較してどれだけ少ない評価回数で同等精度に収束するかを実証。

### Phase 5: OSSの整備とドキュメント化（第9週）
- コードのリファクタリング、型ヒント（Type Hints）とDocstringの追加。
- 論文内のすべての図表をコマンド1つで再生成できるスクリプトの動作確認。
- README.md にインストール手順、使用例、ロボット動作のGIFアニメーション、代表的なパレートフロントとSHAPグラフを掲載。

### Phase 6: IEEE Access 論文執筆と投稿（第10〜12週）
- IEEE AccessのLaTeXテンプレートを用いて論文を執筆（Abstract, Introduction, Related Work, Problem Formulation, Proposed Framework, Experimental Results, Discussion, Conclusion）。
- 図表の視認性（フォントサイズ、カラーブラインド対応配色）と数式の整合性を最終確認し、投稿システムへ提出。

---

## 7. 論文で提示する主要な図表リスト（査読対策）

1. Figure 1: 提案フレームワークの全体アーキテクチャ図（パラメトリック制御器 $\rightarrow$ panda-gym 物理シミュレーション $\rightarrow$ 多目的進化計算 $\rightarrow$ SHAP分析と次元削減フィードバック）。
2. Figure 2: 3つの実験シナリオ（Reaching, Obstacle Avoidance, Pushing）のスナップショットと生成された3次元軌道の比較図。
3. Figure 3: 単目的最適化（CMA-ES, PSO, DE, BO）の収束曲線（10シードの平均と標準偏差バンド）。
4. Figure 4: NSGA-II / NSGA-III によって得られた3次元パレートフロント（Accuracy $f_1$ vs Energy $f_2$ vs Jerk $f_3$）および Knee Point（最適妥協点）の軌道比較。
5. Figure 5: 各目的関数（$f_1, f_2, f_3$）に対する設計変数 $\theta$ の SHAP Beeswarm Summary Plot および平均重要度棒グラフ。
6. Figure 6: パラメータ間の相互作用を示す SHAP Dependence Plot および Sobol 全感度指数 $S_{Ti}$ との比較図。
7. Figure 7: SHAP誘導型次元削減（Full Parameters vs SHAP-Pruned Parameters）による収束速度向上と最終精度維持を示すアブレーション実験グラフ。
8. Table 1: 全シナリオ・全アルゴリズムにおける定量評価結果一覧表（平均値 $\pm$ 標準偏差、成功率、Hypervolume、統計的検定の $p$ 値）。
