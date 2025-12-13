# 逆強化学習を用いた連続制御の学習

本研究では、panda-gymツールキットに含まれるPandaReach-v3環境において、7自由度（7 DOF）のロボットアームエージェントの訓練に焦点を当てる。PandaReach-v3タスクでは、シミュレーション環境内で目標オブジェクトに到達するようロボットアームを制御する。本研究では、このタスク向けに高度な連続制御アルゴリズムであるDeep Deterministic Policy Gradient（DDPG）およびTwin Delayed Deep Deterministic Policy Gradient（TD3）を適用してエージェントを訓練する。さらに、P. AbbeelとA. Y. Ngによる論文「Apprenticeship Learning via Inverse Reinforcement Learning」で提案された投影ベースの逆強化学習アルゴリズムを活用し、訓練済みエージェントを専門家エージェントとして用いながら、同一タスクを実行する見習いエージェントを訓練する。見習いエージェントは連続制御領域において効果的に訓練され、単一の見習いエージェント（IRL段階でTD3を用いて訓練）は専門家エージェントの性能をも上回る成果を達成した。この結果は、逆強化学習が連続制御タスクにおいて効果的に適用可能であることを明確に示している。

<p align="center">
  <img src="assets/Trained%20Agent.gif"/>
</p>

## 連続制御問題向け強化学習アルゴリズム

連続強化学習アルゴリズムは、ロボットアームの関節角度制御のように連続的なアクション空間を扱う環境向けに設計されている。これらのアルゴリズムの目的は、観測された状態空間から連続的なアクション空間への効果的なポリシーを発見し、期待報酬の累積を最大化することにある。

### DDPG

DDPG（Deep Deterministic Policy Gradient）は、連続的な行動空間向けに設計されたアクター・クリティック型アルゴリズムである。ポリシー勾配法とQ学習の長所を統合した手法であり、DDPGではアクターネットワークがポリシーを学習する一方、クリティックネットワークが行動価値関数（Q関数）を近似する。アクターネットワークは連続的な行動を直接出力し、これがクリティックネットワークによって評価されることで最適な行動が導出される仕組みとなっている。

### TD3

TD3はDDPGを改良したアルゴリズムであり、過大評価バイアスなどの課題を解決している。Q値推定のためにツインクリティックを導入しており、DDPGが単一のクリティックネットワークを使用するのに対し、TD3では2つのクリティックネットワークを採用している。さらに訓練の安定化を図るため、更新を遅延させたターゲットネットワークも活用している。TD3はその堅牢性とDDPGを上回る性能向上が評価されている。

## 事後的経験再生（Hindsight Experience Replay: HER）

事後的経験再生（Hindsight Experience Replay: HER）は、強化学習（Reinforcement Learning: RL）環境における報酬の稀少性と二値性という課題に対処するために開発された手法である。多くのロボットタスクにおいて、所望の目標を達成することは稀であり、従来のRLアルゴリズムはこのようなフィードバックからの学習に困難を伴う。HERはこの問題に対し、過去の経験を再利用して学習を行う手法として考案された。具体的には、失敗に終わった試行を成功事例として再ラベル付けし、成功事例とともに再生バッファに格納することで、エージェントは成功例だけでなく失敗例からも学習可能となる。これにより、学習プロセスが大幅に改善される効果が得られる。

## 逆強化学習

逆強化学習に基づく見習い学習（Apprenticeship Learning via Inverse Reinforcement Learning）は、強化学習と逆強化学習の原理を統合した手法であり、エージェントが専門家のデモンストレーションから学習することを可能にする。この手法では、エージェントは専門家が提供するデモンストレーションを観察することで、明示的な指示や報酬信号なしにタスクを実行する方法を学習する。直接的に報酬から学習するのではなく、アルゴリズムは専門家のデモンストレーションから背後にある報酬関数を推論し、その推論された報酬関数に基づいてエージェントの行動を最適化する。

この手法を実装する一つの方法として、投影法アルゴリズムが挙げられる。このアルゴリズムは、専門家の行動とエージェントの行動の差異に基づいて、エージェントのポリシーを反復的に改良する。各反復処理において、アルゴリズムは重みベクトルを計算するが、この重みベクトルは専門家の特徴量期待値とエージェントの特徴量期待値を、重みベクトルのノルムに関する制約条件の下で最大限に分離するように設計される。この重みベクトルを用いて報酬を算出し、前述のアルゴリズムに従ってエージェントのポリシーを訓練する。このプロセスは収束するまで繰り返される。少なくとも1体の訓練済み見習いエージェントは、専門家のパフォーマンスをϵの範囲内まで達成することができる。

## 結果

### DDPG

- エキスパートは500エピソードで学習
- 1000エピソードでのエキスパート平均報酬 = -1.768

<p align="center">
  <img src="Results/DDPG/Expert%20Performance.png" width="300" />
  <img src="Results/DDPG/Expert%20Policy.gif" width="350"/>
  <p align="center">Q学習で訓練したCartPoleエキスパート</p>
</p>

#### 弟子エージェント

- IRL アルゴリズムで10体の弟子エージェントを学習。
- 最良の弟子は500エピソードで平均報酬 -1.852 を達成。

<p align="center">
  <img src="Results/DDPG/Apprentice_1%20Performance.png" width="250"/>
  <img src="Results/DDPG/Apprentice_2%20Performance.png" width="250"/>
  <img src="Results/DDPG/Apprentice_3%20Performance.png" width="250"/>

  <img src="Results/DDPG/Apprentice%201%20Policy.gif" width="250"/>
  <img src="Results/DDPG/Apprentice%202%20Policy.gif" width="250" />
  <img src="Results/DDPG/Apprentice%203%20Policy.gif" width="250"/>
</p>

<p align="center">
  <img src="Results/DDPG/Apprentice_7%20Performance.png" width="250" />
  <img src="Results/DDPG/Apprentice_9%20Performance.png" width="250"/>
  <img src="Results/DDPG/Apprentice_10%20Performance.png" width="250"/>

  <img src="Results/DDPG/Apprentice%207%20Policy.gif" width="250"/>
  <img src="Results/DDPG/Apprentice%209%20Policy.gif" width="250"/>
  <img src="Results/DDPG/Apprentice%2010%20Policy.gif" width="250"/>
</p>

### TD3

- エキスパートは500エピソードで学習
- 1000エピソードでのエキスパート平均報酬 = -1.932

<p align="center">
  <img src="Results/TD3/Expert%20Performance.png" width="300" />
  <img src="Results/TD3/Expert%20Policy.gif" width="350"/>
  <p align="center">Q学習で訓練したCartPoleエキスパート</p>
</p>

#### 弟子エージェント

- IRL アルゴリズムで10体の弟子エージェントを学習。
- 最良の弟子はエキスパートを上回り、500エピソードで平均報酬 -1.852 を達成。

<p align="center">
  <img src="Results/TD3/Apprentice_1%20Performance.png" width="250"/>
  <img src="Results/TD3/Apprentice_2%20Performance.png" width="250"/>
  <img src="Results/TD3/Apprentice_3%20Performance.png" width="250"/>

  <img src="Results/TD3/Apprentice%201%20Policy.gif" width="250"/>
  <img src="Results/TD3/Apprentice%202%20Policy.gif" width="250" />
  <img src="Results/TD3/Apprentice%203%20Policy.gif" width="250"/>
</p>

<p align="center">
  <img src="Results/TD3/Apprentice_7%20Performance.png" width="250" />
  <img src="Results/TD3/Apprentice_9%20Performance.png" width="250"/>
  <img src="Results/TD3/Apprentice_10%20Performance.png" width="250"/>

  <img src="Results/TD3/Apprentice%207%20Policy.gif" width="250"/>
  <img src="Results/TD3/Apprentice%209%20Policy.gif" width="250"/>
  <img src="Results/TD3/Apprentice%2010%20Policy.gif" width="250"/>
</p>

## ドキュメント

プロジェクトの概要と実装は [presentation](docs/Learning%20Continuous%20Control%20using%20IRL.pdf) を参照してください。

## 参考文献
- Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, & Daan Wierstra. (2015). Continuous control with deep reinforcement learning.
- Scott Fujimoto, Herke van Hoof, & David Meger (2018). Addressing Function Approximation Error in Actor-Critic Methods. CoRR, abs/1802.09477.
- Quentin Gallouédec, Nicolas Cazin, Emmanuel Dellandréa, & Liming Chen. (2021). panda-gym: Open-source goal-conditioned environments for robotic learning.
- Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, & Wojciech Zaremba. (2017). Hindsight Experience Replay.
- Abbeel, P. & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning.
- Fujimoto, S., Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. In International Conference on Machine Learning (pp. 1582–1591).
- Omkar Chittar. (n.d.). Omkarchittar/manipulator_control_DDPG - GitHub.
