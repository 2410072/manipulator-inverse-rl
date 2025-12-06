# 逆強化学習で連続制御を学習する

panda-gym の PandaReach-v3 環境を用い、7自由度ロボットアームのエージェントを学習させるプロジェクトです。PandaReach-v3 ではシミュレーション内でロボットアームを制御し、目標位置に到達させます。連続制御の高度な手法である Deep Deterministic Policy Gradient (DDPG) と Twin Delayed Deep Deterministic Policy Gradient (TD3) を用いてエキスパートを作成し、さらに P. Abbeel と A. Y. Ng の論文 “Apprenticeship Learning via Inverse Reinforcement Learning" に基づく射影法IRLで弟子エージェントを学習させます。連続制御領域でも弟子エージェントはエキスパートに近い性能を達成し、TD3 を用いた弟子はエキスパートを上回ることも示されました。これは逆強化学習が連続制御タスクに有効であることを示しています。

<p align="center">
  <img src="assets/Trained%20Agent.gif"/>
</p>

## 連続制御の強化学習アルゴリズム

連続値の行動空間（ロボット関節の角度制御など）に対応する強化学習アルゴリズムを用い、状態を連続行動へ写像する方策を学習して期待報酬を最大化します。 

### DDPG

DDPG は連続行動空間向けのアクター・クリティック手法で、ポリシー勾配とQ学習の利点を組み合わせています。アクターネットワークが方策を、クリティックネットワークが行動価値（Q関数）を近似します。アクターは連続行動を直接出力し、それをクリティックが評価して最適行動へ導きます。

### TD3

TD3 は DDPG を拡張し、過大評価バイアスなどを抑えるために2つのクリティックを用いて Q 値を推定します。また、ターゲットネットワークを遅延更新することで学習を安定化させます。DDPG と比べて堅牢で高性能な手法として知られています。

## Hindsight Experience Replay (HER)
HER はスパースかつバイナリな報酬環境で学習を進めるためのテクニックです。多くのロボットタスクでは目標達成がまれで、従来のRLでは学習が進みにくいという課題があります。HERでは失敗エピソードを「別のゴールを達成した」として付け替え、成功・失敗の両方をリプレイバッファに保存することで学習信号を増やし、学習を大幅に改善します。

## 逆強化学習

Apprenticeship Learning via Inverse Reinforcement Learning は、エキスパートのデモから報酬関数を推定し、その報酬をもとにエージェントを最適化する手法です。明示的な報酬を与えず、専門家の軌跡から隠れた報酬関数を推論し、そこから得られる報酬で方策を改善します。

本プロジェクトでは射影法アルゴリズムを用い、エキスパートとエージェントの特徴期待値の差を最大化する重みベクトルを反復的に求めます。この重みを報酬として RL アルゴリズムに渡し、収束するまで繰り返します。十分小さな ϵ 以内で少なくとも1つの弟子エージェントがエキスパートと同等以上の性能を示します。

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