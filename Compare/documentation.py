# documentation.py - Dynamic documentation functions for Compare.ipynb
from IPython.display import display, Markdown
import config
def display_td3_intro():
    display(Markdown(rf'''
### Twin Delayed Deep Deterministic Policy Gradient (TD3)

TD3 is an off-policy actor-critic algorithm that addresses estimation errors in DDPG.

**Algorithm Steps:**

1. **Initialization:**
   - Initialize critic networks $Q_{{\phi_1}}, Q_{{\phi_2}}$ and actor network $\pi_\theta$ with random parameters.
   - Initialize target networks $\phi'_1 \leftarrow \phi_1, \phi'_2 \leftarrow \phi_2, \theta' \leftarrow \theta$.
   - Initialize replay buffer $\mathcal{{B}}$.

2. **Interaction:**
   - Select action with exploration noise $a \sim \pi_\theta(s) + \epsilon$, observe reward $r$ and new state $s'$, store transition in $\mathcal{{B}}$.

3. **Training:**
   - Sample mini-batch of $N$ transitions $(s, a, r, s', d)$ from $\mathcal{{B}}$.

4. **Target Action Smoothing:**
   - Compute target action with clipped noise to smooth value estimates:
     $$ \tilde{{a}} \leftarrow \text{{clip}}(\pi_{{\theta'}}(s') + \epsilon, a_{{\text{{min}}}}, a_{{\text{{max}}}}), \quad \epsilon \sim \text{{clip}}(\mathcal{{N}}(0, 0.2), -0.5, 0.5) $$

5. **Target Q-Value Calculation:**
   - Compute target Q-value using the minimum of the two target critics (Clipped Double Q-Learning):
     $$ y \leftarrow r + \gamma \min_{{i=1,2}} Q_{{\phi'_i}}(s', \tilde{{a}}) (1 - d) $$

6. **Critic Update:**
   - Update critics by minimizing the Mean Squared Error (MSE) loss:
     $$ L = \frac{{1}}{{N}} \sum (y - Q_{{\phi_i}}(s,a))^2 $$

7. **Actor Update (Delayed):**
   - If current step $t \mod d = 0$ (default $d={config.UPDATE_ACTOR_EVERY}$), update $\theta$ by deterministic policy gradient:
     $$ \nabla_\theta J(\theta) \approx \frac{{1}}{{N}} \sum \nabla_a Q_{{\phi_1}}(s, a)|_{{a=\pi_\theta(s)}} \nabla_\theta \pi_\theta(s) $$
   - Update target networks using soft updates:
     $$ \theta' \leftarrow \tau \theta + (1-\tau)\theta' $$
     $$ \phi'_i \leftarrow \tau \phi_i + (1-\tau)\phi'_i $$

**Symbol Definitions:**
- $s, a, r, s'$: State, Action, Reward, Next State
- $d$: Done flag (1 if episode terminated, else 0)
- $\pi_\theta$: Actor network with parameters $\theta$
- $Q_{{\phi_i}}$: Critic networks (i=1,2) with parameters $\phi_i$
- $\theta', \phi'_i$: Target network parameters
- $\epsilon$: Exploration noise (Action noise) or Smoothing noise
- $\gamma$: Discount factor ({config.GAMMA})
- $\tau$: Soft update coefficient ({config.TAU})
- $\mathcal{{B}}$: Replay buffer
- $N$: Batch size ({config.BATCH_SIZE})
'''))

def display_td3_eval_intro():
    display(Markdown(rf'''
### Evaluation Algorithm

The Expert agent is evaluated over $M={config.EXPERT_EVAL_EPISODES}$ episodes.

**1. Stochastic Policy:**
Unlike standard evaluation which is often deterministic, this implementation utilizes the exploration policy with Gaussian noise:
$$ a_t = \text{{clip}}(\pi_\theta(s_t) + \epsilon, a_{{\text{{min}}}}, a_{{\text{{max}}}}), \quad \epsilon \sim \mathcal{{N}}(0, {config.NOISE_FACTOR}) $$

**2. Metrics:**
- **Mean Return:** Average cumulative reward.
  $$ \bar{{G}} = \frac{{1}}{{M}} \sum_{{i=1}}^M G_i, \quad G_i = \sum_{{t=0}}^T r_t^{{(i)}} $$

- **Success Rate:** Percentage of episodes where the agent successfully reaches the target (distance < 5cm).
  $$ S = \frac{{1}}{{M}} \sum_{{i=1}}^M \mathbb{{I}}(\text{{success}}^{{(i)}}) \times 100 $$

**Symbol Definitions:**
- $M$: Total number of evaluation episodes ({config.EXPERT_EVAL_EPISODES})
- $G_i$: Cumulative return for episode $i$
- $T$: Total timesteps in an episode
- $r_t^{{(i)}}$: Reward at time prediction step $t$ in episode $i$
- $\mathbb{{I}}(\cdot)$: Indicator function (1 if condition is true, 0 otherwise)
- $\epsilon$: Gaussian noise sampled from $\mathcal{{N}}(0, {config.NOISE_FACTOR})$
'''))

def display_irl_intro():
    display(Markdown(rf'''
### Inverse Reinforcement Learning (Projection Method) for Apprentice Training

Apprentice agents are trained using Inverse Reinforcement Learning (IRL). unlike the Expert which learns from a sparse environment reward, Apprentices learn to optimize a reward function that explains the Expert's behavior.

**Theory & Algorithm:**

The goal is to find a policy $\pi$ whose feature expectations match those of the expert $\mu_E$. We use a game-theoretic approach (Projection Method) to iteratively find such a policy.

**1. Initialization:**
- **Apprentice 0:** We start with an initial policy $\pi^{{(0)}}$ (trained with random reward weights $w^{{(0)}}$).
- Compute its feature expectation $\mu^{{(0)}} = \mathbb{{E}}_{{\pi^{{(0)}}}}[\sum \gamma^t \phi(s_t)]$.
- Set iteration $i = 1$.

**2. Weight Computation (Projection):**
- We seek a new weight vector $w^{{(i)}}$ to maximize the margin between the expert's Feature Expectation and the current set of apprentice Feature Expectations.
- The weight vector determines the reward function for the next apprentice:
  $$ R_w(s) = (w^{{(i)}})^T \phi(s) $$

**3. Apprentice Training (TD3):**
- A new Apprentice agent (initialized from scratch) is trained using the **TD3 Algorithm** (see Section 1) to maximize the synthesized reward $R_w(s)$.
- The goal is to find optimal policy:
  $$ \pi^{{(i)}} = \arg\max_\pi \mathbb{{E}}_{{\pi}} \left[ \sum_{{t=0}}^T \gamma^t (w^{{(i)}})^T \phi(s_t) \right] $$

**4. Feature Expectation Update:**
- Estimate $\mu^{{(i)}}$ by Monte Carlo limit of expected features from $\pi^{{(i)}}$.
- Compute margin $t^{{(i)}}$:
  $$ t^{{(i)}} = \min_{{j<i}} (w^{{(i)}})^T (\mu_E - \mu^{{(j)}}) $$

**5. Termination:**
- If $t^{{(i)}} \le \epsilon_{{\text{{irl}}}}$, convergence is reached. Otherwise, $i \leftarrow i+1$ and repeat from Step 2.

**Symbol Definitions:**
- $\pi^{{(i)}}$: Policy of Apprentice $i$
- $\mu_E$: Feature expectations of the Expert
- $\mu^{{(i)}}$: Feature expectations of Apprentice $i$
- $\phi(s)$: Feature vector of state $s$ (Normalized observation + Goal info)
- $w^{{(i)}}$: Reward weight vector at iteration $i$
- $R_w(s)$: Synthesized reward function used for Apprentice training
- $\epsilon_{{\text{{irl}}}}$: Convergence threshold for the margin ({config.EPSILON})
- $\gamma$: Discount factor ({config.GAMMA})
'''))

def display_apprentice_eval_intro():
    display(Markdown(rf'''
### Apprentice Evaluation Algorithm

Each trained Apprentice agent (0 to {config.NUM_APPRENTICES-1}) is evaluated separately over $N={config.N_EPISODES_APPRENTICE}$ episodes.

**1. Stochastic Policy:**
Evaluations use the same stochastic exploration policy (Gaussian noise) as the Expert to ensure a fair comparison:
$$ a_t = \text{{clip}}(\pi_{{\theta_{{\text{{app}}}}}}(s_t) + \epsilon, a_{{\text{{min}}}}, a_{{\text{{max}}}}), \quad \epsilon \sim \mathcal{{N}}(0, {config.NOISE_FACTOR}) $$

**2. Metrics:**
We use same metrics as the Expert evaluation:
- **Mean Return:** Average cumulative reward (from the *original environment*, not the synthetic IRL reward).
  $$ \bar{{G}} = \frac{{1}}{{N}} \sum_{{i=1}}^N G_i $$

- **Success Rate:** Percentage of successful episodes.
  $$ S = \frac{{1}}{{N}} \sum_{{i=1}}^N \mathbb{{I}}(\text{{success}}^{{(i)}}) \times 100 $$

**Symbol Definitions:**
- $N$: Number of evaluation episodes per apprentice ({config.N_EPISODES_APPRENTICE})
- $\pi_{{\theta_{{\text{{app}}}}}}$: Apprentice policy parameterised by $\theta_{{\text{{app}}}}$
- $\epsilon$: Exploration noise $\mathcal{{N}}(0, {config.NOISE_FACTOR})$
- $G_i$: Cumulative return of episode $i$
- $\mathbb{{I}}(\cdot)$: Indicator function
'''))

def display_td3_comparison_plots_intro():
    display(Markdown(rf'''
### TD3 Comparative Analysis

This section compares the performance of the **TD3 Expert** against the **TD3 Apprentices (1-3)**.
The goal is to verify if the apprentices successfully recover the expert's behavior using the synthesized IRL reward.

**1. Metrics Visualization:**
- **Smoothed Score:** Moving average of episode returns ($w={config.PLOT_WINDOW_SIZE}$).
  $$ \bar{{G}}_t = \frac{{1}}{{w}} \sum_{{i=0}}^{{w-1}} G_{{t-i}} $$
- **Success Rate:** Moving average of binary success indicators.

**2. Comparative Focus:**
- **Stability:** Are apprentice learning curves stable compared to the expert?
- **Convergence:** Do apprentices reach similar final performance levels?

**Symbol Definitions:**
- $\bar{{G}}_t$: Smoothed return at episode $t$
- $w$: Window size ({config.PLOT_WINDOW_SIZE})
'''))

def display_gail_intro():
    display(Markdown(rf'''
### Generative Adversarial Imitation Learning (GAIL)

GAIL is an imitation learning algorithm that leverages Generative Adversarial Networks (GANs).
It uses a discriminator $D_\psi$ to distinguish between Expert state-action pairs and those generated by the Apprentice policy $\pi_\theta$.
The Apprentice learns to maximize a reward signal derived from the discriminator's confusion.

**Algorithm Steps:**

1. **Discriminator Training:**
   - Sample expert batch $(s, a)_E$ and apprentice batch $(s, a)_\pi$ of size $N$.
   - Update Discriminator parameters $\psi$ to minimize cross-entropy loss:
     $$ L_D(\psi) = -\frac{{1}}{{N}} \sum [\log D_\psi(s_E, a_E) + \log(1 - D_\psi(s_\pi, a_\pi))] $$

2. **Reward Calculation:**
   - Compute synthetic reward for the Apprentice based on discriminator output:
     $$ r_{{\text{{gail}}}}(s, a) = \log(D_\psi(s, a)) $$

3. **Apprentice Policy Update (TD3):**
   - The Apprentice is trained using the **TD3 Algorithm** (see Section 1) with the synthetic reward $r_{{\text{{gail}}}}$.
   - The objective is to maximize the expected GAIL reward:
     $$ \max_\theta \mathbb{{E}}_{{\pi_\theta}}[r_{{\text{{gail}}}}(s, a)] $$

**Symbol Definitions:**
- $N$: Batch size for discriminator training ({config.BATCH_SIZE})
- $\pi_\theta$: Apprentice Policy parameters $\theta$
- $D_\psi$: Discriminator network parameters $\psi$
- $(s,a)_E$: State-action pairs sampled from Expert trajectories
- $(s,a)_\pi$: State-action pairs sampled from current Policy trajectories
- $\mathbb{{E}}$: Mathematical expectation
'''))

def display_gail_eval_intro():
    display(Markdown(rf'''
### GAIL Apprentice Evaluation Algorithm

Each trained GAIL Apprentice agent (1 to {config.NUM_APPRENTICES-1}) is evaluated over $N={config.N_EPISODES_APPRENTICE}$ episodes.

**1. Stochastic Policy:**
Evaluations use the same stochastic exploration policy (Gaussian noise) as TD3 apprentices for fairness:
$$ a_t = \text{{clip}}(\pi_{{\theta_{{\text{{gail}}}}}}(s_t) + \epsilon, a_{{\text{{min}}}}, a_{{\text{{max}}}}), \quad \epsilon \sim \mathcal{{N}}(0, {config.NOISE_FACTOR}) $$

**2. Metrics:**
- **Mean Return:** Average cumulative reward (from the *original environment*, not the GAIL discriminator reward).
  $$ \bar{{G}} = \frac{{1}}{{N}} \sum_{{i=1}}^N G_i $$

- **Success Rate:** Percentage of successful episodes.
  $$ S = \frac{{1}}{{N}} \sum_{{i=1}}^N \mathbb{{I}}(\text{{success}}^{{(i)}}) \times 100 $$

**Symbol Definitions:**
- $N$: Number of evaluation episodes ({config.N_EPISODES_APPRENTICE})
- $\pi_{{\theta_{{\text{{gail}}}}}}$: GAIL policy parameterised by $\theta_{{\text{{gail}}}}$
- $\epsilon$: Exploration noise $\mathcal{{N}}(0, {config.NOISE_FACTOR})$
- $G_i$: Cumulative return of episode $i$
- $\mathbb{{I}}(\cdot)$: Indicator function
'''))

def display_gail_comparison_plots_intro():
    display(Markdown(rf'''
### GAIL Comparative Analysis

This section compares the **TD3 Expert** (Baseline) against the **GAIL Apprentices (1-3)**.
GAIL apprentices learn directly from expert state-action pairs without a manually defined reward function.

**1. Metrics Visualization:**
We use the same metrics as in the TD3 comparison for consistency:
- **Smoothed Score:** Moving average of true environment returns ($w={config.PLOT_WINDOW_SIZE}$).
  $$ \bar{{G}}_t = \frac{{1}}{{w}} \sum_{{i=0}}^{{w-1}} G_{{t-i}} $$
- **Success Rate:** Moving average of success rate.

**2. Analysis Points:**
- **Sample Efficiency:** How quickly do GAIL apprentices reach expert performance?
- **Robustness:** Does the adversarial training lead to stable policies?

**Symbol Definitions:**
- $\bar{{G}}_t$: Smoothed return
- $w$: Window size ({config.PLOT_WINDOW_SIZE})
'''))

def display_final_comparison_intro():
    display(Markdown(rf'''
### Visual Analysis & Metrics

To analyze and compare the stability and performance of the trained agents (`TD3 Expert`, `Apprentices 1-3`), we visualize the following metrics using comparative dashboards.

**1. Smoothing (Moving Average):**
To reduce variance in the raw episode returns and success rates, we apply a simple moving average (SMA) with a window formulation:
$$ \bar{{x}}_t = \frac{{1}}{{w}} \sum_{{i=0}}^{{w-1}} x_{{t-i}} $$

**2. Comparative Plots:**
- **Performance (Score):** Tracks the cumulative return per episode. Smoothed lines (solid) are overlaid on raw data (faded).
  $$ G_t = \sum_{{k=0}}^T r_k $$
- **Success Rate:** Tracks the binary success outcome ($ 1 $ if target reached, else $ 0 $), averaged over the window $ w $.
- **Raster Plot:** A discrete visualization where each vertical bar represents a successful episode. Dense regions indicate consistent high performance.

**Symbol Definitions:**
- $\bar{{x}}_t$: Smoothed value at episode $ t $
- $x_t$: Raw metric value (Return or Success) at episode $ t $
- $w$: Smoothing window size ($ w={config.PLOT_WINDOW_SIZE} $)
- $G_t$: Cumulative return (Score) for episode $ t $
'''))

