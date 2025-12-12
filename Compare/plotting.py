# plotting.py - Plotting functions for Compare experiments

import matplotlib.pyplot as plt
import numpy as np

# Japanese font support
import matplotlib
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    print("japanize_matplotlib not found. Install with `pip install japanize_matplotlib` for Japanese support.")

matplotlib.rcParams['axes.unicode_minus'] = False


def plot_individual_performance(agent_name, score_history, success_history, window_size=50, save_path=None):
    """
    Plot individual agent performance:
    1. Scores (raw + smoothed)
    2. Success rate (moving average)
    3. Binary success raster plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Scores
    ax1 = axes[0]
    ax1.plot(score_history, alpha=0.6, label='Raw Score')
    if len(score_history) >= window_size:
        avg_scores = [np.mean(score_history[max(0, i-window_size):i+1]) for i in range(len(score_history))]
        ax1.plot(avg_scores, color='red', linewidth=2, label=f'Avg Score (w={window_size})')
    ax1.set_title(f"{agent_name} - Performance (Score)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(np.arange(0, len(score_history)+1, 50))
    
    # Plot 2: Success Rate Moving Average
    ax2 = axes[1]
    if len(success_history) >= window_size:
        avg_success = [np.mean(success_history[max(0, i-window_size):i+1]) for i in range(len(success_history))]
        ax2.plot(avg_success, color='green', linewidth=2, label=f'Success Rate (w={window_size})')
    else:
        ax2.plot(success_history, alpha=0.5, label='Raw Success')
    ax2.set_title(f"{agent_name} - Moving Avg Success Rate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticks(np.arange(0, len(success_history)+1, 50))
    
    # Plot 3: Raster Plot
    ax3 = axes[2]
    indices = [i for i, x in enumerate(success_history) if x >= 0.9]
    if indices:
        ax3.eventplot([indices], lineoffsets=[0], linelengths=0.8, colors=['blue'])
    ax3.set_title(f"{agent_name} - Binary Success Raster")
    ax3.set_xlabel("Episode")
    ax3.set_yticks([])
    ax3.set_xlim(0, len(success_history))
    ax3.grid(True, axis='x')
    ax3.set_xticks(np.arange(0, len(success_history)+1, 50))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparative_dashboard(phase_name, expert_data, apprentices_data, window_size=50, save_path=None):
    """
    Plot comparative dashboard:
    - expert_data: {'name': 'Expert', 'scores': [], 'successes': []}
    - apprentices_data: list of {'name': 'Apprentice X', 'scores': [], 'successes': []}
    """
    if not apprentices_data:
        print("No apprentice data to plot.")
        return

    # Determine max episode length
    max_len = 0
    if expert_data and expert_data.get('scores'):
        max_len = max(max_len, len(expert_data['scores']))
    for app in apprentices_data:
        max_len = max(max_len, len(app.get('scores', [])))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    cmap = plt.get_cmap('tab10')

    # --- Plot 1: Performance (Scores) ---
    ax1 = axes[0]
    if expert_data and expert_data.get('scores'):
        scores = expert_data['scores']
        if len(scores) > window_size:
            smoothed = [np.mean(scores[max(0, i-window_size):i+1]) for i in range(len(scores))]
            ax1.plot(smoothed, color='black', linewidth=2, linestyle='--', label="Expert (Smoothed)")
        else:
            ax1.plot(scores, color='black', linewidth=2, linestyle='--', label="Expert")
    
    for i, app in enumerate(apprentices_data):
        scores = app.get('scores', [])
        if len(scores) > window_size:
            smoothed = [np.mean(scores[max(0, j-window_size):j+1]) for j in range(len(scores))]
            ax1.plot(smoothed, color=cmap(i % 10), label=app['name'])
        else:
            ax1.plot(scores, color=cmap(i % 10), alpha=0.7, label=app['name'])
    
    ax1.set_title(f"{phase_name}: Performance Comparison (Scores)")
    ax1.set_ylabel("Score")
    ax1.set_xlabel("Episode")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    ax1.set_xticks(np.arange(0, max_len + 1, 50))

    # --- Plot 2: Success Rate ---
    ax2 = axes[1]
    if expert_data and expert_data.get('successes'):
        succ = expert_data['successes']
        if len(succ) > window_size:
            smoothed = [np.mean(succ[max(0, i-window_size):i+1]) for i in range(len(succ))]
            ax2.plot(smoothed, color='black', linewidth=2, linestyle='--', label="Expert")
        else:
            ax2.axhline(np.mean(succ), color='black', linestyle='--', label=f"Expert Mean ({np.mean(succ):.2f})")

    for i, app in enumerate(apprentices_data):
        succ = app.get('successes', [])
        if len(succ) > window_size:
            smoothed = [np.mean(succ[max(0, j-window_size):j+1]) for j in range(len(succ))]
            ax2.plot(smoothed, color=cmap(i % 10), label=app['name'])
        else:
            ax2.axhline(np.mean(succ), color=cmap(i % 10), alpha=0.5, linestyle=':', label=f"{app['name']} Mean")

    ax2.set_title(f"{phase_name}: Success Rate Comparison")
    ax2.set_ylabel("Success Rate")
    ax2.set_xlabel("Episode")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    ax2.set_xticks(np.arange(0, max_len + 1, 50))

    # --- Plot 3: Raster Plot ---
    ax3 = axes[2]
    event_collection = []
    colors = []
    labels = []
    
    if expert_data and expert_data.get('successes'):
        indices = [idx for idx, val in enumerate(expert_data['successes']) if val >= 0.9]
        if indices:
            event_collection.append(indices)
            colors.append('black')
            labels.append('Expert')
    
    for i, app in enumerate(apprentices_data):
        succ = app.get('successes', [])
        indices = [idx for idx, val in enumerate(succ) if val >= 0.9]
        if indices:
            event_collection.append(indices)
            colors.append(cmap(i % 10))
            labels.append(app['name'])
    
    if event_collection:
        ax3.eventplot(event_collection, lineoffsets=range(len(event_collection)), 
                      linelengths=0.8, colors=colors)
        for c, l in zip(colors, labels):
            ax3.plot([], [], color=c, label=l)
        ax3.set_yticks(range(len(event_collection)))
        ax3.set_yticklabels(labels)
    
    ax3.set_title(f"{phase_name}: Success Raster Plot")
    ax3.set_xlabel("Episode")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, axis='x')
    ax3.set_xticks(np.arange(0, max_len + 1, 50))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_apprentice_comparison(algorithm_name, apprentice_data_list, window_size=50, save_dir=None):
    """
    Compare Apprentice 1-3 within a single algorithm (TD3 or GAIL).
    Creates 3 separate image files:
    1. Performance (Scores)
    2. Success Rate
    3. Binary Success Raster

    Each image contains a 3x1 grid (rows are Apprentice 1, 2, 3).

    apprentice_data_list: list of dicts with 'name', 'scores', 'successes' for each apprentice (indices 1-3)
    save_dir: Directory Path (Path object or string) to save the images.
    """
    # Filter to only include Apprentice 1, 2, 3 (indices 1, 2, 3 in the list)
    filtered_data = [d for d in apprentice_data_list if d.get('id', -1) in [1, 2, 3]]

    if len(filtered_data) < 3:
        print(f"Not enough apprentice data for comparison (need Apprentice 1-3). Skipping.")
        return

    cmap = plt.get_cmap('tab10')
    max_len = max(len(d.get('scores', [])) for d in filtered_data)

    # helper for common subplot setup
    def _setup_plot(metric_name, save_suffix, ylabel_func, plot_func):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        for row_idx, app_data in enumerate(filtered_data):
            app_name = app_data.get('name', f'Apprentice_{row_idx+1}')
            color = cmap(row_idx % 10)
            ax = axes[row_idx]

            plot_func(ax, app_data, color, window_size, max_len)

            ax.set_title(f"{app_name} - {metric_name}")
            ax.set_ylabel(ylabel_func())
            ax.set_xlabel("Episode")
            if metric_name != "Binary Success Raster":
                ax.grid(True)
                ax.set_xticks(np.arange(0, len(app_data.get('scores', [])) + 1, 50))
            else:
                ax.grid(True, axis='x')
                ax.set_xticks(np.arange(0, max_len + 1, 50))

        fig.suptitle(f"{algorithm_name} Apprentice Comparison (1-3) - {metric_name}", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.95))

        if save_dir:
            save_path = f"{save_dir}/{algorithm_name}_Apprentice_Comparison_{save_suffix}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved {save_path}")
        plt.show()

    # 1. Performance
    def plot_perf(ax, data, color, w, ml):
        scores = data.get('scores', [])
        if len(scores) > w:
            smoothed = [np.mean(scores[max(0, i - w):i + 1]) for i in range(len(scores))]
            ax.plot(smoothed, color=color, linewidth=2)
        else:
            ax.plot(scores, color=color, alpha=0.7)

    _setup_plot("Performance", "Performance", lambda: "Score", plot_perf)

    # 2. Success Rate
    def plot_succ(ax, data, color, w, ml):
        succ = data.get('successes', [])
        if len(succ) > w:
            smoothed = [np.mean(succ[max(0, i - w):i + 1]) for i in range(len(succ))]
            ax.plot(smoothed, color=color, linewidth=2)
        else:
            ax.plot(succ, color=color, alpha=0.5)
        ax.set_ylim(-0.05, 1.05)

    _setup_plot("Success Rate", "Success", lambda: "Rate", plot_succ)

    # 3. Raster
    def plot_raster(ax, data, color, w, ml):
        succ = data.get('successes', [])
        indices = [i for i, x in enumerate(succ) if x >= 0.9]
        if indices:
            ax.eventplot([indices], lineoffsets=[0], linelengths=0.8, colors=[color])
        ax.set_yticks([])
        ax.set_xlim(0, max_len)

    _setup_plot("Binary Success Raster", "Raster", lambda: "", plot_raster)


def plot_cross_algorithm_comparison(td3_data_list, gail_data_list, window_size=50, save_path=None):
    """
    Compare TD3 vs GAIL for each Apprentice (1, 2, 3).

    NEW behavior:
    - For each Apprentice (1,2,3), create ONE figure with 3 rows x 1 column:
      1) Performance (Scores)  : TD3 & GAIL overlaid
      2) Success Rate          : TD3 & GAIL overlaid
      3) Binary Success Raster : TD3 & GAIL in one axis (different y-offsets)
    - This produces 3 figures total (one per Apprentice).
    - If save_path is specified, saves 3 PNG files (one per Apprentice) to the directory of save_path.
    """
    # Filter to only Apprentice 1, 2, 3
    td3_filtered = {d['id']: d for d in td3_data_list if d.get('id', -1) in [1, 2, 3]}
    gail_filtered = {d['id']: d for d in gail_data_list if d.get('id', -1) in [1, 2, 3]}

    apprentice_ids = [1, 2, 3]
    color_td3 = 'tab:blue'
    color_gail = 'tab:orange'
    
    # Handle save directory
    save_dir = None
    if save_path:
        from pathlib import Path
        p = Path(save_path)
        if p.suffix: # If it looks like a file path (has extension)
            save_dir = p.parent
        else:
            save_dir = p
            
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

    for app_id in apprentice_ids:
        td3_data = td3_filtered.get(app_id, {})
        gail_data = gail_filtered.get(app_id, {})

        t_scores = td3_data.get('scores', [])
        g_scores = gail_data.get('scores', [])
        t_succ = td3_data.get('successes', [])
        g_succ = gail_data.get('successes', [])

        # Determine max length for consistent x-axis
        max_len = max(len(t_scores), len(g_scores), len(t_succ), len(g_succ))
        if max_len == 0:
            max_len = 1

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # -------------------------
        # Row 1: Performance (Score)
        # -------------------------
        ax1 = axes[0]

        if len(t_scores) > window_size:
            t_sm = [np.mean(t_scores[max(0, i-window_size):i+1]) for i in range(len(t_scores))]
            ax1.plot(t_sm, color=color_td3, linewidth=2, label='TD3')
        elif len(t_scores) > 0:
            ax1.plot(t_scores, color=color_td3, alpha=0.7, label='TD3')

        if len(g_scores) > window_size:
            g_sm = [np.mean(g_scores[max(0, i-window_size):i+1]) for i in range(len(g_scores))]
            ax1.plot(g_sm, color=color_gail, linewidth=2, label='GAIL')
        elif len(g_scores) > 0:
            ax1.plot(g_scores, color=color_gail, alpha=0.7, label='GAIL')

        ax1.set_title(f"Apprentice {app_id} - Performance (Scores)")
        ax1.set_ylabel("Score")
        ax1.set_xlabel("Episode")
        ax1.grid(True)
        ax1.set_xticks(np.arange(0, max_len + 1, 50))

        # legend only if something plotted
        if (len(t_scores) > 0) or (len(g_scores) > 0):
            ax1.legend(loc='best', fontsize=9)

        # -------------------------
        # Row 2: Success Rate
        # -------------------------
        ax2 = axes[1]

        if len(t_succ) > window_size:
            t_sm = [np.mean(t_succ[max(0, i-window_size):i+1]) for i in range(len(t_succ))]
            ax2.plot(t_sm, color=color_td3, linewidth=2, label='TD3')
        elif len(t_succ) > 0:
            ax2.plot(t_succ, color=color_td3, alpha=0.5, label='TD3')

        if len(g_succ) > window_size:
            g_sm = [np.mean(g_succ[max(0, i-window_size):i+1]) for i in range(len(g_succ))]
            ax2.plot(g_sm, color=color_gail, linewidth=2, label='GAIL')
        elif len(g_succ) > 0:
            ax2.plot(g_succ, color=color_gail, alpha=0.5, label='GAIL')

        ax2.set_title(f"Apprentice {app_id} - Success Rate")
        ax2.set_ylabel("Rate")
        ax2.set_xlabel("Episode")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True)
        ax2.set_xticks(np.arange(0, max_len + 1, 50))

        if (len(t_succ) > 0) or (len(g_succ) > 0):
            ax2.legend(loc='best', fontsize=9)

        # -------------------------
        # Row 3: Binary Success Raster
        # (TD3 and GAIL in ONE axis using different y-offsets)
        # -------------------------
        ax3 = axes[2]

        evt = []
        cols = []
        lbls = []

        t_idx = [i for i, x in enumerate(t_succ) if x >= 0.9]
        if t_idx:
            evt.append(t_idx)
            cols.append(color_td3)
            lbls.append('TD3')

        g_idx = [i for i, x in enumerate(g_succ) if x >= 0.9]
        if g_idx:
            evt.append(g_idx)
            cols.append(color_gail)
            lbls.append('GAIL')

        if evt:
            ax3.eventplot(evt, lineoffsets=range(len(evt)), linelengths=0.8, colors=cols)
            ax3.set_yticks(range(len(evt)))
            ax3.set_yticklabels(lbls)
            # dummy lines for legend
            for c, l in zip(cols, lbls):
                ax3.plot([], [], color=c, label=l)
            ax3.legend(loc='best', fontsize=9)
        else:
            ax3.set_yticks([])
            ax3.set_yticklabels([])

        ax3.set_title(f"Apprentice {app_id} - Binary Success Raster")
        ax3.set_xlabel("Episode")
        ax3.set_xlim(0, max_len)
        ax3.grid(True, axis='x')
        ax3.set_xticks(np.arange(0, max_len + 1, 50))

        fig.suptitle(f"TD3 vs GAIL - Apprentice {app_id} (3×1)", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.95))

        if save_dir:
            final_save_path = save_dir / f"TD3_vs_GAIL_Apprentice_{app_id}_3x1.png"
            plt.savefig(final_save_path, dpi=150, bbox_inches='tight')
            print(f"Saved {final_save_path}")

        plt.show()
