"""
Script vẽ biểu đồ kết quả huấn luyện.

Cách chạy:
    python visualize.py
    python visualize.py --history results/training_history.json
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # Backend không cần GUI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import calculate_moving_average, load_training_history


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ kết quả huấn luyện")
    parser.add_argument(
        "--history", type=str, default="results/training_history.json",
        help="Đường dẫn file lịch sử huấn luyện"
    )
    parser.add_argument(
        "--output", type=str, default="results/training_results.png",
        help="Đường dẫn file ảnh đầu ra"
    )
    parser.add_argument(
        "--window", type=int, default=100,
        help="Kích thước cửa sổ trung bình trượt (mặc định: 100)"
    )
    return parser.parse_args()


def plot_training_results(history: dict, output_path: str, window: int = 100) -> None:
    """
    Vẽ 4 biểu đồ kết quả huấn luyện:
    1. Reward theo episode (với trung bình trượt)
    2. Số bước theo episode (với trung bình trượt)
    3. Epsilon decay
    4. Phân phối reward (histogram)

    Args:
        history: Dictionary chứa lịch sử huấn luyện.
        output_path: Đường dẫn lưu ảnh.
        window: Kích thước cửa sổ trung bình trượt.
    """
    sns.set_theme(style="whitegrid", palette="deep")

    rewards = history["rewards"]
    steps = history["steps"]
    epsilons = history["epsilons"]
    episodes = list(range(1, len(rewards) + 1))

    # Tính trung bình trượt
    reward_ma = calculate_moving_average(rewards, window)
    steps_ma = calculate_moving_average(steps, window)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Kết quả Huấn luyện Q-Learning trên Taxi-v3",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # --- Biểu đồ 1: Reward theo episode ---
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.15, color="steelblue", linewidth=0.5, label="Reward mỗi episode")
    ma_episodes = list(range(window, len(rewards) + 1))
    ax1.plot(ma_episodes, reward_ma, color="darkblue", linewidth=2, label=f"Trung bình trượt ({window} ep)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Tổng Reward")
    ax1.set_title("Reward theo Episode")
    ax1.legend(loc="lower right")
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.3)

    # --- Biểu đồ 2: Số bước theo episode ---
    ax2 = axes[0, 1]
    ax2.plot(episodes, steps, alpha=0.15, color="coral", linewidth=0.5, label="Steps mỗi episode")
    ax2.plot(ma_episodes, steps_ma, color="darkred", linewidth=2, label=f"Trung bình trượt ({window} ep)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Số bước")
    ax2.set_title("Số bước theo Episode")
    ax2.legend(loc="upper right")

    # --- Biểu đồ 3: Epsilon decay ---
    ax3 = axes[1, 0]
    ax3.plot(episodes, epsilons, color="green", linewidth=2)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Epsilon (ε)")
    ax3.set_title("Epsilon Decay (Khám phá → Khai thác)")
    ax3.fill_between(episodes, epsilons, alpha=0.2, color="green")

    # --- Biểu đồ 4: Phân phối reward ---
    ax4 = axes[1, 1]
    # Chia reward thành 2 nửa: đầu vs cuối
    half = len(rewards) // 2
    ax4.hist(
        rewards[:half], bins=30, alpha=0.6, color="salmon",
        label=f"Nửa đầu (ep 1-{half})", density=True
    )
    ax4.hist(
        rewards[half:], bins=30, alpha=0.6, color="steelblue",
        label=f"Nửa sau (ep {half + 1}-{len(rewards)})", density=True
    )
    ax4.set_xlabel("Tổng Reward")
    ax4.set_ylabel("Mật độ")
    ax4.set_title("Phân phối Reward: Trước vs Sau huấn luyện")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Biểu đồ đã lưu tại: {output_path}")
    plt.close()


def plot_q_table_heatmap(q_table_path: str, output_path: str = "results/q_table_heatmap.png") -> None:
    """
    Vẽ heatmap của Q-table (hiển thị một phần các trạng thái).

    Args:
        q_table_path: Đường dẫn file Q-table.
        output_path: Đường dẫn lưu ảnh.
    """
    q_table = np.load(q_table_path)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Chọn 50 trạng thái có Q-value đa dạng nhất để hiển thị
    q_range = np.ptp(q_table, axis=1)  # Range Q-value mỗi trạng thái
    top_states = np.argsort(q_range)[-50:]
    sample_q = q_table[top_states]

    action_labels = ["↓ South", "↑ North", "→ East", "← West", "P Pickup", "D Dropoff"]

    sns.heatmap(
        sample_q,
        xticklabels=action_labels,
        yticklabels=[str(s) for s in top_states],
        cmap="RdYlGn",
        center=0,
        annot=False,
        ax=ax,
    )
    ax.set_xlabel("Hành động")
    ax.set_ylabel("Trạng thái")
    ax.set_title("Q-Table Heatmap (50 trạng thái tiêu biểu)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Q-table heatmap đã lưu tại: {output_path}")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    history = load_training_history(args.history)
    plot_training_results(history, args.output, args.window)
    plot_q_table_heatmap("results/q_table.npy")
