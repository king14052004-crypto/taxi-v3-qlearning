"""
So sánh Q-Learning với Random Agent và SARSA.

Script này giúp sinh viên hiểu sự khác biệt giữa:
- Random Agent: chọn hành động ngẫu nhiên (baseline)
- Q-Learning: thuật toán off-policy TD learning
- SARSA: thuật toán on-policy TD learning

Cách chạy:
    python compare_algorithms.py
"""

import time

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.q_learning_agent import QLearningAgent
from src.utils import calculate_moving_average


class SARSAAgent:
    """
    Agent sử dụng thuật toán SARSA (on-policy TD learning).

    Khác với Q-Learning, SARSA cập nhật Q-value dựa trên hành động
    thực sự được chọn ở bước tiếp theo (không phải hành động tốt nhất):

        Q(s, a) ← Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
                                              ↑
                                    Hành động thực tế chọn (không phải max)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state: int) -> int:
        """Chọn hành động theo ε-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(
        self, state: int, action: int, reward: float, next_state: int, next_action: int
    ) -> None:
        """
        Cập nhật Q-value theo SARSA.

        Điểm khác biệt chính so với Q-Learning:
        - Q-Learning dùng: max_a' Q(s', a')  (off-policy)
        - SARSA dùng:      Q(s', a')         (on-policy)
        """
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        td_target = reward + self.gamma * next_q
        self.q_table[state, action] = current_q + self.lr * (td_target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def run_random_agent(n_episodes: int, seed: int = 42) -> list:
    """Chạy Random Agent làm baseline."""
    env = gym.make("Taxi-v3")
    rewards = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        rewards.append(total_reward)

    env.close()
    return rewards


def run_qlearning(n_episodes: int, seed: int = 42) -> list:
    """Chạy Q-Learning Agent."""
    env = gym.make("Taxi-v3")
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
    )
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)

    env.close()
    return rewards


def run_sarsa(n_episodes: int, seed: int = 42) -> list:
    """Chạy SARSA Agent."""
    env = gym.make("Taxi-v3")
    agent = SARSAAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
    )
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        action = agent.choose_action(state)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)

    env.close()
    return rewards


def main() -> None:
    """Chạy so sánh 3 thuật toán và vẽ biểu đồ."""
    n_episodes = 10000
    window = 100

    print("=" * 60)
    print("    SO SÁNH CÁC THUẬT TOÁN REINFORCEMENT LEARNING")
    print("=" * 60)

    print("\n[1/3] Đang chạy Random Agent...")
    start = time.time()
    random_rewards = run_random_agent(n_episodes)
    print(f"  → Hoàn thành trong {time.time() - start:.1f}s")
    print(f"  → Reward trung bình: {np.mean(random_rewards):.2f}")

    print("\n[2/3] Đang chạy Q-Learning...")
    start = time.time()
    qlearning_rewards = run_qlearning(n_episodes)
    print(f"  → Hoàn thành trong {time.time() - start:.1f}s")
    print(f"  → Reward trung bình (1000 ep cuối): {np.mean(qlearning_rewards[-1000:]):.2f}")

    print("\n[3/3] Đang chạy SARSA...")
    start = time.time()
    sarsa_rewards = run_sarsa(n_episodes)
    print(f"  → Hoàn thành trong {time.time() - start:.1f}s")
    print(f"  → Reward trung bình (1000 ep cuối): {np.mean(sarsa_rewards[-1000:]):.2f}")

    # Vẽ biểu đồ so sánh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("So sánh thuật toán RL trên Taxi-v3", fontsize=14, fontweight="bold")

    # Biểu đồ 1: Reward trung bình trượt
    random_ma = calculate_moving_average(random_rewards, window)
    qlearning_ma = calculate_moving_average(qlearning_rewards, window)
    sarsa_ma = calculate_moving_average(sarsa_rewards, window)

    episodes_ma = range(window, n_episodes + 1)
    ax1.plot(episodes_ma, random_ma, label="Random Agent", color="gray", linewidth=2)
    ax1.plot(episodes_ma, qlearning_ma, label="Q-Learning", color="blue", linewidth=2)
    ax1.plot(episodes_ma, sarsa_ma, label="SARSA", color="orange", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward trung bình trượt")
    ax1.set_title(f"Reward (trung bình trượt {window} episode)")
    ax1.legend()
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.3)

    # Biểu đồ 2: Boxplot so sánh 1000 episode cuối
    last_n = 1000
    data = [random_rewards[-last_n:], qlearning_rewards[-last_n:], sarsa_rewards[-last_n:]]
    labels = ["Random", "Q-Learning", "SARSA"]
    bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True)
    colors = ["lightgray", "lightblue", "lightyellow"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel("Tổng Reward")
    ax2.set_title(f"Phân phối Reward ({last_n} episode cuối)")
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_path = "results/algorithm_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nBiểu đồ so sánh đã lưu tại: {output_path}")
    plt.close()

    # Bảng tổng kết
    print("\n" + "=" * 60)
    print("    BẢNG TỔNG KẾT")
    print("=" * 60)
    print(f"{'Thuật toán':<15} {'Reward TB':>12} {'Reward TB (cuối)':>18} {'Tốt nhất':>10}")
    print("-" * 60)
    print(
        f"{'Random':<15} {np.mean(random_rewards):>12.2f} "
        f"{np.mean(random_rewards[-last_n:]):>18.2f} {max(random_rewards):>10.1f}"
    )
    print(
        f"{'Q-Learning':<15} {np.mean(qlearning_rewards):>12.2f} "
        f"{np.mean(qlearning_rewards[-last_n:]):>18.2f} {max(qlearning_rewards):>10.1f}"
    )
    print(
        f"{'SARSA':<15} {np.mean(sarsa_rewards):>12.2f} "
        f"{np.mean(sarsa_rewards[-last_n:]):>18.2f} {max(sarsa_rewards):>10.1f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
