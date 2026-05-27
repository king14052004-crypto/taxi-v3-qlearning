"""
Script huấn luyện Q-Learning Agent trên môi trường Taxi-v3.

Cách chạy:
    python train.py
    python train.py --episodes 20000 --lr 0.1 --gamma 0.99

Kết quả sẽ được lưu trong thư mục results/.
"""

import argparse
import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from src.q_learning_agent import QLearningAgent
from src.utils import print_episode_info, save_training_history


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện Q-Learning Agent trên Taxi-v3"
    )
    parser.add_argument(
        "--episodes", type=int, default=10000, help="Số episode huấn luyện (mặc định: 10000)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Tốc độ học α (mặc định: 0.1)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Hệ số chiết khấu γ (mặc định: 0.99)"
    )
    parser.add_argument(
        "--epsilon-start", type=float, default=1.0, help="Epsilon ban đầu (mặc định: 1.0)"
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.01, help="Epsilon tối thiểu (mặc định: 0.01)"
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.9995, help="Tốc độ giảm epsilon (mặc định: 0.9995)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (mặc định: 42)"
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict:
    """
    Huấn luyện Q-Learning agent.

    Quy trình huấn luyện:
    1. Khởi tạo môi trường Taxi-v3 và agent
    2. Lặp qua các episode:
       a. Reset môi trường → nhận trạng thái ban đầu
       b. Agent chọn hành động (ε-greedy)
       c. Thực hiện hành động → nhận reward, trạng thái mới
       d. Cập nhật Q-table
       e. Giảm ε (epsilon decay)
    3. Lưu kết quả

    Args:
        args: Tham số dòng lệnh.

    Returns:
        Dictionary chứa lịch sử huấn luyện.
    """
    # ===== 1. Khởi tạo =====
    np.random.seed(args.seed)
    env = gym.make("Taxi-v3")

    agent = QLearningAgent(
        n_states=env.observation_space.n,       # 500 trạng thái
        n_actions=env.action_space.n,           # 6 hành động
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
    )

    print("=" * 60)
    print("    HUẤN LUYỆN Q-LEARNING AGENT TRÊN TAXI-V3")
    print("=" * 60)
    print(f"  Số episode      : {args.episodes}")
    print(f"  Learning rate α : {args.lr}")
    print(f"  Discount factor γ: {args.gamma}")
    print(f"  Epsilon          : {args.epsilon_start} → {args.epsilon_end}")
    print(f"  Epsilon decay    : {args.epsilon_decay}")
    print(f"  Số trạng thái   : {env.observation_space.n}")
    print(f"  Số hành động    : {env.action_space.n}")
    print("=" * 60)

    # Lưu lịch sử huấn luyện
    rewards_history = []
    steps_history = []
    epsilon_history = []

    start_time = time.time()

    # ===== 2. Vòng lặp huấn luyện =====
    for episode in tqdm(range(args.episodes), desc="Huấn luyện"):
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Agent chọn hành động theo ε-greedy
            action = agent.choose_action(state)

            # Thực hiện hành động trong môi trường
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Cập nhật Q-table
            agent.update(state, action, reward, next_state)

            # Chuyển sang trạng thái mới
            state = next_state
            total_reward += reward
            steps += 1

        # Giảm epsilon sau mỗi episode
        agent.decay_epsilon()

        # Ghi lại lịch sử
        rewards_history.append(total_reward)
        steps_history.append(steps)
        epsilon_history.append(agent.epsilon)

        # In thông tin mỗi 2000 episode
        print_episode_info(episode, total_reward, steps, agent.epsilon, interval=2000)

    elapsed = time.time() - start_time
    env.close()

    # ===== 3. Lưu kết quả =====
    agent.save_q_table("results/q_table.npy")

    history = {
        "rewards": rewards_history,
        "steps": steps_history,
        "epsilons": epsilon_history,
        "hyperparameters": {
            "episodes": args.episodes,
            "learning_rate": args.lr,
            "discount_factor": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "seed": args.seed,
        },
        "training_time_seconds": elapsed,
    }
    save_training_history(history, "results/training_history.json")

    # ===== 4. In kết quả tổng kết =====
    last_1000 = rewards_history[-1000:]
    last_1000_steps = steps_history[-1000:]

    print("\n" + "=" * 60)
    print("    KẾT QUẢ HUẤN LUYỆN")
    print("=" * 60)
    print(f"  Thời gian huấn luyện : {elapsed:.1f} giây")
    print(f"  Reward trung bình (1000 episode cuối): {np.mean(last_1000):.2f}")
    print(f"  Reward cao nhất      : {max(rewards_history):.1f}")
    print(f"  Steps trung bình (1000 episode cuối) : {np.mean(last_1000_steps):.1f}")
    print(f"  Epsilon cuối cùng    : {agent.epsilon:.4f}")
    print(f"  Q-table đã lưu tại  : results/q_table.npy")
    print(f"  Lịch sử huấn luyện  : results/training_history.json")
    print("=" * 60)

    return history


if __name__ == "__main__":
    args = parse_args()
    train(args)
