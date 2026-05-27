"""
Script đánh giá và demo Q-Learning Agent đã huấn luyện.

Cách chạy:
    python evaluate.py
    python evaluate.py --episodes 100 --render
"""

import argparse
import time

import gymnasium as gym
import numpy as np

from src.q_learning_agent import QLearningAgent
from src.utils import ACTION_NAMES, decode_state


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Đánh giá Q-Learning Agent đã huấn luyện"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Số episode đánh giá (mặc định: 100)"
    )
    parser.add_argument(
        "--q-table", type=str, default="results/q_table.npy",
        help="Đường dẫn file Q-table (mặc định: results/q_table.npy)"
    )
    parser.add_argument(
        "--render", action="store_true", help="Hiển thị quá trình chạy từng bước"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (mặc định: 0)"
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    """
    Đánh giá agent đã huấn luyện trên nhiều episode.

    Args:
        args: Tham số dòng lệnh.

    Returns:
        Dictionary chứa kết quả đánh giá.
    """
    render_mode = "ansi" if args.render else None
    env = gym.make("Taxi-v3", render_mode=render_mode)

    # Tải agent đã huấn luyện
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
    )
    agent.load_q_table(args.q_table)
    agent.epsilon = 0.0  # Tắt exploration — chỉ khai thác

    print("=" * 60)
    print("    ĐÁNH GIÁ Q-LEARNING AGENT")
    print("=" * 60)
    print(f"  Q-table: {args.q_table}")
    print(f"  Số episode đánh giá: {args.episodes}")
    print(f"  Hiển thị chi tiết: {'Có' if args.render else 'Không'}")
    print("=" * 60)

    total_rewards = []
    total_steps = []
    successes = 0

    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        episode_reward = 0
        steps = 0
        done = False

        if args.render and ep < 3:
            print(f"\n--- Episode {ep + 1} ---")
            state_info = decode_state(state)
            print(f"Trạng thái ban đầu: Taxi tại ({state_info['taxi_row']}, {state_info['taxi_col']})")
            print(f"  Hành khách: {state_info['passenger_status']}")
            print(f"  Đích đến: {state_info['destination_name']}")

        while not done:
            action = agent.choose_best_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if args.render and ep < 3:
                frame = env.render()
                if frame:
                    print(frame)
                print(f"  Hành động: {ACTION_NAMES[action]} | Reward: {reward}")
                time.sleep(0.1)

            state = next_state
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)
        total_steps.append(steps)

        # Episode thành công nếu reward > 0 (trả khách đúng đích)
        if episode_reward > 0:
            successes += 1

    env.close()

    # Kết quả đánh giá
    results = {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "min_reward": np.min(total_rewards),
        "max_reward": np.max(total_rewards),
        "mean_steps": np.mean(total_steps),
        "std_steps": np.std(total_steps),
        "success_rate": successes / args.episodes * 100,
        "total_episodes": args.episodes,
    }

    print("\n" + "=" * 60)
    print("    KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"  Reward trung bình : {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Reward thấp nhất  : {results['min_reward']:.1f}")
    print(f"  Reward cao nhất   : {results['max_reward']:.1f}")
    print(f"  Steps trung bình  : {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
    print(f"  Tỷ lệ thành công  : {results['success_rate']:.1f}%")
    print("=" * 60)

    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
