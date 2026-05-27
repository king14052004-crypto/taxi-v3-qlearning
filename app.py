"""
Giao diện web demo Taxi-v3 Q-Learning với Streamlit.

Chạy:
    streamlit run app.py
"""

import time

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from src.q_learning_agent import QLearningAgent
from src.utils import ACTION_NAMES, LOCATIONS, calculate_moving_average, decode_state

# ============================================================
# Cấu hình trang
# ============================================================
st.set_page_config(
    page_title="Taxi-v3 Q-Learning Demo",
    page_icon="🚕",
    layout="wide",
)

# ============================================================
# CSS tùy chỉnh
# ============================================================
st.markdown("""
<style>
    .taxi-grid {
        font-family: 'Courier New', monospace;
        font-size: 16px;
        line-height: 1.4;
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 16px;
        border-radius: 8px;
        white-space: pre;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .step-info {
        background: #f0f2f6;
        padding: 12px;
        border-radius: 8px;
        margin: 4px 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar — Điều hướng
# ============================================================
st.sidebar.title("🚕 Taxi-v3 Q-Learning")
page = st.sidebar.radio(
    "Chọn trang:",
    [
        "🏠 Giới thiệu",
        "🎓 Huấn luyện",
        "🎮 Demo Agent",
        "📊 Biểu đồ kết quả",
        "⚔️ So sánh thuật toán",
    ],
)


# ============================================================
# Hàm tiện ích dùng chung
# ============================================================

def render_taxi_grid(env_render: str) -> None:
    """Hiển thị bản đồ Taxi-v3 trên giao diện."""
    st.markdown(f'<div class="taxi-grid">{env_render}</div>', unsafe_allow_html=True)


def train_agent_with_progress(
    n_episodes: int,
    lr: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    seed: int,
    progress_bar,
    status_text,
) -> tuple:
    """Huấn luyện agent và cập nhật progress bar."""
    np.random.seed(seed)
    env = gym.make("Taxi-v3")
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=lr,
        discount_factor=gamma,
        epsilon_start=eps_start,
        epsilon_end=eps_end,
        epsilon_decay=eps_decay,
    )

    rewards_history = []
    steps_history = []
    epsilon_history = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        steps_history.append(steps)
        epsilon_history.append(agent.epsilon)

        # Cập nhật progress
        progress = (ep + 1) / n_episodes
        progress_bar.progress(progress)
        if (ep + 1) % max(1, n_episodes // 20) == 0:
            recent = rewards_history[-100:] if len(rewards_history) >= 100 else rewards_history
            status_text.text(
                f"Episode {ep + 1}/{n_episodes} | "
                f"Reward TB (100 ep): {np.mean(recent):.1f} | "
                f"ε = {agent.epsilon:.4f}"
            )

    env.close()
    return agent, rewards_history, steps_history, epsilon_history


# ============================================================
# Trang 1: Giới thiệu
# ============================================================
if page == "🏠 Giới thiệu":
    st.title("🚕 Taxi-v3 Q-Learning")
    st.subheader("Đồ án môn Học Tăng Cường (Reinforcement Learning)")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Bài toán Taxi-v3")
        st.markdown("""
        Một chiếc taxi hoạt động trên lưới **5×5**. Nhiệm vụ:
        1. 🚶 Di chuyển đến vị trí hành khách
        2. 📥 Đón hành khách lên xe
        3. 🚗 Di chuyển đến đích
        4. 📤 Trả hành khách đúng đích

        **Bản đồ:**
        ```
        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+
        ```
        """)

    with col2:
        st.markdown("### 🔢 Thông số")
        st.markdown("""
        | Thành phần | Chi tiết |
        |---|---|
        | **Trạng thái** | 500 = 5×5×5×4 |
        | **Hành động** | 6 (↑↓←→ + Pickup + Dropoff) |
        | **Reward** | +20 (trả đúng), -1 (mỗi bước), -10 (sai) |
        """)

        st.markdown("### 🎯 4 vị trí đặc biệt")
        st.markdown("""
        - 🔴 **R** — góc trên trái
        - 🟢 **G** — góc trên phải
        - 🟡 **Y** — góc dưới trái
        - 🔵 **B** — góc dưới phải
        """)

    st.markdown("---")
    st.markdown("### 📐 Công thức Q-Learning")
    st.latex(r"Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**α (learning rate)** = 0.1\n\nTốc độ học — quyết định mức độ cập nhật Q-value mỗi bước")
    with col2:
        st.info("**γ (discount factor)** = 0.99\n\nHệ số chiết khấu — đánh giá tầm quan trọng của phần thưởng tương lai")
    with col3:
        st.info("**ε (epsilon)** = 1.0 → 0.01\n\nXác suất khám phá — giảm dần để chuyển từ khám phá sang khai thác")

    st.markdown("---")
    st.markdown("### 🔄 Chiến lược ε-greedy")
    st.markdown("""
    | Giai đoạn | Epsilon | Hành vi |
    |---|---|---|
    | Ban đầu | ε = 1.0 | Khám phá 100% — chọn ngẫu nhiên |
    | Giữa | ε ≈ 0.3 | Khám phá 30%, khai thác 70% |
    | Cuối | ε = 0.01 | Gần như luôn chọn tốt nhất |
    """)


# ============================================================
# Trang 2: Huấn luyện
# ============================================================
elif page == "🎓 Huấn luyện":
    st.title("🎓 Huấn luyện Q-Learning Agent")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Hyperparameters")
        n_episodes = st.slider("Số episode", 1000, 50000, 10000, step=1000)
        lr = st.slider("Learning rate (α)", 0.01, 1.0, 0.1, step=0.01)
        gamma = st.slider("Discount factor (γ)", 0.5, 1.0, 0.99, step=0.01)
        eps_start = st.slider("Epsilon ban đầu", 0.1, 1.0, 1.0, step=0.1)
        eps_end = st.slider("Epsilon tối thiểu", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
        eps_decay = st.slider("Epsilon decay", 0.99, 0.9999, 0.9995, step=0.0001, format="%.4f")
        seed = st.number_input("Random seed", value=42, step=1)

        train_btn = st.button("🚀 Bắt đầu huấn luyện", type="primary", use_container_width=True)

    with col2:
        if train_btn:
            st.markdown("### 📈 Quá trình huấn luyện")
            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = time.time()
            agent, rewards, steps, epsilons = train_agent_with_progress(
                n_episodes, lr, gamma, eps_start, eps_end, eps_decay, seed,
                progress_bar, status_text,
            )
            elapsed = time.time() - start_time

            # Lưu vào session state
            st.session_state["agent"] = agent
            st.session_state["rewards"] = rewards
            st.session_state["steps"] = steps
            st.session_state["epsilons"] = epsilons
            st.session_state["train_params"] = {
                "episodes": n_episodes, "lr": lr, "gamma": gamma,
                "eps_start": eps_start, "eps_end": eps_end, "eps_decay": eps_decay,
            }

            progress_bar.progress(1.0)
            status_text.text("✅ Huấn luyện hoàn tất!")

            # Kết quả tổng kết
            st.markdown("### 📊 Kết quả")
            last_1000 = rewards[-1000:]
            last_1000_steps = steps[-1000:]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("⏱️ Thời gian", f"{elapsed:.1f}s")
            m2.metric("🏆 Reward TB (1000 ep cuối)", f"{np.mean(last_1000):.2f}")
            m3.metric("👣 Steps TB", f"{np.mean(last_1000_steps):.1f}")
            m4.metric("🎯 Epsilon cuối", f"{agent.epsilon:.4f}")

            # Biểu đồ nhanh
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            window = 100
            reward_ma = calculate_moving_average(rewards, window)
            ma_x = range(window, len(rewards) + 1)

            ax1.plot(rewards, alpha=0.1, color="steelblue")
            ax1.plot(ma_x, reward_ma, color="darkblue", linewidth=2)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.set_title("Reward theo Episode")
            ax1.axhline(y=0, color="red", linestyle="--", alpha=0.3)

            ax2.plot(epsilons, color="green", linewidth=2)
            ax2.fill_between(range(len(epsilons)), epsilons, alpha=0.2, color="green")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Epsilon")
            ax2.set_title("Epsilon Decay")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.success("Agent đã được huấn luyện! Chuyển sang trang **🎮 Demo Agent** để xem agent chơi.")

        elif "agent" in st.session_state:
            st.info("Agent đã được huấn luyện trước đó. Bấm nút để huấn luyện lại hoặc chuyển sang trang Demo.")
            params = st.session_state.get("train_params", {})
            if params:
                st.json(params)
        else:
            st.info("👈 Điều chỉnh tham số bên trái rồi bấm **Bắt đầu huấn luyện**")

            # Hiển thị bản đồ mẫu
            st.markdown("### 🗺️ Bản đồ Taxi-v3 (mẫu)")
            env = gym.make("Taxi-v3", render_mode="ansi")
            state, _ = env.reset(seed=42)
            frame = env.render()
            render_taxi_grid(frame)
            info = decode_state(state)
            st.markdown(f"""
            **Trạng thái {state}:**
            - Taxi tại: ({info['taxi_row']}, {info['taxi_col']})
            - Hành khách: {info['passenger_status']}
            - Đích đến: {info['destination_name']}
            """)
            env.close()


# ============================================================
# Trang 3: Demo Agent
# ============================================================
elif page == "🎮 Demo Agent":
    st.title("🎮 Demo Agent chơi Taxi-v3")

    if "agent" not in st.session_state:
        st.warning("⚠️ Chưa có agent! Hãy huấn luyện agent trước ở trang **🎓 Huấn luyện**.")

        # Huấn luyện nhanh
        if st.button("⚡ Huấn luyện nhanh (5000 episodes)", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            agent, rewards, steps, epsilons = train_agent_with_progress(
                5000, 0.1, 0.99, 1.0, 0.01, 0.9995, 42,
                progress_bar, status_text,
            )
            st.session_state["agent"] = agent
            st.session_state["rewards"] = rewards
            st.session_state["steps"] = steps
            st.session_state["epsilons"] = epsilons
            st.success("✅ Huấn luyện xong! Bấm nút Demo bên dưới.")
            st.rerun()
    else:
        agent = st.session_state["agent"]
        agent.epsilon = 0.0

        col_ctrl, col_demo = st.columns([1, 2])

        with col_ctrl:
            st.markdown("### ⚙️ Cấu hình Demo")
            demo_seed = st.number_input("Seed (thay đổi để xem scenario khác)", value=42, step=1)
            speed = st.slider("Tốc độ hiển thị (giây/bước)", 0.1, 2.0, 0.5, step=0.1)
            run_demo = st.button("▶️ Chạy Demo", type="primary", use_container_width=True)

        with col_demo:
            if run_demo:
                env = gym.make("Taxi-v3", render_mode="ansi")
                state, _ = env.reset(seed=demo_seed)

                # Thông tin ban đầu
                info = decode_state(state)
                st.markdown("### 🚕 Agent đang chạy...")
                st.markdown(f"""
                **Trạng thái ban đầu:**
                - 🚕 Taxi tại: ({info['taxi_row']}, {info['taxi_col']})
                - 🚶 Hành khách: {info['passenger_status']}
                - 🎯 Đích đến: {info['destination_name']}
                """)

                # Placeholder cho animation
                grid_placeholder = st.empty()
                info_placeholder = st.empty()
                step_log = st.empty()

                total_reward = 0
                step_num = 0
                done = False
                log_entries = []

                while not done and step_num < 50:
                    action = agent.choose_best_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    step_num += 1

                    # Render bản đồ
                    frame = env.render()
                    with grid_placeholder.container():
                        render_taxi_grid(frame)

                    # Thông tin bước
                    state_info = decode_state(next_state)
                    reward_color = "🟢" if reward > 0 else ("🔴" if reward < -1 else "🟡")
                    with info_placeholder.container():
                        st.markdown(f"""
                        **Bước {step_num}:** {ACTION_NAMES[action]}
                        | {reward_color} Reward: **{reward}** | Tổng: **{total_reward}** |
                        Taxi: ({state_info['taxi_row']}, {state_info['taxi_col']}) |
                        Khách: {state_info['passenger_status']}
                        """)

                    log_entries.append(
                        f"Bước {step_num:>2d}: {ACTION_NAMES[action]:<30s} | "
                        f"Reward: {reward:>3d} | Tổng: {total_reward:>4d}"
                    )

                    state = next_state
                    time.sleep(speed)

                env.close()

                # Kết quả cuối
                if total_reward > 0:
                    st.success(f"🎉 Thành công! Tổng reward: **{total_reward}** trong **{step_num}** bước")
                else:
                    st.error(f"❌ Thất bại. Tổng reward: **{total_reward}** trong **{step_num}** bước")

                # Log chi tiết
                with st.expander("📝 Chi tiết từng bước", expanded=False):
                    st.code("\n".join(log_entries))

            else:
                st.info("👈 Bấm **Chạy Demo** để xem agent chơi Taxi-v3")

                # Preview bản đồ
                env = gym.make("Taxi-v3", render_mode="ansi")
                state, _ = env.reset(seed=demo_seed)
                frame = env.render()
                render_taxi_grid(frame)
                info = decode_state(state)
                st.caption(
                    f"Taxi ({info['taxi_row']},{info['taxi_col']}) | "
                    f"Khách: {info['passenger_status']} | "
                    f"Đích: {info['destination_name']}"
                )
                env.close()


# ============================================================
# Trang 4: Biểu đồ kết quả
# ============================================================
elif page == "📊 Biểu đồ kết quả":
    st.title("📊 Biểu đồ kết quả huấn luyện")

    if "rewards" not in st.session_state:
        st.warning("⚠️ Chưa có dữ liệu! Hãy huấn luyện agent trước ở trang **🎓 Huấn luyện**.")

        if st.button("⚡ Huấn luyện nhanh (5000 episodes)", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            agent, rewards, steps, epsilons = train_agent_with_progress(
                5000, 0.1, 0.99, 1.0, 0.01, 0.9995, 42,
                progress_bar, status_text,
            )
            st.session_state["agent"] = agent
            st.session_state["rewards"] = rewards
            st.session_state["steps"] = steps
            st.session_state["epsilons"] = epsilons
            st.success("✅ Huấn luyện xong!")
            st.rerun()
    else:
        rewards = st.session_state["rewards"]
        steps = st.session_state["steps"]
        epsilons = st.session_state["epsilons"]
        window = st.slider("Cửa sổ trung bình trượt", 10, 500, 100, step=10)

        sns.set_theme(style="whitegrid")

        # 4 biểu đồ
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Kết quả Huấn luyện Q-Learning trên Taxi-v3", fontsize=16, fontweight="bold")

        episodes = list(range(1, len(rewards) + 1))
        reward_ma = calculate_moving_average(rewards, window)
        steps_ma = calculate_moving_average(steps, window)
        ma_x = list(range(window, len(rewards) + 1))

        # Reward
        ax = axes[0, 0]
        ax.plot(episodes, rewards, alpha=0.1, color="steelblue")
        ax.plot(ma_x, reward_ma, color="darkblue", linewidth=2, label=f"TB trượt ({window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Reward theo Episode")
        ax.legend()
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.3)

        # Steps
        ax = axes[0, 1]
        ax.plot(episodes, steps, alpha=0.1, color="coral")
        ax.plot(ma_x, steps_ma, color="darkred", linewidth=2, label=f"TB trượt ({window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Số bước")
        ax.set_title("Số bước theo Episode")
        ax.legend()

        # Epsilon
        ax = axes[1, 0]
        ax.plot(episodes, epsilons, color="green", linewidth=2)
        ax.fill_between(episodes, epsilons, alpha=0.2, color="green")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_title("Epsilon Decay")

        # Histogram
        ax = axes[1, 1]
        half = len(rewards) // 2
        ax.hist(rewards[:half], bins=30, alpha=0.6, color="salmon", label=f"Nửa đầu", density=True)
        ax.hist(rewards[half:], bins=30, alpha=0.6, color="steelblue", label=f"Nửa sau", density=True)
        ax.set_xlabel("Reward")
        ax.set_ylabel("Mật độ")
        ax.set_title("Phân phối Reward: Trước vs Sau")
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Q-table heatmap
        if "agent" in st.session_state:
            st.markdown("---")
            st.markdown("### 🗺️ Q-Table Heatmap")
            agent = st.session_state["agent"]
            q_table = agent.q_table

            fig2, ax = plt.subplots(figsize=(10, 8))
            q_range = np.ptp(q_table, axis=1)
            top_states = np.argsort(q_range)[-50:]
            action_labels = ["↓ South", "↑ North", "→ East", "← West", "Pickup", "Dropoff"]
            sns.heatmap(
                q_table[top_states],
                xticklabels=action_labels,
                yticklabels=[str(s) for s in top_states],
                cmap="RdYlGn",
                center=0,
                ax=ax,
            )
            ax.set_xlabel("Hành động")
            ax.set_ylabel("Trạng thái")
            ax.set_title("Q-Table (50 trạng thái có Q-value đa dạng nhất)")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # Thống kê
        st.markdown("---")
        st.markdown("### 📋 Thống kê chi tiết")
        last_n = min(1000, len(rewards))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            | Chỉ số | Giá trị |
            |---|---|
            | Tổng episodes | {len(rewards)} |
            | Reward TB (toàn bộ) | {np.mean(rewards):.2f} |
            | Reward TB ({last_n} ep cuối) | {np.mean(rewards[-last_n:]):.2f} |
            | Reward cao nhất | {max(rewards)} |
            | Reward thấp nhất | {min(rewards)} |
            """)
        with col2:
            st.markdown(f"""
            | Chỉ số | Giá trị |
            |---|---|
            | Steps TB ({last_n} ep cuối) | {np.mean(steps[-last_n:]):.1f} |
            | Steps thấp nhất | {min(steps)} |
            | Epsilon cuối | {epsilons[-1]:.4f} |
            | Tỷ lệ reward > 0 | {sum(1 for r in rewards[-last_n:] if r > 0) / last_n * 100:.1f}% |
            """)


# ============================================================
# Trang 5: So sánh thuật toán
# ============================================================
elif page == "⚔️ So sánh thuật toán":
    st.title("⚔️ So sánh thuật toán RL")
    st.markdown("So sánh hiệu quả giữa **Q-Learning**, **SARSA** và **Random Agent** trên Taxi-v3.")

    col1, col2 = st.columns([1, 3])

    with col1:
        n_eps = st.slider("Số episode", 1000, 20000, 5000, step=1000)
        compare_seed = st.number_input("Seed", value=42, step=1, key="cmp_seed")
        run_compare = st.button("🏁 Chạy so sánh", type="primary", use_container_width=True)

    with col2:
        if run_compare:
            progress = st.progress(0)

            # Random Agent
            st.text("🎲 Đang chạy Random Agent...")
            env = gym.make("Taxi-v3")
            random_rewards = []
            for ep in range(n_eps):
                env.reset(seed=compare_seed + ep)
                total_r = 0
                done = False
                s = 0
                while not done and s < 200:
                    _, r, term, trunc, _ = env.step(env.action_space.sample())
                    done = term or trunc
                    total_r += r
                    s += 1
                random_rewards.append(total_r)
            env.close()
            progress.progress(0.33)

            # Q-Learning
            st.text("🧠 Đang chạy Q-Learning...")
            env = gym.make("Taxi-v3")
            ql_agent = QLearningAgent(env.observation_space.n, env.action_space.n)
            ql_rewards = []
            for ep in range(n_eps):
                state, _ = env.reset(seed=compare_seed + ep)
                total_r = 0
                done = False
                while not done:
                    action = ql_agent.choose_action(state)
                    ns, r, term, trunc, _ = env.step(action)
                    done = term or trunc
                    ql_agent.update(state, action, r, ns)
                    state = ns
                    total_r += r
                ql_agent.decay_epsilon()
                ql_rewards.append(total_r)
            env.close()
            progress.progress(0.66)

            # SARSA
            st.text("📘 Đang chạy SARSA...")
            env = gym.make("Taxi-v3")
            sarsa_q = np.zeros((env.observation_space.n, env.action_space.n))
            sarsa_eps = 1.0
            sarsa_rewards = []
            for ep in range(n_eps):
                state, _ = env.reset(seed=compare_seed + ep)
                action = np.random.randint(env.action_space.n) if np.random.random() < sarsa_eps else int(np.argmax(sarsa_q[state]))
                total_r = 0
                done = False
                while not done:
                    ns, r, term, trunc, _ = env.step(action)
                    done = term or trunc
                    na = np.random.randint(env.action_space.n) if np.random.random() < sarsa_eps else int(np.argmax(sarsa_q[ns]))
                    sarsa_q[state, action] += 0.1 * (r + 0.99 * sarsa_q[ns, na] - sarsa_q[state, action])
                    state = ns
                    action = na
                    total_r += r
                sarsa_eps = max(0.01, sarsa_eps * 0.9995)
                sarsa_rewards.append(total_r)
            env.close()
            progress.progress(1.0)

            # Biểu đồ so sánh
            st.markdown("### 📊 Kết quả so sánh")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("So sánh thuật toán RL trên Taxi-v3", fontsize=14, fontweight="bold")

            window = 100
            r_ma = calculate_moving_average(random_rewards, window)
            q_ma = calculate_moving_average(ql_rewards, window)
            s_ma = calculate_moving_average(sarsa_rewards, window)
            ma_x = range(window, n_eps + 1)

            ax1.plot(ma_x, r_ma, label="Random", color="gray", linewidth=2)
            ax1.plot(ma_x, q_ma, label="Q-Learning", color="blue", linewidth=2)
            ax1.plot(ma_x, s_ma, label="SARSA", color="orange", linewidth=2)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward (TB trượt)")
            ax1.set_title(f"Reward trung bình trượt ({window} episode)")
            ax1.legend()
            ax1.axhline(y=0, color="red", linestyle="--", alpha=0.3)

            last_n = min(1000, n_eps)
            data = [random_rewards[-last_n:], ql_rewards[-last_n:], sarsa_rewards[-last_n:]]
            labels = ["Random", "Q-Learning", "SARSA"]
            bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True)
            colors = ["lightgray", "lightblue", "moccasin"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
            ax2.set_ylabel("Reward")
            ax2.set_title(f"Phân phối Reward ({last_n} episode cuối)")
            ax2.axhline(y=0, color="red", linestyle="--", alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Bảng tổng kết
            st.markdown("### 📋 Bảng tổng kết")
            last_n = min(1000, n_eps)
            st.markdown(f"""
            | Thuật toán | Reward TB (toàn bộ) | Reward TB ({last_n} ep cuối) | Tốt nhất |
            |---|---|---|---|
            | Random | {np.mean(random_rewards):.2f} | {np.mean(random_rewards[-last_n:]):.2f} | {max(random_rewards)} |
            | **Q-Learning** | {np.mean(ql_rewards):.2f} | **{np.mean(ql_rewards[-last_n:]):.2f}** | {max(ql_rewards)} |
            | SARSA | {np.mean(sarsa_rewards):.2f} | {np.mean(sarsa_rewards[-last_n:]):.2f} | {max(sarsa_rewards)} |
            """)

            # Giải thích
            with st.expander("📖 Giải thích sự khác biệt", expanded=True):
                st.markdown("""
                | Đặc điểm | Q-Learning | SARSA |
                |---|---|---|
                | **Loại** | Off-policy | On-policy |
                | **Cập nhật** | Dùng `max Q(s', a')` | Dùng `Q(s', a')` thực tế |
                | **Tính chất** | Mạnh dạn hơn (optimistic) | Thận trọng hơn (conservative) |
                | **Giải thích** | Luôn giả định bước tiếp theo sẽ chọn tốt nhất | Cập nhật dựa trên hành động thực tế đã chọn |

                **Random Agent** chọn hoàn toàn ngẫu nhiên → reward rất âm vì phạt -1 mỗi bước và -10 khi pickup/dropoff sai.
                """)

        else:
            st.info("👈 Chọn số episode và bấm **Chạy so sánh**")

            st.markdown("""
            ### 📖 Về các thuật toán

            **Q-Learning (Off-policy)**
            ```
            Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
            ```
            Cập nhật dựa trên hành động tốt nhất có thể ở bước tiếp theo.

            **SARSA (On-policy)**
            ```
            Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
            ```
            Cập nhật dựa trên hành động thực tế đã chọn ở bước tiếp theo.

            **Random Agent (Baseline)**
            Chọn hành động hoàn toàn ngẫu nhiên — dùng làm baseline để so sánh.
            """)


# ============================================================
# Footer
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Đồ án Học Tăng Cường**\n\n"
    "Taxi-v3 + Q-Learning\n\n"
    "📚 [Gymnasium Docs](https://gymnasium.farama.org/environments/toy_text/taxi/)"
)
