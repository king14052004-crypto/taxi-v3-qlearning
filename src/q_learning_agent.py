"""
Q-Learning Agent cho bài toán Taxi-v3.

Q-Learning là thuật toán off-policy, temporal-difference (TD) learning.
Agent cập nhật Q-table theo công thức:
    Q(s, a) = Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]

Trong đó:
    - s: trạng thái hiện tại
    - a: hành động được chọn
    - r: phần thưởng nhận được
    - s': trạng thái kế tiếp
    - α (alpha): tốc độ học (learning rate)
    - γ (gamma): hệ số chiết khấu (discount factor)
    - ε (epsilon): xác suất khám phá (exploration rate)
"""

import numpy as np


class QLearningAgent:
    """Agent sử dụng thuật toán Q-Learning để học chính sách tối ưu."""

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
        """
        Khởi tạo agent.

        Args:
            n_states: Số lượng trạng thái trong môi trường.
            n_actions: Số lượng hành động có thể thực hiện.
            learning_rate: Tốc độ học α — quyết định mức độ cập nhật Q-value.
            discount_factor: Hệ số chiết khấu γ — đánh giá tầm quan trọng
                             của phần thưởng tương lai.
            epsilon_start: Giá trị ε ban đầu (khám phá nhiều).
            epsilon_end: Giá trị ε tối thiểu (khai thác nhiều hơn).
            epsilon_decay: Tốc độ giảm ε sau mỗi episode.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Khởi tạo Q-table = bảng giá trị Q(s, a) với tất cả bằng 0
        # Kích thước: (số trạng thái) x (số hành động)
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state: int) -> int:
        """
        Chọn hành động theo chiến lược ε-greedy.

        Với xác suất ε: chọn hành động ngẫu nhiên (exploration - khám phá).
        Với xác suất 1-ε: chọn hành động tốt nhất (exploitation - khai thác).

        Args:
            state: Trạng thái hiện tại.

        Returns:
            Hành động được chọn.
        """
        if np.random.random() < self.epsilon:
            # Khám phá: chọn hành động ngẫu nhiên
            return np.random.randint(self.n_actions)
        else:
            # Khai thác: chọn hành động có Q-value cao nhất
            return int(np.argmax(self.q_table[state]))

    def choose_best_action(self, state: int) -> int:
        """
        Luôn chọn hành động tốt nhất (greedy) — dùng khi đánh giá.

        Args:
            state: Trạng thái hiện tại.

        Returns:
            Hành động có Q-value cao nhất.
        """
        return int(np.argmax(self.q_table[state]))

    def update(
        self, state: int, action: int, reward: float, next_state: int
    ) -> float:
        """
        Cập nhật Q-value theo công thức Q-Learning.

        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Trạng thái hiện tại.
            action: Hành động đã thực hiện.
            reward: Phần thưởng nhận được.
            next_state: Trạng thái kế tiếp.

        Returns:
            Giá trị TD error (sai số temporal difference).
        """
        # Giá trị Q hiện tại
        current_q = self.q_table[state, action]

        # Giá trị Q tốt nhất ở trạng thái kế tiếp
        best_next_q = np.max(self.q_table[next_state])

        # TD target = r + γ * max Q(s', a')
        td_target = reward + self.gamma * best_next_q

        # TD error = TD target - Q hiện tại
        td_error = td_target - current_q

        # Cập nhật Q-value
        self.q_table[state, action] = current_q + self.lr * td_error

        return td_error

    def decay_epsilon(self) -> None:
        """Giảm ε theo hệ số decay, nhưng không thấp hơn epsilon_end."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filepath: str) -> None:
        """Lưu Q-table ra file."""
        np.save(filepath, self.q_table)

    def load_q_table(self, filepath: str) -> None:
        """Tải Q-table từ file."""
        self.q_table = np.load(filepath)
