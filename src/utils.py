"""
Các hàm tiện ích hỗ trợ cho dự án Taxi-v3 Q-Learning.
"""

import json
from pathlib import Path

import numpy as np


# Ý nghĩa các hành động trong Taxi-v3
ACTION_NAMES = {
    0: "Di chuyển xuống (South)",
    1: "Di chuyển lên (North)",
    2: "Di chuyển phải (East)",
    3: "Di chuyển trái (West)",
    4: "Đón khách (Pickup)",
    5: "Trả khách (Dropoff)",
}

# Các vị trí trên bản đồ Taxi-v3
LOCATIONS = {
    0: "R (Đỏ - góc trên trái)",
    1: "G (Xanh lá - góc trên phải)",
    2: "Y (Vàng - góc dưới trái)",
    3: "B (Xanh dương - góc dưới phải)",
}


def decode_state(state: int) -> dict:
    """
    Giải mã trạng thái Taxi-v3 thành các thành phần có ý nghĩa.

    Taxi-v3 có 500 trạng thái = 5 (hàng taxi) × 5 (cột taxi) × 5 (vị trí khách) × 4 (đích đến)

    Args:
        state: Số nguyên đại diện cho trạng thái (0-499).

    Returns:
        Dictionary chứa thông tin giải mã.
    """
    taxi_row = state // 100
    remainder = state % 100
    taxi_col = remainder // 20
    remainder = remainder % 20
    passenger_loc = remainder // 4
    destination = remainder % 4

    passenger_status = (
        "Đang trên xe"
        if passenger_loc == 4
        else f"Tại {LOCATIONS[passenger_loc]}"
    )

    return {
        "taxi_row": taxi_row,
        "taxi_col": taxi_col,
        "passenger_location": passenger_loc,
        "passenger_status": passenger_status,
        "destination": destination,
        "destination_name": LOCATIONS[destination],
    }


def calculate_moving_average(values: list, window: int = 100) -> np.ndarray:
    """
    Tính trung bình trượt (moving average) cho danh sách giá trị.

    Args:
        values: Danh sách giá trị cần tính.
        window: Kích thước cửa sổ trung bình trượt.

    Returns:
        Mảng numpy chứa giá trị trung bình trượt.
    """
    values_array = np.array(values)
    if len(values_array) < window:
        return np.cumsum(values_array) / np.arange(1, len(values_array) + 1)

    cumsum = np.cumsum(np.insert(values_array, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def save_training_history(history: dict, filepath: str) -> None:
    """
    Lưu lịch sử huấn luyện ra file JSON.

    Args:
        history: Dictionary chứa lịch sử huấn luyện.
        filepath: Đường dẫn file lưu.
    """
    serializable = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, list):
            serializable[key] = value
        else:
            serializable[key] = value

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_training_history(filepath: str) -> dict:
    """
    Tải lịch sử huấn luyện từ file JSON.

    Args:
        filepath: Đường dẫn file cần tải.

    Returns:
        Dictionary chứa lịch sử huấn luyện.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def print_episode_info(
    episode: int, reward: float, steps: int, epsilon: float, interval: int = 1000
) -> None:
    """
    In thông tin episode ra console.

    Args:
        episode: Số thứ tự episode.
        reward: Tổng phần thưởng của episode.
        steps: Số bước trong episode.
        epsilon: Giá trị epsilon hiện tại.
        interval: In thông tin mỗi bao nhiêu episode.
    """
    if (episode + 1) % interval == 0:
        print(
            f"Episode {episode + 1:>6d} | "
            f"Reward: {reward:>7.1f} | "
            f"Steps: {steps:>4d} | "
            f"Epsilon: {epsilon:.4f}"
        )
