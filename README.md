# 🚕 Taxi-v3 Q-Learning — Đồ án Học Tăng Cường

Dự án đồ án môn **Học Tăng Cường (Reinforcement Learning)**: Giải bài toán Taxi-v3 bằng thuật toán **Q-Learning**.

## Mục lục

- [Giới thiệu bài toán](#giới-thiệu-bài-toán)
- [Thuật toán Q-Learning](#thuật-toán-q-learning)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Kết quả](#kết-quả)
- [Giải thích chi tiết](#giải-thích-chi-tiết)

---

## Giới thiệu bài toán

### Môi trường Taxi-v3

Taxi-v3 là một bài toán kinh điển trong Reinforcement Learning, được cung cấp bởi thư viện [Gymnasium](https://gymnasium.farama.org/environments/toy_text/taxi/) (trước đây là OpenAI Gym).

**Mô tả**: Một chiếc taxi hoạt động trên lưới 5×5. Nhiệm vụ là:
1. Di chuyển đến vị trí hành khách
2. Đón hành khách lên xe
3. Di chuyển đến đích
4. Trả hành khách

```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

### Không gian trạng thái và hành động

| Thành phần | Chi tiết |
|---|---|
| **Số trạng thái** | 500 = 5 (hàng) × 5 (cột) × 5 (vị trí khách) × 4 (đích) |
| **Số hành động** | 6: South, North, East, West, Pickup, Dropoff |
| **Reward** | +20 (trả khách đúng), -1 (mỗi bước di chuyển), -10 (pickup/dropoff sai) |

### 4 vị trí đặc biệt
- **R** (Red) — góc trên trái
- **G** (Green) — góc trên phải
- **Y** (Yellow) — góc dưới trái
- **B** (Blue) — góc dưới phải

---

## Thuật toán Q-Learning

### Tổng quan

Q-Learning là thuật toán **off-policy, model-free** thuộc nhóm **Temporal Difference (TD) Learning**. Agent học bằng cách tương tác trực tiếp với môi trường mà không cần biết trước mô hình chuyển trạng thái.

### Công thức cập nhật

```
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]
```

Trong đó:
- `Q(s, a)` — giá trị hành động a tại trạng thái s
- `α` (alpha) — tốc độ học (learning rate), thường = 0.1
- `r` — phần thưởng nhận được
- `γ` (gamma) — hệ số chiết khấu (discount factor), thường = 0.99
- `max_a' Q(s', a')` — giá trị Q lớn nhất tại trạng thái kế tiếp

### Chiến lược ε-greedy

Để cân bằng giữa **khám phá** (exploration) và **khai thác** (exploitation):
- Với xác suất `ε`: chọn hành động **ngẫu nhiên** (khám phá)
- Với xác suất `1-ε`: chọn hành động **tốt nhất** theo Q-table (khai thác)
- `ε` giảm dần theo thời gian: ban đầu khám phá nhiều, sau đó khai thác nhiều hơn

---

## Cấu trúc dự án

```
taxi-v3-qlearning/
├── README.md                    # Tài liệu hướng dẫn (file này)
├── requirements.txt             # Danh sách thư viện cần cài
├── .gitignore                   # File git bỏ qua
│
├── src/                         # Mã nguồn chính
│   ├── __init__.py
│   ├── q_learning_agent.py      # Class Q-Learning Agent
│   └── utils.py                 # Các hàm tiện ích
│
├── train.py                     # Script huấn luyện agent
├── evaluate.py                  # Script đánh giá agent đã huấn luyện
├── visualize.py                 # Script vẽ biểu đồ kết quả
├── compare_algorithms.py        # So sánh Q-Learning vs SARSA vs Random
├── app.py                       # Giao diện web demo (Streamlit)
│
├── notebook/
│   └── taxi_v3_tutorial.ipynb   # Jupyter Notebook hướng dẫn từng bước
│
└── results/                     # Thư mục lưu kết quả (tự động tạo)
    ├── q_table.npy              # Q-table đã huấn luyện
    ├── training_history.json    # Lịch sử huấn luyện
    ├── training_results.png     # Biểu đồ kết quả
    ├── q_table_heatmap.png      # Heatmap Q-table
    └── algorithm_comparison.png # Biểu đồ so sánh thuật toán
```

---

## Cài đặt

### Yêu cầu
- Python 3.9+

### Các bước cài đặt

```bash
# 1. Clone repo
git clone https://github.com/king14052004-crypto/taxi-v3-qlearning.git
cd taxi-v3-qlearning

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

---

## Hướng dẫn sử dụng

### 0. Giao diện Web Demo (Streamlit)

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501` — giao diện web bao gồm:
- **Giới thiệu**: Lý thuyết bài toán + công thức Q-Learning
- **Huấn luyện**: Điều chỉnh hyperparameters và huấn luyện trực tiếp trên web
- **Demo Agent**: Xem agent chơi game từng bước (animation trên bản đồ)
- **Biểu đồ**: Reward, steps, epsilon decay, Q-table heatmap
- **So sánh thuật toán**: Q-Learning vs SARSA vs Random Agent

### 1. Huấn luyện Agent

```bash
# Chạy với tham số mặc định (10,000 episodes)
python train.py

# Tùy chỉnh tham số
python train.py --episodes 20000 --lr 0.1 --gamma 0.99 --epsilon-decay 0.9995
```

**Tham số:**
| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--episodes` | 10000 | Số episode huấn luyện |
| `--lr` | 0.1 | Tốc độ học α |
| `--gamma` | 0.99 | Hệ số chiết khấu γ |
| `--epsilon-start` | 1.0 | Epsilon ban đầu |
| `--epsilon-end` | 0.01 | Epsilon tối thiểu |
| `--epsilon-decay` | 0.9995 | Hệ số giảm epsilon |
| `--seed` | 42 | Random seed |

### 2. Đánh giá Agent

```bash
# Đánh giá trên 100 episodes
python evaluate.py

# Đánh giá với hiển thị chi tiết
python evaluate.py --render --episodes 5
```

### 3. Vẽ biểu đồ

```bash
python visualize.py
```

### 4. So sánh thuật toán

```bash
python compare_algorithms.py
```

### 5. Jupyter Notebook (hướng dẫn từng bước)

```bash
jupyter notebook notebook/taxi_v3_tutorial.ipynb
```

---

## Kết quả

Sau khi huấn luyện 10,000 episodes, agent đạt được:

| Chỉ số | Giá trị |
|---|---|
| Reward trung bình (1000 ep cuối) | ~7.5 - 8.0 |
| Tỷ lệ thành công | ~99% |
| Số bước trung bình | ~13 bước |

Agent học được chính sách gần tối ưu: đón khách và trả khách đúng đích trong số bước ít nhất.

---

## Giải thích chi tiết

### Q-Table là gì?

Q-Table là bảng kích thước `500 × 6` (500 trạng thái × 6 hành động), lưu giá trị Q(s, a) — ước lượng "mức độ tốt" của việc thực hiện hành động `a` tại trạng thái `s`.

Ban đầu, tất cả giá trị Q = 0 (agent không biết gì). Qua quá trình huấn luyện, agent dần học được giá trị Q chính xác → chọn được hành động tối ưu.

### Tại sao cần Epsilon Decay?

- **Ban đầu** (ε = 1.0): Agent khám phá ngẫu nhiên để tìm hiểu môi trường
- **Dần dần** (ε giảm): Agent bắt đầu tin tưởng Q-table và khai thác kiến thức đã học
- **Cuối cùng** (ε ≈ 0.01): Agent gần như luôn chọn hành động tốt nhất, chỉ thỉnh thoảng khám phá

### So sánh Q-Learning và SARSA

| Đặc điểm | Q-Learning | SARSA |
|---|---|---|
| Loại | Off-policy | On-policy |
| Cập nhật | Dùng `max Q(s', a')` | Dùng `Q(s', a')` thực tế |
| Tính chất | Mạnh dạn hơn | Thận trọng hơn |
| Hội tụ | Có thể nhanh hơn | An toàn hơn |

---

## Tài liệu tham khảo

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. [Gymnasium Taxi-v3 Documentation](https://gymnasium.farama.org/environments/toy_text/taxi/)
3. Watkins, C. J. C. H. (1989). *Learning from delayed rewards*. PhD thesis, Cambridge University.
