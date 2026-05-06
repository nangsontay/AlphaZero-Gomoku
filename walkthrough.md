# Update: 6/5/2026

## Problem
 **không biết chặn một Open Triple**

## Root Cause
**Spatial Alignment Bug** giữa Input và Target:

1. **Flipped Input:**
   Trong file `game.py`, hàm `current_state()` trích xuất ma trận bàn cờ dưới dạng Tensors, nhưng cuối cùng lại sử dụng lệnh cắt mảng Python `[::-1]` để lật ngược bàn cờ theo chiều dọc (Top thành Bottom).

2. **Mismatched Targets:**
   Trong file training `train_gpu_evaluator.py`, hàm `get_equi_data()` cố gắng sinh thêm dữ liệu bằng cách xoay (rotate) và lật (flip) bàn cờ. Tuy nhiên, nó áp dụng toán tử lật dọc `np.flipud` lên mảng xác suất `mcts_prob` một cách hỗn loạn. Kết quả là mạng Neural được cho ăn một "đầu vào bị lật ngược", nhưng lại yêu cầu dự đoán một "nhãn xác suất không lật ngược".

3. **Mạng Fully Convolutional (ResNet):**
   Mạng Policy Head trong thiết kế mới là mạng tích chập hoàn toàn (1x1 Convolutions). Đặc thù của mạng này là nó bảo toàn vị trí không gian. Nó không thể tự lật vị trí của Feature Map để khớp với cái nhãn sai lệch kia. Hậu quả là mạng Policy Head học ra rác, dự đoán xác suất loạn, phá toàn bộ prior probs của MCTS. 

## Các thay đổi đã thực hiện

### 1. Đồng bộ hóa không gian trong `game.py`
Xóa bỏ việc lật ngược bàn cờ. Bàn cờ gửi vào Neural Network giờ đây khớp 100% tỷ lệ 1:1 với bàn cờ hiển thị cho người chơi.
```diff
- return square_state[:, ::-1, :]
+ return square_state
```

### 2. Chuẩn hóa Data Augmentation trong các file `train*.py`
Cập nhật lại hàm `get_equi_data()` trong cả file:
* `train_gpu_evaluator.py`

Thay vì dùng `np.flipud` chắp vá, chúng tôi đã đổi thuật toán augmentation về dạng đối xứng chuẩn tắc (xoay `np.rot90` và lật ngang `np.fliplr` trên cả State và Probabilities cùng lúc).

```diff
- equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(board_height, board_width)), i)
- extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
+ equi_mcts_prob = np.rot90(mcts_prob.reshape(board_height, board_width), i)
+ extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
```
