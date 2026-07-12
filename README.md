# MRI Protocol Optimization with Synthetic Brain Phantoms

Project này xây dựng một pipeline mô phỏng MRI tổng hợp để nghiên cứu tối ưu tham số chụp MRI (TR, TE) và huấn luyện một mạng CNN ước lượng giá trị mục tiêu $J$ từ một ảnh MRI cùng với các tham số TR/TE.

Mục tiêu chính là tìm kiếm cấu hình chụp tốt hơn cho độ tương phản giữa các mô (ví dụ WM vs GM) thông qua một hàm mục tiêu kết hợp giữa CNR và chi phí thời gian quét.

## Tóm tắt pipeline

Repository hiện tại bao gồm các bước sau:

1. Sinh synthetic brain phantom ở dạng label map.
2. Mô phỏng ảnh MRI giả lập bằng mô hình spin-echo.
3. Tính toán các chỉ số CNR (Contrast-to-Noise Ratio).
4. Tạo dataset gồm ảnh MRI, tham số TR/TE và giá trị mục tiêu $J$.
5. Huấn luyện CNN để dự đoán $J$ cho các cặp tham số mới, có thể dùng baseline MSE ([training/train.py](training/train.py)) hoặc training ranking-aware ([training/train_rank.py](training/train_rank.py)).
6. Dùng mô hình đã huấn luyện để đề xuất cặp $(TR^*, TE^*)$ cho một ảnh anchor đầu vào.

## Cấu trúc thư mục

- [analysis/](analysis/) – báo cáo phân tích, sweep grid TR/TE, CSV kết quả và các script đánh giá vật lý.
- [app/](app/) – module ứng dụng nhẹ, có thể mở rộng cho giao diện hoặc luồng chạy tiếp theo.
- [data/](data/) – dữ liệu phantom gốc được lưu dưới dạng thư mục con `phantom_XXXX/`.
- [dataset/](dataset/) – script sinh dataset, ảnh MRI và file metadata.
- [metrics/](metrics/) – hàm tính CNR và ước lượng độ nhiễu.
- [models/](models/) – kiến trúc CNN dùng cho regression.
- [optimization/](optimization/) – script đề xuất tham số và đánh giá chất lượng đề xuất.
- [phantoms/](phantoms/) – generator phantom tổng hợp.
- [simulator/](simulator/) – mô phỏng tín hiệu MRI theo phương trình spin-echo.
- [training/](training/) – pipeline huấn luyện, dataset PyTorch và checkpoint.
- [tests/](tests/) – test cơ bản cho phantom generator.
- [visualization/](visualization/) – nơi có thể lưu hoặc mở rộng các biểu đồ trực quan hóa.

## Môi trường

Yêu cầu Python 3.9+ và các package trong [requirements.txt](requirements.txt).

Cài đặt:

```bash
pip install -r requirements.txt
```

> Nếu dùng GPU, hãy đảm bảo PyTorch phù hợp với phiên bản CUDA/driver hệ thống.

## Quick start

### 1) Sinh phantom

Thư mục [data/phantoms/](data/phantoms/) đã có các phantom ví dụ. Nếu muốn tự sinh thêm, chạy snippet sau:

```bash
PYTHONPATH=. python3 - <<'PY'
from phantoms.generator import generate_batch
generate_batch(n=100, out_root='data/phantoms', base_seed=42)
PY
```

### 2) Tạo dataset ảnh MRI và metadata

Script này sẽ tạo ảnh MRI cho các cặp $(TR, TE)$ và ghi vào [dataset/metadata.csv](dataset/metadata.csv):

```bash
PYTHONPATH=. python3 dataset/generate_dataset.py
```

Kết quả gồm:

- [dataset/images/](dataset/images/) – các ảnh anchor và candidate được lưu theo định dạng `p{phantom_idx:04d}_anchor_tr{TR:04d}_te{TE:03d}.png`
- [dataset/metadata.csv](dataset/metadata.csv) – bảng dữ liệu train/label dùng cho pipeline học máy

### 3) Huấn luyện mô hình

Huấn luyện mô hình dự đoán giá trị mục tiêu $J$ từ ảnh MRI và các tham số đầu vào bằng baseline MSE:

```bash
PYTHONPATH=. python3 training/train.py --epochs 20 --batch_size 64 --device cpu
```

Nếu muốn thử biến thể ranking-aware (listwise / pairwise hinge loss), chạy:

```bash
PYTHONPATH=. python3 training/train_rank.py --epochs 30 --batch_size 8 --device cpu
```

Mô hình tốt nhất được lưu trong [training/checkpoints/](training/checkpoints/).

### 4) Đề xuất tham số mới cho một ảnh đầu vào

Ví dụ dùng một ảnh anchor có sẵn trong [dataset/images/](dataset/images/):

```bash
PYTHONPATH=. python3 optimization/recommend_params.py \
  --image dataset/images/p0000_anchor_tr0200_te010.png \
  --tr0 200 --te0 10 --j0 1.23 \
  --checkpoint training/checkpoints/best_model.pt
```

Lưu ý: `--j0` cần là giá trị $J$ tương ứng với ảnh anchor đầu vào. Trong thực tế nên lấy từ CNR/metric tính trên ảnh đã có segmentation hoặc từ metadata tương ứng.

### 5) Đánh giá đề xuất trên tập phantom held-out

```bash
PYTHONPATH=. python3 optimization/evaluate_recommendations.py \
  --test_phantoms 90,91,92,93,94,95,96,97,98,99 \
  --tr0 200 --te0 10
```

Kết quả được lưu trong [optimization/evaluation/](optimization/evaluation/).

## Dữ liệu và định dạng đầu vào

- [data/phantoms/](data/phantoms/) chứa các thư mục phantom riêng lẻ theo định dạng `phantom_XXXX/`.
  - `phantom.png`: ảnh trực quan hóa phantom bằng màu sắc phân vùng.
  - `label_map.npy`: ma trận nhãn 2D dạng `uint8`, kích thước `256 x 256`.
  - Mã nhãn:
    - `0` = background
    - `1` = White Matter (WM)
    - `2` = Gray Matter (GM)
    - `3` = Cerebrospinal Fluid (CSF)

- [dataset/metadata.csv](dataset/metadata.csv) là bảng dữ liệu tổng hợp với các cột chính:
  - `phantom`: chỉ số phantom
  - `image_path`: đường dẫn tới ảnh tương ứng
  - `TR0`, `TE0`: tham số của ảnh anchor
  - `J0`: giá trị mục tiêu tại anchor
  - `TRc`, `TEc`: tham số candidate
  - `Jc`: giá trị mục tiêu tại candidate

## Các module chính

- [phantoms/generator.py](phantoms/generator.py): sinh phantom tổng hợp và lưu label map.
- [simulator/mri_simulator.py](simulator/mri_simulator.py): mô phỏng tín hiệu MRI theo công thức spin-echo.
- [metrics/cnr.py](metrics/cnr.py): tính các chỉ số CNR cho các vùng WM, GM, CSF.
- [dataset/generate_dataset.py](dataset/generate_dataset.py): sinh dataset ảnh và ghi metadata cho các cặp `(TR, TE)`.
- [models/cnn_model.py](models/cnn_model.py): mô hình CNN regression dùng ảnh + các feature scalar làm đầu vào.
- [training/train.py](training/train.py): pipeline huấn luyện baseline bằng MSE, split theo phantom để tránh data leakage.
- [training/train_rank.py](training/train_rank.py): biến thể huấn luyện ranking-aware dùng listwise loss + pairwise hinge để học thứ tự các candidate tốt hơn.
- [optimization/recommend_params.py](optimization/recommend_params.py): quét lưới candidate và đề xuất tham số tối ưu.
- [optimization/evaluate_recommendations.py](optimization/evaluate_recommendations.py): so sánh đề xuất CNN với ground truth trên tập held-out.

## Lưu ý quan trọng

- Dữ liệu trong repo là dữ liệu tổng hợp, không phải ảnh MRI lâm sàng thật.
- Hàm mục tiêu $J$ ở đây là biến thể giáo dục của CNR có thêm penalty theo thời gian quét.
- Pipeline hiện tại dùng thiết kế anchor-based: mỗi mẫu huấn luyện bao gồm một ảnh anchor và một hoặc nhiều candidate `(TRc, TEc)` để dự đoán `Jc`.
- [training/train.py](training/train.py) chia train/val/test theo phantom, nhằm tránh rò rỉ dữ liệu giữa các tập.

## Test nhanh

```bash
PYTHONPATH=. pytest -q
```

## Output quan trọng

Sau khi chạy các script, các file quan trọng thường gồm:

- [dataset/metadata.csv](dataset/metadata.csv) – bảng chứa ảnh, TR, TE và giá trị $J$.
- [training/checkpoints/](training/checkpoints/) – checkpoint huấn luyện cho baseline và ranking-aware training.
- [training/checkpoints/training_history_rank.csv](training/checkpoints/training_history_rank.csv) – lịch sử loss/metric khi chạy [training/train_rank.py](training/train_rank.py).
- [training/checkpoints/training_summary_rank.json](training/checkpoints/training_summary_rank.json) – summary metrics cho biến thể ranking-aware.
- [optimization/recommendations/](optimization/recommendations/) – heatmap và file kết quả đề xuất tham số.
- [analysis/report.md](analysis/report.md) – báo cáo phân tích vật lý và tối ưu tham số.
