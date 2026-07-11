# MRI Protocol Optimization with Synthetic Brain Phantoms

Dự án này xây dựng một pipeline mô phỏng MRI tổng hợp để nghiên cứu tối ưu tham số chụp MRI (TR, TE) và huấn luyện một mạng CNN ước lượng giá trị mục tiêu $J$ từ một ảnh MRI cùng với các tham số TR/TE.

Mục tiêu chính là tìm kiếm cấu hình chụp tối ưu cho độ tương phản giữa các mô (ví dụ WM vs GM) thông qua một hàm mục tiêu kết hợp giữa CNR và chi phí thời gian quét.

## Tổng quan

Repository này bao gồm các bước sau:

1. Tạo phantom não tổng hợp dạng label map.
2. Mô phỏng ảnh MRI giả lập theo phương trình spin-echo.
3. Tính toán các chỉ số CNR (Contrast-to-Noise Ratio).
4. Tạo dataset gồm ảnh MRI + tham số TR/TE + giá trị mục tiêu $J$.
5. Huấn luyện CNN để dự đoán $J$.
6. Dùng model đã huấn luyện để đề xuất cặp tham số $(TR^*, TE^*)$ mới.

## Cấu trúc thư mục

- [analysis/](analysis/) – báo cáo phân tích, sweep grid TR/TE, CSV kết quả và các script đánh giá vật lý.
- [app/](app/) – module ứng dụng nhẹ có thể dùng cho giao diện/khởi chạy tiếp theo.
- [data/](data/) – dữ liệu phantom gốc được lưu dưới dạng thư mục phụ.
- [dataset/](dataset/) – script sinh dataset và files ảnh/metadata đã tạo.
- [metrics/](metrics/) – hàm tính CNR và độ nhiễu.
- [models/](models/) – kiến trúc CNN (ResNet-based) dùng cho regression.
- [optimization/](optimization/) – script đề xuất tham số và đánh giá chất lượng đề xuất.
- [phantoms/](phantoms/) – generator phantom tổng hợp.
- [simulator/](simulator/) – mô phỏng tín hiệu MRI spin-echo.
- [training/](training/) – pipeline huấn luyện, dataset PyTorch và checkpoint.
- [tests/](tests/) – test cơ bản cho phantom generator.
- [visualization/](visualization/) – nơi có thể lưu hoặc mở rộng các biểu đồ trực quan hóa.

## Cấu trúc dữ liệu

Dự án sử dụng các cấu trúc dữ liệu sau để nối các bước từ sinh phantom đến huấn luyện và đề xuất tham số:

- [data/phantoms/](data/phantoms/) chứa các thư mục phantom riêng lẻ theo định dạng `phantom_XXXX/`.
  - `phantom.png`: ảnh trực quan hóa phantom bằng màu sắc phân vùng.
  - `label_map.npy`: ma trận nhãn 2D dạng `uint8`, kích thước `256 x 256`.
  - Mã nhãn:
    - `0` = background
    - `1` = White Matter (WM)
    - `2` = Gray Matter (GM)
    - `3` = Cerebrospinal Fluid (CSF)

- [dataset/images/](dataset/images/) chứa ảnh MRI giả lập được lưu theo tên chuẩn:
  - `p{phantom_idx:04d}_tr{TR:04d}_te{TE:03d}.png`
  - Ví dụ: `p0000_tr0200_te010.png`

- [dataset/metadata.csv](dataset/metadata.csv) là bảng dữ liệu tổng hợp với các cột chính:
  - `phantom`: chỉ số phantom
  - `image_path`: đường dẫn tới ảnh tương ứng
  - `TR`: giá trị TR (ms)
  - `TE`: giá trị TE (ms)
  - `CNR_WM_GM`: giá trị CNR giữa WM và GM
  - `J`: hàm mục tiêu dùng cho học máy và tối ưu hóa

- [training/dataset.py](training/dataset.py) định nghĩa `MRIParamDataset`, trả về tuple `(image, tr_te, target)`:
  - `image`: tensor ảnh grayscale có shape `(1, H, W)`
  - `tr_te`: vector gồm 2 giá trị đã chuẩn hóa `[TR_norm, TE_norm]`
  - `target`: scalar `J` đã chuẩn hóa để train regression

## Cấu trúc code-doc / module documentation

Mỗi module trong repo có một vai trò riêng và có thể được đọc theo thứ tự sau để hiểu toàn bộ pipeline:

1. [phantoms/generator.py](phantoms/generator.py)
   - Chứa `PhantomConfig`, `generate_phantom()`, `save_phantom()` và `generate_batch()`.
   - Định nghĩa cách sinh phantom tổng hợp và lưu dưới dạng label map.

2. [simulator/mri_simulator.py](simulator/mri_simulator.py)
   - Chứa `get_phantom_tissue_params()`, `spin_echo_signal()`, `simulate_mri()` và `batch_simulate()`.
   - Là nơi mô phỏng tín hiệu MRI theo công thức spin-echo.

3. [metrics/cnr.py](metrics/cnr.py)
   - Chứa các hàm `tissue_mean()`, `estimate_noise_std()`, `compute_cnr()` và các wrapper `cnr_wm_gm()`, `cnr_gm_csf()`, `cnr_wm_csf()`.
   - Dùng để đánh giá chất lượng tương phản của ảnh.

4. [dataset/generate_dataset.py](dataset/generate_dataset.py)
   - Chứa logic sinh dataset ảnh và ghi metadata cho toàn bộ lưới `(TR, TE)`.

5. [models/cnn_model.py](models/cnn_model.py)
   - Chứa lớp `MRIParamNet`, một mạng regression dùng ảnh + 2 tham số scalar TR/TE để dự đoán `J`.

6. [training/train.py](training/train.py)
   - Chứa pipeline huấn luyện đầy đủ: load dataset, split train/val/test, train loop, evaluate và save checkpoint.

7. [optimization/recommend_params.py](optimization/recommend_params.py)
   - Chứa các hàm `load_model()`, `load_image()`, `recommend()` và `plot_recommendation()`.
   - Được dùng để quét lưới candidate và đề xuất tham số tối ưu.

8. [optimization/evaluate_recommendations.py](optimization/evaluate_recommendations.py)
   - Chứa `get_true_optimum()`, `get_j_at()` và `main()` để so sánh đề xuất của CNN với ground truth.

## Môi trường

Yêu cầu Python 3.9+ và các package trong [requirements.txt](requirements.txt).

Cài đặt:

```bash
pip install -r requirements.txt
```

> Nếu đang dùng GPU, hãy đảm bảo đã cài PyTorch tương thích với CUDA.

## Pipeline chạy nhanh

### 1) Tạo hoặc sử dụng phantom

Thư mục [data/phantoms/](data/phantoms/) đã chứa các phantom ví dụ. Nếu muốn tạo thêm phantom mới, có thể chỉnh sửa và chạy:

```bash
PYTHONPATH=. python3 phantoms/generator.py
```

### 2) Tạo dataset ảnh MRI và metadata

Script này sẽ tạo hàng loạt ảnh MRI cho các cặp $(TR, TE)$ và ghi vào [dataset/metadata.csv](dataset/metadata.csv):

```bash
PYTHONPATH=. python3 dataset/generate_dataset.py
```

### 3) Huấn luyện CNN

Huấn luyện mô hình dự đoán giá trị mục tiêu $J$ từ ảnh MRI và hai tham số TR/TE:

```bash
PYTHONPATH=. python3 training/train.py --epochs 20 --batch_size 64 --device cpu
```

Mô hình được lưu trong [training/checkpoints/](training/checkpoints/).

### 4) Đề xuất tham số mới cho một ảnh đầu vào

Ví dụ sử dụng một ảnh có sẵn trong [dataset/images/](dataset/images/):

```bash
PYTHONPATH=. python3 optimization/recommend_params.py \
  --image dataset/images/p0000_tr0200_te010.png \
  --tr0 200 --te0 10 \
  --checkpoint training/checkpoints/best_model.pt
```

### 5) Đánh giá đề xuất trên tập phantom held-out

```bash
PYTHONPATH=. python3 optimization/evaluate_recommendations.py \
  --test_phantoms 90,91,92,93,94,95,96,97,98,99 \
  --tr0 200 --te0 10
```

Kết quả được lưu trong [optimization/evaluation/](optimization/evaluation/).

## Thành phần cốt lõi

### MRI simulator

File [simulator/mri_simulator.py](simulator/mri_simulator.py) mô phỏng tín hiệu theo công thức:

$$
S = PD \cdot \left(1 - e^{-TR/T1}\right) \cdot e^{-TE/T2}
$$

### CNR metric

File [metrics/cnr.py](metrics/cnr.py) tính các chỉ số contrast-to-noise ratio cho các vùng mô WM, GM, CSF.

### CNN regression model

File [models/cnn_model.py](models/cnn_model.py) dùng backbone ResNet-18 điều chỉnh để nhận ảnh grayscale 1 kênh, kết hợp với hai feature scalar TR/TE và dự đoán giá trị $J$.

## Lưu ý quan trọng

- Dữ liệu trong repo là dữ liệu tổng hợp, không phải ảnh MRI lâm sàng thật.
- Hàm mục tiêu $J$ ở đây là một biến thể giáo dục của CNR có thêm penalty theo thời gian quét.
- Khi đánh giá đề xuất bằng script [optimization/evaluate_recommendations.py](optimization/evaluate_recommendations.py), model chỉ nhận một ảnh duy nhất tại $(TR_0, TE_0)$ và dùng nó để quét toàn bộ lưới candidate $(TR, TE)$; đây là một giả định đơn giản phù hợp cho thử nghiệm, không phải mô hình thu thập thực tế đầy đủ.

## Ghi chú về output

Sau khi chạy các script, các file quan trọng thường gồm:

- [dataset/metadata.csv](dataset/metadata.csv) – bảng chứa ảnh, TR, TE, CNR và $J$.
- [training/checkpoints/](training/checkpoints/) – checkpoint huấn luyện.
- [optimization/evaluation/recommendation_evaluation.csv](optimization/evaluation/recommendation_evaluation.csv) – kết quả đánh giá đề xuất.
- [analysis/report.md](analysis/report.md) – báo cáo phân tích vật lý và tối ưu tham số.

## Test nhanh

```bash
PYTHONPATH=. pytest -q
```

Nếu bạn muốn, tôi có thể tiếp tục viết thêm một README phiên bản ngắn gọn hơn cho GitHub hoặc tạo một file docs/usage.md hướng dẫn từng bước chi tiết hơn.
