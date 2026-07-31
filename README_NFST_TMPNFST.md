# NFST & TMPNFST - Hướng dẫn nhanh

## 1. Cấu trúc file

```text
adbench/baseline/
|-- NFST/
|   |-- __init__.py       - Xuất các class public `NFST`, `NFSTModel`.
|   |-- run.py            - Adapter kết nối NFST với ADBench.
|   |-- model.py          - Điều phối toàn bộ quá trình fit và predict.
|   |-- anchor_mapping.py - Tạo anchor và ánh xạ dữ liệu vào anchor-space.
|   |-- scatter.py        - Tạo graph, nhóm con và các scatter matrix.
|   |-- solver.py         - Tìm không gian chiếu NFST.
|   `-- scoring.py        - Tạo centroid và tính anomaly score.
`-- TMPNFST/
    |-- __init__.py       - Xuất các class public `TMPNFST`, `TMPNFSTModel`.
    |-- run.py            - Adapter kết nối TMPNFST với ADBench.
    |-- model.py          - Điều phối phân nhóm, NFST và predict.
    |-- partition.py      - Tạo k-NN graph và pseudo-class bằng spectral clustering.
    |-- scatter.py        - Tính scatter matrix từ pseudo-class.
    `-- scoring.py        - Tạo centroid và tính anomaly score.
```

## 2. Tham số quan trọng

### 2.1 NFST

| Tham số | Ảnh hưởng chính |
|---|---|
| `n_anchors` | Số anchor; ảnh hưởng độ chi tiết, tốc độ và chất lượng nhóm con. |
| `sigma` | Độ rộng Gaussian; quá nhỏ/lớn làm membership kém phân biệt. |
| `alpha` | Cân bằng cấu trúc global và local. |
| `random_state` | Ảnh hưởng MiniBatch K-means và khả năng tái lập. |
| `rank_tolerance` | Quyết định rank số học của total scatter. |
| `null_tolerance` | Quyết định eigenvalue nào được xem là null. |
| `selection_mode` | `null` có thể fail nếu nullity bằng 0; `smallest` thực dụng hơn. |
| `max_components` | Giới hạn số chiều chiếu được giữ lại. |
| `max_graph_bytes` | Dừng sớm nếu dense graph vượt giới hạn RAM. |

### 2.2 TMPNFST

| Tham số | Ảnh hưởng chính |
|---|---|
| `n_neighbors` | Độ liên thông của k-NN graph; quá nhỏ dễ tạo graph rời rạc. |
| `n_components` | Số pseudo-class cần chia. |
| `sigma` | Độ mạnh của cạnh Gaussian trong graph. |
| `random_state` / seed | Ảnh hưởng split, eigensolver và K-means. |
| `n_init` | Số lần khởi tạo K-means trong spectral-space. |
| `eigen_tolerance` | Độ chính xác ARPACK; quá chặt có thể khó hội tụ. |
| `eigen_maxiter` | Số vòng lặp tối đa; quá thấp có thể làm seed fail. |
| `rank_tolerance`, `null_tolerance` | Điều khiển rank/null-space ở phase NFST. |
| `selection_mode`, `max_components` | Cách chọn và số chiều chiếu cuối cùng. |

Cấu hình TMPNFST benchmark hiện tại: `n_neighbors=15`, `n_components=3`,
`eigen_tolerance=1e-6`, `eigen_maxiter=50000`, `selection_mode="smallest"`,
`max_components=2`.

## 3. Metric đánh giá

| Metric | Ý nghĩa ngắn |
|---|---|
| AUC-PR | Metric xếp hạng chính; phù hợp dữ liệu có ít anomaly. Cao hơn tốt hơn. |
| AUC-ROC | Khả năng xếp anomaly cao hơn normal trên nhiều threshold. Cao hơn tốt hơn. |
| Fit time | Thời gian huấn luyện model. Thấp hơn nhanh hơn. |
| Inference time | Thời gian chấm điểm test. Thấp hơn nhanh hơn. |
| Failed runs | Số seed không hoàn thành do lỗi số học, bộ nhớ hoặc dữ liệu. |

Ranking dùng **Mean AUC-PR**, hòa điểm thì dùng **Mean AUC-ROC**.

## 4. Các loại dataset trong ADBench

- **Loại A - Classical/tabular:** 47 dataset, dữ liệu bảng số học truyền thống.
- **Loại B - Computer Vision:** 61 dataset, mỗi dataset có embedding ResNet18 và ViT
  (`122` file `.npz`).
- **Loại C - NLP/text:** 13 dataset, mỗi dataset có embedding BERT và RoBERTa
  (`26` file `.npz`).

Tổng cộng: **121 bộ dữ liệu gốc**, được lưu thành **195 file `.npz`**.
Benchmark hiện tại chỉ dùng nhóm **Classical**.

## 5. Benchmark hiện tại

### 5.1 Chín model

`NFST`, `TMPNFST`, `IForest`, `OCSVM`, `COPOD`, `ECOD`, `HBOS`, `KNN`, `LOF`.

### 5.2 Chín dataset

| Dataset | Mẫu | Features | Anomaly | Train/Test dự kiến |
|---|---:|---:|---:|---:|
| Hepatitis | 80 | 19 | 16.25% | 56 / 24 |
| Ionosphere | 351 | 32 | 35.90% | 245 / 106 |
| Vowels | 1,456 | 12 | 3.43% | 1,019 / 437 |
| Cardio | 1,831 | 21 | 9.61% | 1,281 / 550 |
| InternetAds | 1,966 | 1,555 | 18.72% | 1,376 / 590 |
| Fault | 1,941 | 27 | 34.67% | 1,358 / 583 |
| Glass | 214 | 7 | 4.21% | 149 / 65 |
| Yeast | 1,484 | 8 | 34.16% | 1,038 / 446 |
| WDBC | 367 | 30 | 2.72% | 256 / 111 |

- Split phân tầng: **70% train / 30% test**.
- MinMaxScaler chỉ fit trên train, sau đó áp dụng cho train và test.
- Chạy các seed **1, 2, 3**; mỗi seed tạo split khác nhau nhưng giữ cùng tỷ lệ.
- Tổng số lần chạy: `9 dataset × 9 model × 3 seed = 243`.
- Các detector chạy unsupervised; `y_train` không được dùng làm nhãn thật để fit.

## 6. Command chạy benchmark

### 6.1 Chạy lần đầu trên Windows

Yêu cầu: có Internet và Windows Package Manager (`winget`, thường có sẵn trên
Windows 10/11). Mở **PowerShell ngay tại thư mục gốc `ADBench`**, sau đó copy
toàn bộ block dưới đây:

```powershell
# 1. Cài đúng Python 3.10.11 nếu máy chưa có
$hasPython310 = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.10 -c "import sys; raise SystemExit(0 if sys.version_info[:3] == (3,10,11) else 1)"
    $hasPython310 = ($LASTEXITCODE -eq 0)
}
if (-not $hasPython310) {
    winget install --exact --id Python.Python.3.10 --version 3.10.11 --scope user --accept-package-agreements --accept-source-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
}

# 2. Tạo môi trường riêng .venv và cài dependency đã được repo xác minh
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -EnvironmentPath .\.venv
}

# 3. Kiểm tra môi trường rồi chạy benchmark 9 dataset × 9 model × 3 seed
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe .\scripts\compare_nfst_tmpnfst_7_models.py
```

Lần cài đầu có thể lâu và cần vài GB dung lượng vì ADBench có TensorFlow, Torch
và các baseline khác. Những lần sau không cài lại `.venv`.

### 6.2 Chạy lại từ lần thứ hai

Mở PowerShell tại thư mục `ADBench` và chạy một dòng:

```powershell
& .\.venv\Scripts\python.exe .\scripts\compare_nfst_tmpnfst_7_models.py
```

### 6.3 `Activate` nghĩa là gì?

Activate làm cho lệnh `python` tự trỏ đến `.venv`. Đây là bước tùy chọn vì các
command phía trên đã gọi trực tiếp Python trong `.venv`.

```powershell
.\.venv\Scripts\Activate.ps1
python .\scripts\compare_nfst_tmpnfst_7_models.py
```

Thoát môi trường bằng:

```powershell
deactivate
```

### 6.4 Khi Python và dependency đã được chuẩn bị sẵn

Chạy từ thư mục gốc repo:

```bash
python scripts/compare_nfst_tmpnfst_7_models.py
```

Nếu hệ thống dùng tên lệnh `python3`:

```bash
python3 scripts/compare_nfst_tmpnfst_7_models.py
```

### 6.5 Chạy toàn bộ dataset Classical tương thích giới hạn RAM

```bash
python scripts/compare_nfst_tmpnfst_7_models.py --all-compatible
```

## 7. Kết quả đầu ra và lưu ý

- Terminal: ranking và winner riêng cho từng dataset.
- Excel: một file `<dataset>_model_ranking.xlsx` cho mỗi dataset.
- Tổng hợp: CSV và JSON diagnostics trong `adbench/result/nfst_multi_dataset/`.
- `Pass` chỉ có nghĩa model chạy xong; không đồng nghĩa phân nhóm hoặc độ chính xác tối ưu.
- Không chọn tham số bằng test score; nếu tuning, nên dùng validation hoặc tiêu chí ổn định graph.
