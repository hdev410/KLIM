# NFST Algorithm Specification

## Authority

The sole mathematical source of truth is `references/KLIM_Group.pdf`, especially Eqs. (9)-(17) and Algorithm 1. The detector's official project name is **NFST**: Anchor-based Kernelized One-Class LPP-NFST. The historical PDF filename is intentionally unchanged.

## Problem and data contract

`NFSTModel.fit(X_normal)` accepts a finite, non-empty `float64` matrix `(n,d)`. `NFSTModel.decision_function(X_test)` returns finite scores `(n_test,)`, where larger values mean more anomalous. The standalone model performs no feature scaling; callers must use a consistent training-fitted transformation when scaling is required.

The thin ADBench adapter is unsupervised: `NFST.fit(X_train, y_train)` fits on all rows of `X_train` and ignores `y_train`. This avoids treating ADBench's masked unsupervised labels as ground truth. ADBench owns splitting, scaling, metrics, timing, and CSV output.

## Source-defined stages

Let `U:(m,d)` be MiniBatch K-Means anchors. PDF Eq. (9) defines

```math
s_{ij}=\exp\left(-\frac{\lVert x_i-u_j\rVert_2^2}{\sigma}\right),
```

and Eq. (10) defines normalized memberships

```math
z_{ij}=\frac{s_{ij}}{\sum_{k=1}^{m}s_{ik}},\qquad Z\in\mathbb{R}^{n\times m}.
```

The implementation evaluates this ratio as a stable row-wise softmax of `-dist2/sigma`. A supplied `sigma` must be finite and positive. When `sigma=None`, training resolves it once from the median positive nearest-anchor squared distance and stores `sigma_`; prediction reuses that value.

With `H=I-11^T/n`, Eqs. (12)-(16) define

```math
\Psi_G=Z^T H,
\quad W=ZZ^T,
\quad D_{ii}=\sum_j W_{ij},
\quad L=D-W,
```

```math
S_{wreg}^{\phi}
=\alpha Z^T HZ+(1-\alpha)Z^T LZ,
\qquad
S_T^{\phi}=Z^T HZ.
```

Self-similarities are included in the degree sum. The code constructs the equivalent scatter expression directly and symmetrizes only floating-point roundoff. It does not implement the rejected reconstruction expression, signed global/local subtraction, generalized eigenproblem, ridge `gamma`, or a fixed `null_dim`.

## NFST projection solver

The source two-stage solver is:

1. Symmetrically decompose `S_T` and retain its numerical range basis `Q:(m,r)`.
2. Form `M=Q.T @ S_wreg @ Q:(r,r)`.
3. Symmetrically solve `M a_i=lambda_i a_i`, sorted ascending.
4. In source-strict mode, select only eigenvectors satisfying the scale-aware near-zero rule.
5. Return `V=Q @ A:(m,p)`.

Rank selection uses `rank_tolerance * max(1, max(abs(eigenvalues(S_T))))`. Null selection uses `null_tolerance * max(1, max(abs(eigenvalues(M))))`. `max_components`, when provided, only caps already-valid null directions; it never admits a non-null eigenvector. The solver raises a clear error when `rank(S_T)=0` or the reduced nullity is zero. No explicit inverse or generalized eigensolver is used.

Diagnostics retain the total and reduced spectra, applied thresholds, selected eigenvalues, range rank, nullity, orthogonality error, and residual norms.

The default `selection_mode="null"` is the source-strict behavior above. Real datasets can have zero reduced nullity, particularly for `alpha>0`. An explicit practical variant, `selection_mode="smallest"`, selects the `max_components` eigenvectors with the smallest reduced eigenvalues without calling them null vectors. This mode requires an explicit positive `max_components`, records its mode and selected eigenvalues in diagnostics, and is an algorithmic low-eigen extension rather than the exact-null rule in the PDF. It must never be activated silently.

## Pseudo-subclasses and scoring

PDF Eq. (11) assigns each training row by

```math
c_i=\arg\max_j z_{ij}.
```

Ties use NumPy's deterministic first-maximum rule. Projected training samples are `Y_train=ZV`. For each non-empty subclass,

```math
b_j=\operatorname{mean}_{i\in C_j}(z_iV).
```

Empty anchor IDs are excluded from `base_points_` and recorded through `valid_cluster_ids_` and counts. Test samples reuse the fitted anchors, `sigma_`, projection, and base points:

```math
A(x^*)=\min_j\lVert z(x^*)V-b_j\rVert_2^2.
```

Scoring is batched. Tiny negative squared distances caused by cancellation are clamped to zero; materially negative or non-finite values raise errors.

## Fitted state

Successful fitting stores at least `anchors_`, `sigma_`, `projection_`, `eigenvalues_`, `base_points_`, `valid_cluster_ids_`, `n_features_in_`, and `diagnostics_`. `is_fitted_` becomes true only after every stage succeeds. A failed refit clears all prior learned state, preventing partially fitted reuse. Prediction never refits training components.

## Public hyperparameters

| Parameter | Contract |
|---|---|
| `n_anchors` | Positive integer; must not exceed training rows. |
| `sigma` | `None` or a finite positive Eq. (9) denominator. |
| `alpha` | Finite value in `[0,1]`. |
| `random_state` | Integer seed used by anchor generation. |
| `batch_size` | Positive MiniBatch K-Means batch size. |
| `tol` | Positive numerical tolerance used by mapping/scatter validation. |
| `rank_tolerance` | Positive relative threshold for `range(S_T)`. |
| `null_tolerance` | Positive relative threshold for reduced null eigenvalues. |
| `max_components` | Optional positive cap on valid null directions. |
| `selection_mode` | `"null"` by default; opt-in `"smallest"` practical low-eigen variant requires `max_components`. |
| `max_graph_bytes` | Positive memory guard for dense `W` and `L`. |
| `score_batch_size` | Positive test-scoring batch size. |

## Complexity and known limitations

Anchor mapping costs `O(nmd)`. The exact source graph uses dense `W,L:(n,n)`, requiring `O(n^2)` memory and `O(n^2m)`-scale work. The implementation estimates the two dense float64 matrices and rejects work above `max_graph_bytes`; it does not invent a sparse approximation.

The strict source null-space rule can legitimately yield zero valid projection dimensions, especially when regularization removes the reduced null space. Strict mode reports this as a fit error. The opt-in low-eigen mode permits a practical projection but changes the selection rule and must be reported as a variant. Feature scaling, bandwidth choice, `alpha`, component count, and numerical tolerances remain dataset-sensitive integration risks.

## Resolved decisions

| Issue | Implemented policy |
|---|---|
| Mathematical authority | PDF Eqs. (9)-(17) and Algorithm 1 only. |
| Local/global scatter | Direct source-equivalent `alpha Z^T HZ + (1-alpha) Z^T LZ`. |
| Degree | `D_ii=sum_j W_ij`, including the diagonal. |
| Solver | Range of `S_T`, then ordinary symmetric reduced null eigensystem. |
| Projection size | Numerical nullity, optionally capped by `max_components`. |
| Empty subclasses | Exclude them from centroids and scoring. |
| Standalone scaling | No internal scaler. |
| ADBench labels | Ignore `y_train`; fit all supplied training rows. |
| Scalability | Exact dense graph with an explicit memory guard. |

---

# Hướng dẫn triển khai NFST chi tiết bằng tiếng Việt

## 1. Mục đích và nguyên tắc đọc tài liệu

Phần này giải thích cách thuật toán NFST đi từ dữ liệu huấn luyện đến điểm bất
thường. Nguồn toán học duy nhất là `references/KLIM_Group.pdf`. Code được đối
chiếu chỉ nằm trong `adbench/baseline/NFST/`.

Các nhãn dưới đây được dùng xuyên suốt:

- **[Theo PDF]**: công thức hoặc thao tác được nêu trực tiếp trong PDF.
- **[Kỹ thuật triển khai]**: kiểm tra dữ liệu, chống tràn số, giới hạn bộ nhớ
  hoặc tổ chức code; không thay đổi công thức nếu chạy thành công.
- **[Chưa xác định]**: PDF không quy định đủ để kết luận đây là lựa chọn duy
  nhất đúng.
- **[Biến thể ngoài PDF]**: code cung cấp thêm một cách xử lý không phải thuật
  toán gốc trong PDF.

Nếu PDF và code không đủ thông tin, tài liệu ghi **chưa xác định**, không suy
đoán thêm.

## 2. NFST giải quyết bài toán gì?

NFST là detector bất thường một lớp:

- Khi huấn luyện, model nhận một ma trận đặc trưng `X_train`.
- Model học hình dạng và các nhóm con của dữ liệu được xem là lớp bình thường.
- Khi dự đoán, mỗi mẫu mới được biến đổi vào không gian NFST.
- Mẫu càng xa các tâm nhóm bình thường thì anomaly score càng lớn.

Trong ADBench, adapter bỏ qua `y_train`. Vì vậy model không sử dụng nhãn bất
thường để học. Nếu `X_train` chứa các điểm bất thường chưa được biết, chúng vẫn
có thể ảnh hưởng đến anchor, subclass và centroid.

## 3. Bức tranh tổng thể

```text
X_train: (n, d)
    |
    | MiniBatch K-means
    v
Anchors U: (m, d)
    |
    | Gaussian similarity + chuẩn hóa
    v
Anchor memberships Z: (n, m)
    |
    +--> Subclass assignment: argmax từng hàng Z
    |
    +--> W = Z Z^T, D, L = D - W
    |
    +--> S_T = Z^T H Z
    |
    `--> S_wreg = alpha S_T + (1-alpha) Z^T L Z
                |
                | Range-space + null-space solver
                v
          Projection V: (m, p)
                |
                | Z V
                v
       Projected training samples: (n, p)
                |
                | Trung bình theo subclass
                v
          Base points B: (q, p)

X_test --> Z_test --> Y_test = Z_test V
                         |
                         | khoảng cách bình phương đến base point gần nhất
                         v
                  anomaly scores: (n_test,)
```

Ký hiệu:

- `n`: số mẫu huấn luyện.
- `d`: số đặc trưng ban đầu.
- `m`: số anchor.
- `r`: hạng số học của total scatter.
- `p`: số chiều chiếu NFST được chọn.
- `q`: số subclass không rỗng, `q <= m`.

## 4. Vai trò của từng file

| File | Trách nhiệm |
|---|---|
| `anchor_mapping.py` | Tạo anchor, tính Gaussian similarity và tạo `Z`. |
| `scatter.py` | Gán subclass, tạo `W`, `D`, `L` và các scatter matrix. |
| `solver.py` | Tìm range-space, null-space và projection `V`. |
| `scoring.py` | Tạo subclass base points và tính anomaly score. |
| `model.py` | Điều phối toàn bộ quá trình huấn luyện và dự đoán. |
| `run.py` | Adapter mỏng để NFST có API phù hợp ADBench. |
| `__init__.py` | Xuất hai lớp public `NFST` và `NFSTModel`. |

`model.py` là file điều phối thuật toán chính. Các phép toán cụ thể được chia
cho bốn file `anchor_mapping.py`, `scatter.py`, `solver.py` và `scoring.py`.
`run.py` không chứa công thức NFST.

## 5. Bước 1 - Anchor mapping

File: `adbench/baseline/NFST/anchor_mapping.py`

### 5.1 Tại sao cần anchor?

Nếu dùng kernel trực tiếp cho `n` mẫu, model có thể phải tạo ma trận kernel
`n x n`, rất tốn bộ nhớ và thời gian. PDF thay thế toàn bộ tập train bằng `m`
anchor, với `m` nhỏ hơn nhiều so với `n`.

Mỗi anchor có thể hiểu là một điểm đại diện cho một vùng dữ liệu.

### 5.2 `_validate_positive_integer(name, value)`

Đây là hàm kiểm tra kỹ thuật:

- `n_anchors`, `batch_size` và `n_init` phải là số nguyên.
- Giá trị phải lớn hơn hoặc bằng 1.
- Không chấp nhận `True` hoặc `False` như số nguyên.

**Đánh giá:** [Kỹ thuật triển khai]. PDF không mô tả validation.

### 5.3 `_validate_sigma(sigma)`

Hàm đảm bảo:

- `sigma` có thể là `None`; hoặc
- phải là một số hữu hạn và lớn hơn 0.

`sigma` là mẫu số trong Gaussian kernel. Nếu `sigma <= 0`, công thức (9) không
có ý nghĩa phù hợp cho cách triển khai này.

**Đánh giá:** kiểm tra tính hợp lệ là [Kỹ thuật triển khai].

### 5.4 `_validate_input(X, expected_features=None)`

Hàm chuẩn hóa hợp đồng dữ liệu:

- `X` phải là ma trận hai chiều.
- Phải có ít nhất một hàng và một cột.
- Chỉ chấp nhận dữ liệu số thực hữu hạn.
- Chuyển dữ liệu sang `float64`.
- Khi dự đoán, số feature phải giống lúc huấn luyện.

Hàm không scale feature. Nó chỉ kiểm tra và chuyển kiểu.

**Đánh giá:** [Kỹ thuật triển khai]. PDF không quy định cách scale dữ liệu.

### 5.5 `_squared_euclidean_distances(X, anchors)`

Hàm tính:

```math
\lVert x_i-u_j\rVert_2^2
```

cho mọi cặp mẫu `x_i` và anchor `u_j`. Kết quả có kích thước `(n, m)`.

Ví dụ một mẫu cách ba anchor với khoảng cách bình phương:

```text
[0.2, 1.5, 4.0]
```

thì anchor thứ nhất gần mẫu nhất.

Code dùng `float64`, chặn khoảng cách âm do sai số và chuyển giá trị tràn số
thành số hữu hạn lớn nhất.

**Đánh giá:** phép tính khoảng cách là [Theo PDF]; xử lý tràn số là
[Kỹ thuật triển khai].

### 5.6 `AnchorMapping.__init__(...)`

Constructor lưu:

- `n_anchors`: số anchor `m`.
- `sigma`: bandwidth Gaussian.
- `random_state`: seed cho K-means.
- `batch_size`: kích thước batch của MiniBatch K-means.
- `n_init`: số lần khởi tạo K-means.

Nó cũng tạo các thuộc tính trạng thái như `anchors_`, `sigma_` và
`anchor_mapping_fitted_`.

PDF yêu cầu dùng `m` anchor và Mini-batch K-means nhưng không cho giá trị cụ
thể của `m`, `batch_size`, `n_init` hoặc seed.

**Đánh giá:** MiniBatch K-means là [Theo PDF]; giá trị các tham số là
[Chưa xác định].

### 5.7 `fit_anchor_mapping(X)`

Đây là hàm học anchor:

1. Kiểm tra `X`.
2. Kiểm tra `n_anchors <= n_samples`.
3. Chạy `MiniBatchKMeans`.
4. Lưu các tâm cụm thành `anchors_`.
5. Tính khoảng cách bình phương từ train đến anchors.
6. Xác định `sigma_`.

Nếu người dùng truyền `sigma`, code dùng đúng giá trị đó. Nếu `sigma=None`,
code lấy median của các khoảng cách bình phương dương từ mỗi mẫu đến anchor gần
nhất. Nếu mọi khoảng cách bằng 0, code dùng machine epsilon.

PDF chỉ xuất hiện `sigma` trong công thức (9), không nói phải chọn nó như thế
nào. Vì vậy:

- Dùng một `sigma` dương trong công thức: **[Theo PDF]**.
- Quy tắc median tự động: **[Chưa xác định]**.
- Machine-epsilon fallback: **[Kỹ thuật triển khai]**, không được PDF xác nhận
  là quy tắc toán học.

### 5.8 `compute_similarity(X)`

Hàm thực hiện trực tiếp công thức (9):

```math
s_{ij}
=\exp\left(-\frac{\lVert x_i-u_j\rVert_2^2}{\sigma}\right).
```

Nếu mẫu gần anchor, `s_ij` gần 1. Nếu xa anchor, `s_ij` gần 0.

Ví dụ:

```text
Khoảng cách bình phương: [0.2, 1.5, 4.0]
sigma = 1
Similarity xấp xỉ:       [0.819, 0.223, 0.018]
```

**Đánh giá:** [Theo PDF], công thức (9).

### 5.9 `transform_anchor_space(X)`

Hàm tạo membership `Z` theo công thức (10):

```math
z_{ij}=\frac{s_{ij}}{\sum_k s_{ik}}.
```

Mỗi hàng của `Z`:

- không âm;
- có `m` phần tử;
- tổng xấp xỉ bằng 1.

Với similarity ở ví dụ trên:

```text
S = [0.819, 0.223, 0.018]
Tổng = 1.060
Z xấp xỉ [0.773, 0.210, 0.017]
```

Mẫu này thuộc về anchor thứ nhất nhiều nhất.

Code trừ khoảng cách nhỏ nhất trước khi gọi `exp`. Đây là kỹ thuật tương đương
softmax ổn định, giúp tránh mọi similarity cùng bị làm tròn về 0. Tỉ lệ sau
chuẩn hóa không thay đổi về mặt toán học.

Hàm chỉ dùng `anchors_` và `sigma_` đã học; không fit lại trên dữ liệu test.

**Đánh giá:** chuẩn hóa là [Theo PDF]; cách tính ổn định là
[Kỹ thuật triển khai].

### 5.10 `fit_transform_anchor_space(X)`

Hàm tiện ích gọi liên tiếp:

```text
fit_anchor_mapping(X)
transform_anchor_space(X)
```

Kết quả trả về là `Z_train:(n,m)`.

## 6. Bước 2 - Subclass và scatter matrices

File: `adbench/baseline/NFST/scatter.py`

### 6.1 `validate_memberships(Z)`

Hàm kiểm tra:

- `Z` là ma trận hai chiều hữu hạn.
- Mọi phần tử không âm.
- Tổng mỗi hàng xấp xỉ 1.

Đây là điều kiện cần để mỗi hàng được hiểu như soft membership.

**Đánh giá:** ý nghĩa membership đến từ [Theo PDF]; validation và tolerance là
[Kỹ thuật triển khai].

### 6.2 `assign_subclasses(Z)`

PDF công thức (11) gán mẫu vào anchor có membership lớn nhất:

```math
c_i=\arg\max_j z_{ij}.
```

Ví dụ:

```text
Z_i = [0.773, 0.210, 0.017]
```

thì mẫu `i` được gán vào subclass 0 nếu đánh số từ 0 trong Python.

Nếu có hai giá trị bằng nhau, `np.argmax` chọn vị trí đầu tiên. PDF không quy
định cách phá hòa.

**Đánh giá:** `argmax` là [Theo PDF]; quy tắc chọn vị trí đầu tiên khi hòa là
[Chưa xác định].

### 6.3 `_symmetrize`, `_validate_symmetric`, `_validate_psd`

Các hàm này kiểm tra tính chất số học:

- Scatter và Laplacian phải đối xứng.
- Các ma trận được kỳ vọng là positive semidefinite trong sai số cho phép.
- Sai số làm tròn rất nhỏ được loại bằng `(M + M.T)/2`.

Các hàm không thay đổi công thức chính, nhưng dừng sớm nếu ma trận sai nghiêm
trọng.

**Đánh giá:** [Kỹ thuật triển khai].

### 6.4 `construct_graph_laplacian(Z)`

Hàm tạo graph theo PDF:

```math
W=ZZ^T.
```

`W_ij` lớn khi hai mẫu có phân bố membership giống nhau. Kích thước của `W` là
`(n,n)`.

Sau đó code tính degree:

```math
D_{ii}=\sum_j W_{ij}
```

và Laplacian:

```math
L=D-W.
```

Code không lưu một ma trận `D` dense riêng. Nó bắt đầu từ `-W`, rồi cộng degree
vào đường chéo. Kết quả toán học vẫn là `D-W`.

Code giữ cả `W_ii`, tức self-similarity, trong tổng degree. PDF nói `W=ZZ^T`
nhưng không chỉ rõ có phải xóa đường chéo trước khi tính `D` hay không.

**Đánh giá:**

- `W=ZZ^T` và `L=D-W`: [Theo PDF].
- Cách xác định degree bằng tổng hàng: cách hiểu Laplacian chuẩn, nhưng chi tiết
  không được PDF viết rõ hoàn toàn.
- Giữ self-similarity: [Chưa xác định].
- `max_graph_bytes`: [Kỹ thuật triển khai] để ngăn hết bộ nhớ.

### 6.5 Vì sao `W` và `L` tốn bộ nhớ?

Vì cả hai có kích thước `(n,n)`. Nếu `n` tăng gấp đôi, số phần tử tăng khoảng
bốn lần. Code hiện triển khai graph dense đúng theo công thức, không tự tạo
phiên bản sparse.

Giới hạn `max_graph_bytes` chỉ quyết định có cho phép chạy hay không. Nó không
xấp xỉ hoặc thay đổi `W` và `L`.

### 6.6 `construct_scatter_matrices(Z, alpha)`

Hàm tạo hai thành phần:

#### Global scatter

PDF dùng:

```math
S_T^\phi=Z^THZ,
\qquad
H=I-\frac{1}{n}11^T.
```

Code không tạo `H:(n,n)` trực tiếp. Nó trừ trung bình từng cột:

```text
centered = Z - mean(Z, axis=0)
global_scatter = centered.T @ centered
```

Hai cách tương đương toán học. Kết quả có kích thước `(m,m)`.

#### Local scatter

Code tính:

```math
S_L=Z^TLZ.
```

Đây là dạng cuối của thành phần local trong công thức (16). Code không cần tính
`L^(1/2)` vì PDF đã rút gọn về `Z^TLZ`.

#### Regularized scatter

Code kết hợp:

```math
S_{wreg}^{\phi}
=\alpha Z^THZ+(1-\alpha)Z^TLZ.
```

Ý nghĩa:

- `alpha` gần 1: ưu tiên co toàn bộ dữ liệu về cấu trúc toàn cục.
- `alpha` gần 0: ưu tiên thành phần graph/local.

PDF chỉ quy định `alpha` nằm trong `[0,1]`, không quy định giá trị tốt nhất cho
từng dataset.

Hàm cũng gán subclass bằng `argmax` và ghi lại các subclass không có mẫu.

**Đánh giá:** các scatter là [Theo PDF], công thức (12)-(16). Cách chọn
`alpha` là [Chưa xác định].

## 7. Bước 3 - NFST solver

File: `adbench/baseline/NFST/solver.py`

Đây là bước quyết định không gian chiếu. Mục tiêu theo PDF:

```math
V^TS_{wreg}^{\phi}V\rightarrow 0
```

nhưng đồng thời:

```math
V^TS_T^\phi V>0.
```

Nói đơn giản:

- Phép chiếu phải làm scatter trong lớp gần bằng 0.
- Nhưng không được chọn hướng mà toàn bộ dữ liệu không còn biến thiên.

### 7.1 Các hàm validation

`_validate_tolerance`, `_validate_max_components` và `_validate_scatter` kiểm
tra:

- tolerance là số dương hữu hạn;
- số component là số nguyên dương hoặc `None`;
- scatter là ma trận vuông, hữu hạn.

`_symmetrize_roundoff` chỉ cho phép sai lệch đối xứng nhỏ và trung bình hóa
`(M+M.T)/2`.

**Đánh giá:** [Kỹ thuật triển khai].

### 7.2 `solve_nfst_projection(...)` - Giai đoạn range-space

PDF yêu cầu tìm cơ sở trực chuẩn `Q` của range của `S_T`.

Code thực hiện:

1. Phân rã đối xứng `S_T` bằng `np.linalg.eigh`.
2. Lấy các eigenvector có eigenvalue lớn hơn `rank_threshold`.
3. Ghép chúng thành `Q:(m,r)`.

PDF Algorithm 1 ghi "via SVD", trong khi code dùng eigen-decomposition. Vì
`S_T=Z^THZ` là ma trận đối xứng positive semidefinite, các eigenvector có
eigenvalue dương sinh cùng range-space mà SVD cần tìm.

Tuy nhiên PDF không quy định ngưỡng số học:

```text
rank_threshold =
    rank_tolerance * max(1, max(abs(eigenvalues(S_T))))
```

Do đó:

- Tìm `range(S_T)`: [Theo PDF].
- Dùng `eigh` thay SVD trên ma trận đối xứng PSD: triển khai tương đương về
  range-space.
- Giá trị `rank_tolerance`: [Chưa xác định].

Nếu không có eigenvalue hợp lệ, `rank(S_T)=0` và solver dừng vì không có
range-space để giữ total variance.

### 7.3 Tạo bài toán thu gọn

Sau khi có `Q`, code tạo:

```math
M=Q^TS_{wreg}^{\phi}Q.
```

`M` có kích thước `(r,r)`, thường nhỏ hơn `(m,m)`.

Sau đó code giải:

```math
Ma_i=\lambda_i a_i
```

bằng `np.linalg.eigh` và sắp eigenvalue tăng dần.

**Đánh giá:** [Theo PDF], công thức (17) và Algorithm 1.

### 7.4 Chế độ `selection_mode="null"`

Code xem eigenvalue là gần zero khi:

```text
abs(lambda_i) <=
null_tolerance * max(1, max(abs(all reduced eigenvalues)))
```

Sau đó:

```math
V=QA,
```

trong đó `A` chứa các eigenvector gần zero.

Đây là ý nghĩa của null-space: sau phép chiếu, regularized within-class scatter
xấp xỉ 0.

Nếu không có eigenvalue gần zero, code báo:

```text
Reduced NFST problem has numerical nullity zero
```

Code không tự chọn một hướng khác trong chế độ strict.

**Đánh giá:**

- Chỉ chọn eigenvector có eigenvalue xấp xỉ zero: [Theo PDF].
- Công thức cụ thể của `null_tolerance`: [Chưa xác định], vì PDF chỉ ghi
  `lambda_i ≈ 0`.
- `max_components` cắt bớt số null vector hợp lệ: PDF không quy định, nên
  [Chưa xác định].

### 7.5 Chế độ `selection_mode="smallest"`

Ở chế độ này, code lấy `max_components` eigenvector có eigenvalue nhỏ nhất, kể
cả khi chúng không gần zero.

Điều đó giúp model vẫn tạo được projection khi numerical nullity bằng 0. Tuy
nhiên nó không thỏa yêu cầu "eigenvectors corresponding to zero eigenvalues"
của PDF.

**Đánh giá:** **[Biến thể ngoài PDF]**.

Kết quả chạy bằng chế độ này phải được gọi rõ là biến thể low-eigen, không được
trình bày như nghiệm null-space gốc.

### 7.6 Projection và diagnostics

Code tạo:

```math
V=QA.
```

`V` có kích thước `(m,p)`. Nó biến một membership vector `z:(1,m)` thành:

```math
y=zV,\qquad y:(1,p).
```

Solver lưu diagnostics:

- spectrum của `S_T`;
- spectrum của bài toán thu gọn;
- rank threshold;
- null threshold;
- numerical rank và nullity;
- eigenvalue được chọn;
- sai số trực giao;
- null residual và range residual.

Diagnostics giúp kiểm tra model có thật sự tìm được null-space hay chỉ chạy
biến thể low-eigen. Đây là [Kỹ thuật triển khai].

## 8. Bước 4 - Base points và anomaly score

File: `adbench/baseline/NFST/scoring.py`

### 8.1 Các hàm validation

Các hàm `_validate_projection`, `_validate_assignments`, `_validate_points` và
`_validate_batch_size` đảm bảo:

- `V` có `m` hàng.
- Mỗi mẫu train có đúng một subclass hợp lệ.
- Điểm chiếu và centroid hữu hạn.
- Batch size hợp lệ.

**Đánh giá:** [Kỹ thuật triển khai].

### 8.2 `build_subclass_base_points(...)`

Hàm đầu tiên chiếu train:

```math
Y_{train}=Z_{train}V.
```

Sau đó, với mỗi subclass không rỗng:

```math
b_j=\operatorname{mean}_{i\in C_j}(z_iV).
```

`b_j` là base point hay centroid của subclass `j` trong không gian NFST.

Ví dụ:

```text
Subclass 0 có ba điểm chiếu:
[0.2, 0.3]
[0.4, 0.3]
[0.3, 0.6]

Base point:
b_0 = [0.3, 0.4]
```

Code bỏ subclass rỗng thay vì tạo một centroid giả bằng zero. PDF không nói
cách xử lý subclass rỗng.

**Đánh giá:**

- Tính centroid từ các điểm chiếu: [Theo PDF], Algorithm 1 bước 11.
- Bỏ subclass rỗng: [Chưa xác định].

### 8.3 `score_anchor_memberships(...)`

Hàm nhận `Z_test`, sau đó chiếu:

```math
Y_{test}=Z_{test}V.
```

Nó chuyển kết quả cho `score_projected_samples`.

**Đánh giá:** [Theo PDF], Algorithm 1 bước 12.

### 8.4 `score_projected_samples(...)`

Với mỗi mẫu test đã chiếu `y*`, code tính khoảng cách bình phương đến mọi base
point:

```math
\lVert y^*-b_j\rVert_2^2.
```

Anomaly score là khoảng cách nhỏ nhất:

```math
A(x^*)=\min_j\lVert y^*-b_j\rVert_2^2.
```

Ví dụ:

```text
Khoảng cách bình phương đến ba base point:
[0.04, 1.20, 2.10]

Anomaly score = 0.04
```

Mẫu này gần một nhóm bình thường nên score thấp.

Nếu khoảng cách là:

```text
[3.50, 4.20, 5.00]
```

thì score là `3.50`, cho thấy mẫu xa mọi nhóm bình thường và có khả năng bất
thường cao hơn.

Code tính theo batch để giảm bộ nhớ. Sai số âm rất nhỏ từ phép tính số thực được
chặn về 0; giá trị âm đáng kể hoặc không hữu hạn gây lỗi.

**Đánh giá:** anomaly score là [Theo PDF], Algorithm 1 bước 13; batching và xử
lý roundoff là [Kỹ thuật triển khai].

## 9. `NFSTModel` điều phối các bước như thế nào?

File: `adbench/baseline/NFST/model.py`

### 9.1 Constructor

`NFSTModel.__init__` chỉ:

- kiểm tra và lưu hyperparameter;
- tạo trạng thái chưa fit;
- không chạy bất kỳ công thức học nào.

### 9.2 `_reset_fitted_state()`

Hàm xóa toàn bộ trạng thái đã học:

- anchors và sigma;
- `Z_train`;
- scatter matrices;
- range basis và projection;
- eigenvalues;
- base points;
- diagnostics;
- các fitted flags.

`fit()` gọi hàm này trước khi học. Nếu một lần refit thất bại giữa chừng, model
không giữ lại trạng thái cũ hoặc trạng thái học dở.

**Đánh giá:** [Kỹ thuật triển khai].

### 9.3 `_check_is_fitted()`

Ngăn gọi `decision_function()` trước khi huấn luyện hoàn tất.

**Đánh giá:** [Kỹ thuật triển khai].

### 9.4 `_new_anchor_mapping()`

Tạo một `AnchorMapping` mới từ các tham số của model. Điều này bảo đảm mỗi lần
fit bắt đầu với một bộ mapping sạch.

### 9.5 `fit(X_normal)`

Đây là luồng huấn luyện end-to-end:

```text
1. Xóa fitted state cũ
2. Học anchors và tạo Z_train
3. Tạo subclass, graph và scatter matrices
4. Giải NFST projection V
5. Chiếu train và tạo subclass base points
6. Chỉ khi tất cả thành công mới commit trạng thái và đặt is_fitted_=True
```

Các lời gọi chính:

```python
mapping.fit_transform_anchor_space(X_normal)
construct_scatter_matrices(Z_train, alpha)
solve_nfst_projection(S_T, S_wreg)
build_subclass_base_points(Z_train, V, assignments)
```

`fit()` không tự scale dữ liệu và không nhận nhãn.

### 9.6 Các hàm fit từng phần

`fit_anchor_mapping(X_normal)` chỉ chạy bước anchor và tạo `Z_train`.

`transform_anchor_space(X)` dùng mapping đã học để tạo `Z` cho dữ liệu khác.

`construct_scatter_from_anchor_mapping()` dùng `Z_train` hiện có để tạo scatter.

`fit_scatter_construction(X_normal)` chạy anchor mapping rồi scatter, nhưng chưa
chạy solver và scoring.

Các hàm này hữu ích để test từng stage. Chỉ `fit()` hoàn chỉnh mới đặt
`is_fitted_=True`.

### 9.7 `decision_function(X_test)`

Luồng inference:

```text
X_test
  -> dùng anchors và sigma đã fit để tạo Z_test
  -> chiếu Z_test bằng V đã fit
  -> tính khoảng cách đến base points đã fit
  -> trả về một score cho mỗi mẫu
```

Hàm không fit lại anchors, sigma, projection hoặc centroid.

## 10. Adapter ADBench

File: `adbench/baseline/NFST/run.py`

### 10.1 `NFST.__init__(seed, model_name, ...)`

Adapter:

- nhận `seed` theo detector contract của ADBench;
- nhận `model_name` để tương thích API;
- tạo một `NFSTModel`;
- chuyển `seed` thành `random_state` của model.

### 10.2 `NFST.fit(X_train, y_train)`

Hàm gọi:

```python
self.model.fit(X_train)
```

`y_train` chỉ tồn tại để tương thích ADBench và bị bỏ qua.

### 10.3 `NFST.predict_score(X_test)`

Hàm gọi:

```python
self.model.decision_function(X_test)
```

Nó không sửa hoặc đảo chiều score. Score lớn hơn nghĩa là xa các base point
bình thường hơn.

## 11. Những phần được xác định và chưa xác định

| Thành phần | Trạng thái |
|---|---|
| MiniBatch K-means tạo anchors | Theo PDF |
| Gaussian `exp(-distance_squared/sigma)` | Theo PDF |
| Chuẩn hóa similarity thành `Z` | Theo PDF |
| Phân subclass bằng `argmax` | Theo PDF |
| `W=ZZ^T`, `L=D-W` | Theo PDF |
| `S_T=Z^THZ` | Theo PDF |
| `S_wreg=alpha Z^THZ+(1-alpha)Z^TLZ` | Theo PDF |
| Range-space rồi null-space | Theo PDF |
| Chọn eigenvector có eigenvalue gần zero | Theo PDF |
| Centroid và nearest-centroid squared distance | Theo PDF |
| Tự chọn sigma bằng median nearest distance | Chưa xác định |
| Giá trị tốt nhất của số anchor | Chưa xác định |
| Giá trị tốt nhất của alpha | Chưa xác định |
| Rank/null tolerance cụ thể | Chưa xác định |
| Xử lý subclass rỗng | Chưa xác định |
| Feature scaling bên trong NFST | PDF không quy định; code không thực hiện |
| `selection_mode="smallest"` | Biến thể ngoài PDF |
| Memory guard và batching | Kỹ thuật triển khai |

## 12. Điểm mâu thuẫn trong PDF

Ở phần derivation đầu tài liệu, một câu văn ghi "vanishing of between scatter
variance", nhưng các công thức (4), (6), (7), (16), (17) và Algorithm 1 sử dụng
within-class scatter.

Do nguồn tự có mâu thuẫn về câu chữ:

- Không tự tạo thêm một cách hiểu mới.
- Code hiện theo các công thức và Algorithm 1.
- Câu văn mâu thuẫn được ghi nhận là **chưa xác định**.

## 13. Cách đọc kết quả benchmark

### 13.1 AUC-PR

AUC-PR đo khả năng tìm được anomaly trong khi hạn chế báo động nhầm. Nó phù hợp
với dữ liệu có rất ít anomaly và nên là metric chính.

Ví dụ có 100 giao dịch, gồm 5 gian lận. Một model xếp phần lớn 5 giao dịch gian
lận lên đầu và ít đưa giao dịch bình thường lên cùng nhóm sẽ có AUC-PR cao.

Giá trị càng gần 1 càng tốt. Mốc "tốt" không cố định cho mọi dataset; cần so
sánh các model trên cùng split và xem tỷ lệ anomaly gốc.

### 13.2 AUC-ROC

AUC-ROC đo khả năng một anomaly ngẫu nhiên nhận score cao hơn một normal sample
ngẫu nhiên, xét trên nhiều threshold.

- Gần `0.5`: khả năng xếp hạng gần ngẫu nhiên.
- Gần `1.0`: phân biệt hai nhóm rất tốt.

AUC-ROC là metric phụ vì trên dữ liệu cực mất cân bằng, nó có thể nhìn khá cao
trong khi model vẫn tạo nhiều false positive.

### 13.3 Fit time và inference time

- `fit time`: thời gian học anchors, `Z_train`, scatter, projection và base
  points.
- `inference time`: thời gian biến đổi `X_test` và tính anomaly scores.

Thời gian càng thấp càng nhanh, nhưng không biểu thị độ chính xác. Chỉ nên ưu
tiên tốc độ khi chất lượng AUC-PR và AUC-ROC vẫn đáp ứng yêu cầu.

### 13.4 Quy tắc so sánh công bằng

Các model phải dùng:

- cùng dữ liệu và cùng train/test split;
- cùng preprocessing;
- cùng seed khi có randomness;
- cùng định nghĩa anomaly label;
- score cùng chiều: lớn hơn nghĩa là bất thường hơn.

Nên báo cáo:

```text
mean AUC-PR +/- std
mean AUC-ROC +/- std
mean fit time
mean inference time
failed runs
```

trên nhiều seed. Không chọn hyperparameter bằng kết quả test vì điều đó làm sai
lệch so sánh.
