# Phân vùng polyp đại tràng: so sánh CNN và Transformer, cải thiện tổng quát hóa ngoài miền

Code cho đề tài cuối khóa môn Máy học nâng cao. Báo cáo LaTeX nằm trong `report/`, và mỗi bảng,
hình trong báo cáo có một script tương ứng ở đây (xem mục "Từ code đến báo cáo").

- **Mô hình:** U-Net, MultiResUNet (#59), CE-Net (#60), TransUNet (#57), Swin-Unet (#56), và mô
  hình đề xuất C1 (encoder PVTv2/ConvNeXt + decoder MultiResUNet).
- **Cải tiến:** C2 boundary loss, C3 augmentation theo miền + TTA, C1 encoder mạnh hơn (tùy chọn).
- **Đánh giá:** Dice, IoU, recall, precision, HD95 ở độ phân giải gốc; kiểm định Wilcoxon,
  Friedman + Nemenyi; khoảng tin cậy bootstrap 95%.

## Cấu trúc thư mục

```
Project/
├── README.md, requirements.txt
├── configs/                  cấu hình YAML (base.yaml + một file mỗi mô hình)
│   └── ablation/             phần ghi đè cho C2, C3, C2+C3
├── data/                     dữ liệu (không đưa vào git), splits/ do script tạo ra
├── src/
│   ├── models/               unet, multiresunet, cenet, transunet, swinunet, respath_unet
│   ├── data.py               dataset, đọc/ghi file split
│   ├── transforms.py         augmentation baseline và strong (C3)
│   ├── losses.py             BCE + Dice, boundary loss (C2)
│   ├── metrics.py            Dice, IoU, recall, precision, HD95
│   ├── inference.py          dự đoán, TTA, đưa về độ phân giải gốc
│   ├── config.py, utils.py, plotting.py
├── scripts/                  download_data, prepare_splits, check_duplicates, train, evaluate,
│                             benchmark, stats_tests, plot_*, make_dummy_data, run_all.sh
├── app/gradio_app.py         demo web
├── notebooks/colab_pipeline.ipynb
├── tests/test_smoke.py       kiểm thử nhanh trên CPU
├── outputs/                  kết quả chạy (runs/, metrics/, tables/, predictions/)
└── report/                   báo cáo LaTeX; hình được ghi vào report/figures/
```

## Cài đặt

```bash
pip install -r requirements.txt     # trên Colab/Kaggle: torch đã có sẵn
python -m pytest tests -q            # 10 bài kiểm thử, chạy trên CPU trong khoảng 15 giây
```

## Chạy thử nhanh không cần GPU và không cần tải dữ liệu

```bash
python scripts/make_dummy_data.py --root data/dummy
python scripts/prepare_splits.py --root data/dummy --out data/splits_dummy
python scripts/train.py --config configs/unet.yaml --opts data.root=data/dummy \
    data.splits_dir=data/splits_dummy data.img_size=128 model.params.base_channels=16 \
    train.epochs=10 train.batch_size=4 train.lr=1e-3 experiment=dummy_unet
python scripts/evaluate.py --run outputs/runs/dummy_unet
```

Dữ liệu giả gồm các hình elip trên nền nhiễu; trên dữ liệu này U-Net đạt val Dice khoảng 0,95
sau 10–20 epoch. Mục đích chỉ là kiểm tra pipeline, không có ý nghĩa khoa học.

## Quy trình thí nghiệm đầy đủ

```bash
python scripts/download_data.py                  # PraNet TrainDataset + TestDataset -> data/polyp
python scripts/prepare_splits.py                 # data/splits: 1.305 train, 145 val, 5 tập test
python scripts/check_duplicates.py               # ảnh gần trùng giữa train và test

# Baseline (mỗi lệnh ~1 giờ trên T4, ước tính chưa đo)
python scripts/train.py --config configs/unet.yaml
python scripts/train.py --config configs/multiresunet.yaml
python scripts/train.py --config configs/cenet.yaml
python scripts/train.py --config configs/transunet.yaml
python scripts/train.py --config configs/swinunet.yaml
for m in unet multiresunet cenet transunet swinunet; do
  python scripts/evaluate.py --run outputs/runs/$m --save-pred
done

# Ablation trên baseline tốt nhất (ví dụ cenet) -> run "cenet_c2_boundary", ...
python scripts/train.py --config configs/cenet.yaml configs/ablation/c3_domain_aug.yaml
python scripts/train.py --config configs/cenet.yaml configs/ablation/c2_boundary.yaml
python scripts/train.py --config configs/cenet.yaml configs/ablation/c2_c3.yaml

# Thống kê, chi phí, hình
python scripts/stats_tests.py --models unet multiresunet cenet transunet swinunet --reference cenet \
    --ablation cenet cenet_c3_domain_aug cenet_c2_boundary cenet_c2_c3
python scripts/benchmark.py --configs configs/unet.yaml configs/multiresunet.yaml configs/cenet.yaml \
    configs/transunet.yaml configs/swinunet.yaml
python scripts/plot_dataset_examples.py
python scripts/plot_report_figures.py --models unet multiresunet cenet transunet swinunet --reference cenet
```

Hoặc chạy tất cả bằng một lệnh: `bash scripts/run_all.sh` (đặt `BEST=` và `REFERENCE=` theo kết
quả tuần 4). Mọi lệnh đều chạy lại được: `train.py` tự resume từ `last.pt` nếu bị ngắt giữa chừng.

**Ghi đè cấu hình** bằng `--opts`, ví dụ `--opts train.epochs=50 train.batch_size=8
data.root=/content/data/polyp output_dir=/content/drive/MyDrive/polyp_runs`.

## Từ code đến báo cáo

| Trong báo cáo | Lệnh | File kết quả |
|---|---|---|
| Bảng 2 (chia dữ liệu) | `prepare_splits.py` | in ra màn hình, `data/splits/` |
| Kiểm tra ảnh trùng (Mục 3.3) | `check_duplicates.py` | `outputs/eval/duplicates.csv` |
| Bảng 5 (mDice 5 tập, p-value) | `evaluate.py` + `stats_tests.py --models` | `outputs/tables/table5_main_rows.tex` |
| Bảng 6 (tổng quát hóa, CI) | `stats_tests.py --models` | `outputs/tables/table6_ood_rows.tex` |
| Bảng 7 (ablation) | `stats_tests.py --ablation` | `outputs/tables/table7_ablation_rows.tex` |
| Bảng 8 (tham số, FLOPs, FPS) | `benchmark.py` | `outputs/tables/table8_cost_rows.tex` |
| Hình 1 | `plot_dataset_examples.py` | `report/figures/dataset_examples.pdf` |
| Hình 3, Hình 4 | `plot_report_figures.py` | `report/figures/generalization_gap.pdf`, `failure_cases.pdf` |
| Friedman + Nemenyi | `stats_tests.py --models` | `outputs/tables/friedman_nemenyi.csv` |

Các file `.tex` chứa sẵn các dòng của bảng; dán vào báo cáo thay cho các ô `\tbd`. Với bản tiếng
Việt, thêm `--vi` để dùng dấu phẩy thập phân.

## Ghi chú về tái hiện (nêu trong báo cáo)

- **U-Net:** dùng padding và BatchNorm (bản gốc không có) để đầu ra cùng kích thước đầu vào. 31,0M tham số.
- **MultiResUNet:** theo mã gốc của tác giả (alpha = 1,67, Res path dài 4-3-2-1). 7,24M tham số.
- **CE-Net:** theo mã gốc (ResNet-34 ImageNet, DAC, RMP). Ảnh vào phải là bội số của 32 và ≥ 192. 29,0M tham số.
- **TransUNet:** encoder R50+ViT-B/16 nạp từ checkpoint ImageNet-21k qua timm
  (`vit_base_r50_s16_224.orig_in21k`, cùng checkpoint gốc của TransUNet); decoder CUP như mã
  gốc; chuẩn hóa ảnh mean = std = 0,5 theo checkpoint. Ảnh vào 224. 105,3M tham số.
- **Swin-Unet:** encoder Swin-T ImageNet-1k qua timm với độ sâu đầy đủ (2, 2, 6, 2), trong khi
  bản gốc dùng (2, 2, 2, 2); decoder khởi tạo ngẫu nhiên thay vì sao chép từ encoder. Ảnh vào 224. 34,3M tham số.
- **HD95:** tự cài bằng scipy cho mọi mô hình; biên là phần mask bị loại khi erosion 3×3; dự
  đoán rỗng nhận HD95 bằng đường chéo ảnh.
- **Lịch học:** AdamW, warmup 2 epoch rồi cosine; chọn checkpoint theo Dice trên validation.

## Chạy trên Google Colab hoặc Kaggle

Mở `notebooks/colab_pipeline.ipynb`. Notebook gắn Google Drive, cài thư viện, tải dữ liệu
vào ổ cục bộ của Colab (đọc nhanh hơn Drive), và lưu `outputs/` lên Drive để không mất khi bị
ngắt phiên. Mỗi lệnh dừng ô ngay khi lỗi, nên lỗi hiện ra là lỗi đầu tiên.

**Khi không tải được dữ liệu** (link Google Drive của PraNet hay bị giới hạn lượt tải): mở hai
link được in ra, tải `TrainDataset.zip` và `TestDataset.zip` bằng trình duyệt, đặt vào
`MyDrive/advanced_ml/data_zips/`, rồi chạy lại ô Dữ liệu. Script giải nén từ đó và không tải lại.
Lần tải thành công đầu tiên cũng tự lưu một bản zip vào thư mục này cho các phiên sau. Trên Kaggle, tải thư mục Project lên làm Dataset, bật GPU, rồi chạy
`bash scripts/run_all.sh` với `DATA_ROOT` trỏ đến dữ liệu.

## Demo

```bash
pip install gradio
python app/gradio_app.py --run outputs/runs/cenet --share
```

Demo chỉ phục vụ mục đích học thuật, không phải thiết bị y tế.
