# Dữ liệu

Code dùng bản dữ liệu do PraNet phát hành (https://github.com/DengPingFan/PraNet). Chạy
`python scripts/download_data.py` để tải và giải nén vào `data/polyp/`:

```
data/polyp/
├── TrainDataset/            1.450 ảnh: 900 Kvasir-SEG + 550 CVC-ClinicDB
│   ├── images/
│   └── masks/
└── TestDataset/
    ├── Kvasir/              100 ảnh   (trong miền)
    ├── CVC-ClinicDB/         62 ảnh   (trong miền)
    ├── CVC-ColonDB/         380 ảnh   (ngoài miền)
    ├── ETIS-LaribPolypDB/   196 ảnh   (ngoài miền)
    └── CVC-300/              60 ảnh   (ngoài miền)
        ├── images/
        └── masks/
```

`python scripts/prepare_splits.py` tạo các file trong `data/splits/`: `train.txt` (1.305 ảnh),
`val.txt` (145 ảnh, 10% mỗi nguồn, seed 42) và `test_<bộ>.txt`. Mỗi dòng gồm đường dẫn ảnh,
đường dẫn mask và tên bộ nguồn, ngăn cách bằng tab.

Dữ liệu không được đưa vào git và không phát hành lại; mỗi bộ tuân theo điều khoản gốc của nó.
