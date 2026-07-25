# Kaggle Job

Chạy `scripts/train.py --model all` trên GPU miễn phí của Kaggle Kernels thay vì máy local,
điều khiển toàn bộ từ terminal local qua Kaggle CLI.

## Chuẩn bị (1 lần)

1. Cấu hình Kaggle API credential nếu chưa có (xem [../data/README.md](../data/README.md)).
2. `id` trong `kernel-metadata.json` đã điền sẵn `nguynvntnpht/bean-leaf-training` - đổi lại nếu
   dùng account Kaggle khác.

## Chạy

```bash
# 1. Đẩy job lên Kaggle, chạy ngầm bằng GPU
kaggle kernels push -p ./kaggle_job

# 2. Kiểm tra trạng thái (Running / Complete / Error)
kaggle kernels status nguynvntnpht/bean-leaf-training

# 3. Khi Complete: tải checkpoint + log về máy local
kaggle kernels output nguynvntnpht/bean-leaf-training -p ./outputs
```

`run_job.py` tự clone repo từ GitHub (nhánh `main`) về máy ảo Kaggle, cài dependencies, rồi train
cả 4 model classification với dataset được mount sẵn qua `dataset_sources` trong
`kernel-metadata.json` (không cần `--download`). Nếu bạn vừa push code mới lên GitHub mà chưa thấy
phản ánh trong lần chạy tiếp theo, nhớ `git push` trước khi `kaggle kernels push` - job luôn clone
bản mới nhất trên GitHub, không dùng code local trực tiếp.
