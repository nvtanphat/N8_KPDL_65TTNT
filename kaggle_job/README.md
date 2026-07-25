# Kaggle Job

Chạy `scripts/train.py --model all` trên GPU miễn phí của Kaggle Kernels thay vì máy local,
điều khiển toàn bộ từ terminal local qua Kaggle CLI.

## Chuẩn bị (1 lần)

1. Cấu hình Kaggle API credential nếu chưa có (xem [../data/README.md](../data/README.md)).
2. `id` trong `kernel-metadata.json` đã điền sẵn `nguynvntnpht/bean-leaf-training-script` - đổi lại
   nếu dùng account Kaggle khác. Kaggle tự tạo slug từ `title`, nên `id` phải khớp với slug đó
   (`title` "Bean Leaf Training Script" → slug `bean-leaf-training-script`) - nếu sửa `title`, nhớ
   sửa `id` theo, không thì lệnh push sẽ tạo nhầm kernel khác.

## Chạy

> ⚠️ **Trên Windows:** set `PYTHONUTF8=1` trước khi gọi `kaggle` CLI (vd:
> `set PYTHONUTF8=1 && kaggle kernels push ...` trên cmd, hoặc `$env:PYTHONUTF8=1` trên
> PowerShell). Do `run_job.py` chứa tiếng Việt có dấu (UTF-8), CLI đọc file bằng codepage mặc định
> của Windows sẽ crash với `'charmap' codec can't decode byte ...`. `PYTHONIOENCODING=utf-8` KHÔNG
> đủ để fix lỗi này (chỉ ảnh hưởng stdout, không ảnh hưởng `open()`) - phải dùng `PYTHONUTF8=1`.

```bash
# 1. Đẩy job lên Kaggle, chạy ngầm bằng GPU
kaggle kernels push -p ./kaggle_job

# 2. Kiểm tra trạng thái (Running / Complete / Error)
kaggle kernels status nguynvntnpht/bean-leaf-training-script

# 3. Khi Complete: tải checkpoint + log về máy local
kaggle kernels output nguynvntnpht/bean-leaf-training-script -p ./outputs
```

`run_job.py` tự clone repo từ GitHub (nhánh `main`) về máy ảo Kaggle, cài dependencies, rồi train
cả 4 model classification với dataset được mount sẵn qua `dataset_sources` trong
`kernel-metadata.json` (không cần `--download`). Nếu bạn vừa push code mới lên GitHub mà chưa thấy
phản ánh trong lần chạy tiếp theo, nhớ `git push` trước khi `kaggle kernels push` - job luôn clone
bản mới nhất trên GitHub, không dùng code local trực tiếp. Ngược lại, `kernel-metadata.json` và
`run_job.py` được tải thẳng từ local lên Kaggle (không qua git), nên sửa 2 file này chỉ cần
`kaggle kernels push` lại, không cần push GitHub trước.
