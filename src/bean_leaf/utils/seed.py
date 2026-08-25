import os
import random

import numpy as np
import torch


def set_seed(seed, deterministic=False):
    """
    Cố định toàn bộ nguồn ngẫu nhiên ảnh hưởng tới 1 lần train: khởi tạo trọng số (torch),
    thứ tự shuffle của DataLoader (torch), augmentation ngẫu nhiên trong transform
    (torch + random), và các thao tác numpy.

    Vì sao cần: BeanLeafLite train from-scratch nên kết quả phụ thuộc mạnh vào khởi tạo
    trọng số - 2 lần chạy cùng code, cùng config vẫn lệch vài điểm accuracy. Không cố định
    seed thì không thể phân biệt "model tốt hơn" với "lần chạy may hơn".

    deterministic=True ép cuDNN chọn thuật toán tất định (chậm hơn ~10-20%). Chỉ bật khi
    cần tái lập bit-exact; để tái lập ở mức thống kê thì set_seed() là đủ.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
