"""
report_dataset_stats.py — Sinh số liệu cho mục báo cáo "Lý do chọn mô hình".

Tạo HAI bảng tái lập được:
  • Bảng 1: Kiểm kê dữ liệu sau tiền xử lý (số window / session mỗi user).
  • Bảng 2: Dung lượng mô hình (tham số, kích thước, latency CPU) theo kiến trúc.

Bảng độ chính xác open-set KHÔNG sinh ở đây (tốn vài phút train) — nó được
lấy từ artifacts/<arch>/results_<mode>/openset_summary.json do pipeline chính
(main.py trong notebook) sinh ra. Script này chỉ lo phần nhẹ, chạy < 5 giây.

Chạy:  cd ml_pipeline && python report_dataset_stats.py
       (tuỳ chọn) python report_dataset_stats.py --data_dir processed
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Dùng đúng định nghĩa model canonical trong web_demo (cùng kiến trúc ship app)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web_demo"))
from models import build_backbone  # noqa: E402

WINDOW_T = 200
N_CH = 9
ARCHS = ["cnn", "convlstm", "convlstm_bi"]


def natural_key(name: str):
    digits = "".join(c for c in name if c.isdigit())
    return (int(digits) if digits else 0, name)


def dataset_table(data_dir: Path) -> None:
    users = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()],
        key=lambda d: natural_key(d.name),
    )
    if not users:
        print(f"  (không tìm thấy user trong {data_dir})")
        return

    print("\n=== BẢNG 1 — Kiểm kê dữ liệu sau tiền xử lý ===")
    print(f"{'user':<8}{'walking':>10}{'all(inert)':>12}{'sessions':>10}")
    print("-" * 40)
    tot_w = tot_a = 0
    for d in users:
        fw, fa = d / "X_walking.npy", d / "X_inertial.npy"
        fy = d / "y_walking.npy"
        nw = np.load(fw, mmap_mode="r").shape[0] if fw.exists() else 0
        na = np.load(fa, mmap_mode="r").shape[0] if fa.exists() else 0
        ns = len(np.unique(np.load(fy, allow_pickle=True))) if fy.exists() else 0
        tot_w += nw
        tot_a += na
        print(f"{d.name:<8}{nw:>10}{na:>12}{ns:>10}")
    print("-" * 40)
    print(f"{'TỔNG':<8}{tot_w:>10}{tot_a:>12}    n_users={len(users)}")
    print(f"  Window shape: ({WINDOW_T}, {N_CH})   "
          f"Số định danh: {len(users)}   Session/user: ~6")


def model_table() -> None:
    print("\n=== BẢNG 2 — Dung lượng mô hình theo kiến trúc ===")
    print(f"{'arch':<14}{'encoder_params':>16}{'total_params':>14}"
          f"{'size_MB':>10}{'latency_ms':>12}")
    print("-" * 66)
    dummy = torch.zeros(10, N_CH, WINDOW_T)
    for arch in ARCHS:
        m = build_backbone(arch, n_users=19)
        m.eval()
        enc = sum(p.numel() for n, p in m.named_parameters()
                  if not n.startswith("classifier"))
        tot = sum(p.numel() for p in m.parameters())
        size_mb = tot * 4 / 1e6  # float32

        with torch.no_grad():
            for _ in range(3):  # warmup
                m(dummy)
            t0 = time.perf_counter()
            for _ in range(20):
                m(dummy)
            lat_ms = (time.perf_counter() - t0) / 20 * 1000

        print(f"{arch:<14}{enc:>16,}{tot:>14,}{size_mb:>10.2f}{lat_ms:>12.1f}")
    print("  (latency: 20 lần forward, batch 10 cửa sổ, CPU)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="processed")
    a = p.parse_args()
    data_dir = Path(a.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parent / data_dir

    dataset_table(data_dir)
    model_table()
    print("\nBảng độ chính xác open-set: xem artifacts/<arch>/results_<mode>/"
          "openset_summary.json (sinh bởi main.py).")


if __name__ == "__main__":
    main()
