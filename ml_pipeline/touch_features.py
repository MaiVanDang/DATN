"""
touch_features.py — tiện ích nhỏ cho dữ liệu touch.

Đặc trưng touch dùng cho mô hình là bộ SUB-WINDOW 33-D (tap + scroll), định
nghĩa trong per_user_eval_cosine.touch_subwindows. File này chỉ còn giữ hàm
strip_user_prefix mà per_user_eval_cosine dùng để bỏ tiền tố user_id khỏi
session_id.
"""


def strip_user_prefix(prefixed: str, user_id: str) -> str:
    pre = f"{user_id}_"
    return prefixed[len(pre):] if prefixed.startswith(pre) else prefixed
