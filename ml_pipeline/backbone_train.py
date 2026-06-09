"""
backbone_train.py — Stage 1: train multi-class user-identification backbone

Mục đích: dạy CNN 1D học general "behavioral features" của các user. Sau
khi train xong, vứt classifier head, giữ lại encoder (output 128-D
embedding) — đây là phần được ship với app.

Train task: phân loại user-id (n_users lớp) trên TRAIN sessions của TẤT CẢ user.
            VAL set = VAL sessions (dùng để early stop).

Không nhúng nhãn activity (sitting/standing/walking) vào loss — backbone
tự học activity-agnostic representation qua việc nhìn data hỗn hợp.

Output:
  • model    : BackboneCNN đã train
  • val_acc  : multi-class accuracy trên val (chỉ để debug, không phải
               metric chính)
"""

import numpy as np
import torch
import torch.nn as nn

from models   import build_backbone
from config   import (
    BACKBONE_ARCH, BACKBONE_EPOCHS, BACKBONE_LR, BACKBONE_BATCH, BACKBONE_PATIENCE,
    EMBED_DIM, PRELOAD_TO_DEVICE, USE_AMP, ENABLE_TF32, USE_FUSED_ADAM,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda" and ENABLE_TF32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

def _make_grad_scaler():
    """torch.amp.GradScaler('cuda') tu PyTorch 2.4 — fallback len torch.cuda.amp."""
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()

def _autocast_ctx(dtype=torch.float16):
    """torch.amp.autocast('cuda', dtype=...) tu PyTorch 2.4 — fallback len torch.cuda.amp."""
    try:
        return torch.amp.autocast("cuda", dtype=dtype)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=dtype)

def _build_multiclass_dataset(users: dict,
                              user_to_id: dict[str, int],
                              session_filter: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Gộp data của TẤT CẢ user, chỉ lấy windows thuộc session_filter.
    Trả về (X, y) với y = user-id (int) theo user_to_id mapping.
    """
    X_list, y_list = [], []
    for uid, data in users.items():
        mask = np.isin(data["session"].astype(str), list(session_filter))
        if not mask.any():
            continue
        X_list.append(data["X"][mask])
        y_list.append(np.full(int(mask.sum()), user_to_id[uid], dtype=np.int64))

    if not X_list:
        raise ValueError("Không có window nào sau session_filter — pipeline lỗi.")

    X = np.concatenate(X_list).astype(np.float32)
    y = np.concatenate(y_list)
    return X, y

def train_backbone(users: dict,
                   user_to_id: dict[str, int],
                   train_sessions_global: set[str],
                   val_sessions_global: set[str],
                   seed: int = 42,
                   verbose: bool = True) -> tuple[nn.Module, float]:
    """
    Train backbone CNN multi-class identification.

    Parameters
    ----------
    users                : dict trả về từ load_users()
    user_to_id           : {user_id_str: int_label}
    train_sessions_global: tập sessions (PREFIXED) dùng để train
    val_sessions_global  : tập sessions (PREFIXED) dùng để early-stop
    seed                 : random seed

    Returns
    -------
    (model, best_val_acc)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_users = len(user_to_id)

    X_tr, y_tr = _build_multiclass_dataset(users, user_to_id, train_sessions_global)
    X_val, y_val = _build_multiclass_dataset(users, user_to_id, val_sessions_global)

    if verbose:
        print(f"    Backbone train: {len(X_tr):,} windows  "
              f"val: {len(X_val):,} windows  n_users={n_users}")

    model = build_backbone(BACKBONE_ARCH, n_users=n_users, embed_dim=EMBED_DIM).to(DEVICE)
    if verbose:
        print(f"    Backbone params: {model.count_params():,}")

    criterion = nn.CrossEntropyLoss()

    optimizer_kwargs = dict(lr=BACKBONE_LR, weight_decay=1e-4)
    if USE_FUSED_ADAM and DEVICE.type == "cuda":
        try:
            optimizer = torch.optim.Adam(model.parameters(), fused=True, **optimizer_kwargs)
        except (TypeError, RuntimeError):
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    use_amp = USE_AMP and DEVICE.type == "cuda"
    scaler  = _make_grad_scaler() if use_amp else None

    X_tr_t  = torch.from_numpy(np.ascontiguousarray(X_tr.transpose(0, 2, 1))).float()
    y_tr_t  = torch.from_numpy(y_tr).long()
    X_val_t = torch.from_numpy(np.ascontiguousarray(X_val.transpose(0, 2, 1))).float()
    y_val_t = torch.from_numpy(y_val).long()

    if PRELOAD_TO_DEVICE and DEVICE.type == "cuda":
        X_tr_t  = X_tr_t.to(DEVICE)
        y_tr_t  = y_tr_t.to(DEVICE)
        X_val_t = X_val_t.to(DEVICE)
        y_val_t = y_val_t.to(DEVICE)

    best_val_acc = -1.0
    best_state   = None
    no_improve   = 0

    for epoch in range(1, BACKBONE_EPOCHS + 1):

        model.train()
        n = X_tr_t.size(0)
        perm = torch.randperm(n, device=X_tr_t.device)
        total_loss = 0.0

        for i in range(0, n, BACKBONE_BATCH):
            idx = perm[i:i + BACKBONE_BATCH]
            xb = X_tr_t.index_select(0, idx)
            yb = y_tr_t.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with _autocast_ctx(dtype=torch.float16):
                    logits, _ = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * yb.size(0)

        model.eval()
        with torch.no_grad():
            n_val = X_val_t.size(0)
            n_correct = 0
            for i in range(0, n_val, BACKBONE_BATCH):
                xb = X_val_t[i:i + BACKBONE_BATCH]
                yb = y_val_t[i:i + BACKBONE_BATCH]
                if use_amp:
                    with _autocast_ctx(dtype=torch.float16):
                        logits, _ = model(xb)
                else:
                    logits, _ = model(xb)
                preds = logits.argmax(dim=1)
                n_correct += int((preds == yb).sum().item())
            val_acc = n_correct / n_val

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"      ep{epoch:3d}  loss={total_loss/n:.4f}  "
                  f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}  "
                  f"patience={no_improve}/{BACKBONE_PATIENCE}")

        if no_improve >= BACKBONE_PATIENCE:
            if verbose:
                print(f"      Early stop tại epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return model, best_val_acc

@torch.no_grad()
def extract_embeddings(model: nn.Module,
                       X: np.ndarray,
                       batch_size: int = 256) -> np.ndarray:
    """
    Map (N, WINDOW_SIZE, 9) → (N, 128) embedding.

    Backbone đã ở DEVICE; X được move sang DEVICE theo batch.
    """
    model.eval()
    X_t = torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1))).float()

    use_amp = USE_AMP and DEVICE.type == "cuda"

    embs = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(DEVICE, non_blocking=False)
        if use_amp:
            with _autocast_ctx(dtype=torch.float16):
                e = model.embed(xb).float()
        else:
            e = model.embed(xb)
        embs.append(e.cpu().numpy())

    return np.concatenate(embs).astype(np.float32)
