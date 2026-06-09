#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_embeddings.py
====================================================================
CHAY LOCAL (may co PyTorch + code pipeline cua ban) de XUAT EMBEDDING
cua CA 3 BACKBONE × 2 CONTEXT MODE ra file .npz, roi gui lai cho minh
phan tich (chon ham cham diem tot nhat tren tung backbone).

KHONG train lai gi — chi nap checkpoint da co, chay forward, luu embedding.

------------------------------------------------------------------
CHUAN BI (dat script o thu muc goc project, canh cac module .py):
  - config.py, dataset.py, models.py, backbone_train.py ... (pipeline cua ban)
  - processed/userX/...                  (du lieu da tien xu ly)
  - cac thu muc backbone da giai nen, vi du:
        cnn/models_all/backbone.pt,        cnn/models_walking/backbone.pt
        convlstm/models_all/backbone.pt,   convlstm/models_walking/backbone.pt
        convlstm_bi/models_all/backbone.pt,convlstm_bi/models_walking/backbone.pt

CACH CHAY:
  python export_embeddings.py --data_dir processed --root .

  (--root la noi chua 3 thu muc cnn/ convlstm/ convlstm_bi/.
   Neu ten thu muc khac, sua DICT `BACKBONES` ben duoi.)

KET QUA: thu muc  emb_out/  gom 6 file:
   emb_cnn_all.npz  emb_cnn_walking.npz
   emb_convlstm_all.npz  emb_convlstm_walking.npz
   emb_convlstm_bi_all.npz  emb_convlstm_bi_walking.npz
=> NEN emb_out/ thanh 1 zip va gui lai cho minh.
------------------------------------------------------------------
"""
import argparse, json, traceback
from pathlib import Path
import numpy as np

BACKBONES = {
    "cnn":         ("cnn",         "cnn"),
    "convlstm":    ("convlstm",    "convlstm"),
    "convlstm_bi": ("convlstm_bi", "convlstm_bi"),
}
MODES = ["all", "walking"]
EXCLUDE_USERS = {"user11", "user17"}


def log(m): print(m, flush=True)


def import_pipeline():
    """Import cac ham can tu code pipeline cua ban (phai chay tu thu muc project)."""
    try:
        import torch
    except Exception:
        log("‼  Khong import duoc torch. Hay chay tren may/Colab co PyTorch.")
        raise
    try:
        from models import build_backbone
        from dataset import load_users
        from backbone_train import extract_embeddings, DEVICE
        return build_backbone, load_users, extract_embeddings, DEVICE
    except Exception:
        log("‼  Khong import duoc module pipeline (models/dataset/backbone_train).")
        log("   -> Hay dat export_embeddings.py o THU MUC GOC project, canh cac file .py do.")
        raise


def session_of(labels):
    """Nhan 'userX_session_Y' -> 'session_Y' (giu 2 token cuoi)."""
    return np.array(["_".join(str(s).split("_")[-2:]) for s in labels])


def export_one(build_backbone, load_users, extract_embeddings, DEVICE,
               data_dir, root, arch_key, folder, arch_name, mode, outdir):
    import torch
    ckpt = Path(root) / folder / f"models_{mode}" / "backbone.pt"
    meta_p = Path(root) / folder / f"export_{mode}" / "backbone_metadata.json"
    if not ckpt.exists():
        log(f"  [bo qua] khong thay {ckpt}")
        return None
    meta = json.load(open(meta_p, encoding="utf-8")) if meta_p.exists() else {}
    seen = meta.get("users_trained", [])
    D = int(meta.get("embed_dim", 128))

    bb = build_backbone(arch_name, n_users=max(1, len(seen)), embed_dim=D).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    bb.load_state_dict(state)
    bb.eval()

    users = load_users(Path(data_dir), exclude_users=EXCLUDE_USERS, context_mode=mode)
    out = {}
    n_tot = 0
    for u in sorted(users):
        X = users[u]["X"]
        Z = extract_embeddings(bb, X).astype(np.float32)
        sess = session_of(users[u]["session"])
        out[f"{u}||Z"] = Z
        out[f"{u}||sess"] = sess.astype(str)
        n_tot += len(Z)
    out["__meta__"] = np.array([json.dumps(
        {"arch": arch_key, "mode": mode, "embed_dim": D,
         "seen": seen, "users": sorted(users)}, ensure_ascii=False)])
    fp = Path(outdir) / f"emb_{arch_key}_{mode}.npz"
    np.savez_compressed(fp, **out)
    log(f"  ✔ {fp.name}: {len(users)} user, {n_tot} window, dim={D}")
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="processed",
                    help="thu muc chua userX/ (vd processed hoac duong dan Drive)")
    ap.add_argument("--root", default=".",
                    help="thu muc chua cnn/ convlstm/ convlstm_bi/")
    ap.add_argument("--outdir", default="emb_out")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    build_backbone, load_users, extract_embeddings, DEVICE = import_pipeline()
    log(f"# DEVICE = {DEVICE}")
    log(f"# data_dir = {args.data_dir} | root = {args.root}")

    made = []
    for arch_key, (folder, arch_name) in BACKBONES.items():
        for mode in MODES:
            log(f"\n[{arch_key} · {mode}]")
            try:
                fp = export_one(build_backbone, load_users, extract_embeddings,
                                DEVICE, args.data_dir, args.root,
                                arch_key, folder, arch_name, mode, args.outdir)
                if fp:
                    made.append(fp)
            except Exception:
                log("  !! loi o cau hinh nay:")
                traceback.print_exc()

    log("\n" + "=" * 60)
    log(f"XONG. Da tao {len(made)} file trong {args.outdir}/")
    for f in made:
        log("   " + str(f))
    log("\n-> Nen thu muc emb_out/ thanh 1 zip va gui lai de minh phan tich.")
    log("   (Linux/Mac:  zip -r emb_out.zip emb_out )")


if __name__ == "__main__":
    main()
