"""
Export BackboneCNN: PyTorch checkpoint -> TFLite (NWC format)

Run from repo root (f:/DATN):
    python ml_pipeline/export/export_tflite_cnn.py

Output:
    ml_pipeline/training/cnn/models/backbone.tflite
"""

import os, sys
import numpy as np

# ── Check TensorFlow ──────────────────────────────────────────────────────
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")
except ImportError:
    sys.exit("Need TensorFlow: pip install tensorflow==2.13.*")

import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PT_PATH    = os.path.join(REPO_ROOT, "ml_pipeline", "training", "cnn", "models", "backbone.pt")
OUT_TFLITE = os.path.join(REPO_ROOT, "ml_pipeline", "training", "cnn", "models", "backbone.tflite")
ASSETS_DIR = os.path.join(REPO_ROOT, "android_app", "B_authenticator_app", "app", "src", "main", "assets")

# ── 1. BackboneCNN (PyTorch, NCW format) ──────────────────────────────────
class BackboneCNN(nn.Module):
    def __init__(self, in_ch=9, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch,     64,        kernel_size=5, padding=2),  # 0
            nn.BatchNorm1d(64),                                          # 1
            nn.ReLU(),                                                    # 2
            nn.MaxPool1d(2),                                             # 3
            nn.Conv1d(64,       128,       kernel_size=3, padding=1),   # 4
            nn.BatchNorm1d(128),                                         # 5
            nn.ReLU(),                                                    # 6
            nn.MaxPool1d(2),                                             # 7
            nn.Conv1d(128,      embed_dim, kernel_size=3, padding=1),   # 8
            nn.BatchNorm1d(embed_dim),                                   # 9
            nn.ReLU(),                                                    # 10
        )
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, 23)

    def forward(self, x):   # x: [N, 9, 200]
        x = self.encoder(x)
        return self.pool(x).squeeze(-1)   # [N, 128]

print(f"Loading: {PT_PATH}")
sd = torch.load(PT_PATH, map_location="cpu")
pt_model = BackboneCNN()
pt_model.load_state_dict(sd)
pt_model.eval()
print("PyTorch model OK")

# helper: tensor -> numpy (compatible with NumPy 2.x / torch built against NumPy 1.x)
def to_np(tensor):
    return np.array(tensor.detach().float().cpu().tolist(), dtype=np.float32)

# ── 2. Build Keras model (NWC: [N, 200, 9]) ────────────────────────────────
def build_keras_backbone():
    inp = tf.keras.Input(shape=(200, 9), name="input")

    x = tf.keras.layers.Conv1D(64,  5, padding="same", use_bias=True, name="conv1")(inp)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)

    x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=True, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool2")(x)

    x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=True, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)

    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    return tf.keras.Model(inp, x, name="backbone")

keras_model = build_keras_backbone()

# ── 3. Transfer weights PyTorch -> Keras ───────────────────────────────────
def set_conv(layer, w_key, b_key):
    # PyTorch: [out, in, k]  ->  Keras: [k, in, out]
    w = to_np(sd[w_key]).transpose(2, 1, 0)
    b = to_np(sd[b_key])
    layer.set_weights([w, b])

def set_bn(layer, prefix):
    layer.set_weights([
        to_np(sd[f"{prefix}.weight"]),
        to_np(sd[f"{prefix}.bias"]),
        to_np(sd[f"{prefix}.running_mean"]),
        to_np(sd[f"{prefix}.running_var"]),
    ])

set_conv(keras_model.get_layer("conv1"), "encoder.0.weight", "encoder.0.bias")
set_bn  (keras_model.get_layer("bn1"),  "encoder.1")
set_conv(keras_model.get_layer("conv2"), "encoder.4.weight", "encoder.4.bias")
set_bn  (keras_model.get_layer("bn2"),  "encoder.5")
set_conv(keras_model.get_layer("conv3"), "encoder.8.weight", "encoder.8.bias")
set_bn  (keras_model.get_layer("bn3"),  "encoder.9")
print("Weights transferred to Keras")

# ── 4. Smoke-test Keras output shape ──────────────────────────────────────
np.random.seed(42)
x_nwc = np.random.randn(1, 200, 9).astype(np.float32)
keras_out = keras_model.predict(x_nwc, verbose=0)
assert keras_out.shape == (1, 128), f"Unexpected Keras output shape: {keras_out.shape}"
print(f"Keras output shape OK: {keras_out.shape}  norm={np.linalg.norm(keras_out):.3f}")

# ── 5. Convert to TFLite (float32) ────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = []   # keep float32
tflite_bytes = converter.convert()

with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_bytes)
print(f"Saved TFLite -> {OUT_TFLITE}  ({len(tflite_bytes)//1024} KB)")

# ── 6. Verify TFLite ───────────────────────────────────────────────────────
interp = tf.lite.Interpreter(model_content=tflite_bytes)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"TFLite input : shape={inp_d['shape']}  dtype={inp_d['dtype'].__name__}")
print(f"TFLite output: shape={out_d['shape']}  dtype={out_d['dtype'].__name__}")

interp.set_tensor(inp_d["index"], x_nwc)
interp.invoke()
tflite_out = interp.get_tensor(out_d["index"])
diff_kt = float(np.abs(tflite_out - keras_out).max())
print(f"Max diff Keras vs TFLite: {diff_kt:.2e}  (must be < 1e-5)")
if diff_kt > 1e-4:
    sys.exit(f"ERROR: Keras vs TFLite diff too large ({diff_kt:.6f})")

# ── 7. Copy assets to Android app ─────────────────────────────────────────
import shutil

EXPORT_DIR = os.path.join(REPO_ROOT, "ml_pipeline", "training", "cnn", "export")
COPY_LIST  = [
    ("backbone.tflite",            OUT_TFLITE),
    ("impostor_pool_inertial.npy", os.path.join(EXPORT_DIR, "impostor_pool_inertial.npy")),
    ("impostor_pool_touch.npy",    os.path.join(EXPORT_DIR, "impostor_pool_touch.npy")),
    ("touch_scaler.json",          os.path.join(EXPORT_DIR, "touch_scaler.json")),
]

print(f"\nCopying assets to {ASSETS_DIR}:")
os.makedirs(ASSETS_DIR, exist_ok=True)
for dest_name, src in COPY_LIST:
    dst = os.path.join(ASSETS_DIR, dest_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  OK   {dest_name}")
    else:
        print(f"  SKIP (not found): {src}")

print("\n=== Done! Rebuild the app in Android Studio ===")
