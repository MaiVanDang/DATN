"""
Unified TFLite export: BackboneCNN / ConvLSTM / ConvLSTM-Bi

Usage (run from repo root f:/DATN):
    python ml_pipeline/export/export_tflite.py

To switch model, change MODEL on line 16.
"""

import os, sys
import numpy as np

MODEL = "cnn"
TRAIN_MODE = "walking"

assert MODEL in ("cnn", "convlstm", "convlstm_bi"), f"Unknown model: {MODEL}"

try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")
except ImportError:
    sys.exit("Need: pip install tensorflow")

import torch

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = os.path.join(ROOT, "web_demo", "artifacts")

MODEL_DIR = {
    "cnn":         os.path.join(ARTIFACTS, "cnn_v2"),
    "convlstm":    os.path.join(ARTIFACTS, "convlstm_v2"),
    "convlstm_bi": os.path.join(ARTIFACTS, "convlstm_bi_v2"),
}[MODEL]

PT_PATH    = os.path.join(MODEL_DIR, f"models_{TRAIN_MODE}", "backbone.pt")
EXPORT_DIR = os.path.join(MODEL_DIR, f"export_{TRAIN_MODE}")
OUT_TFLITE = os.path.join(EXPORT_DIR, "backbone.tflite")
ASSETS_DIR = os.path.join(ROOT, "android_app", "B_authenticator_app",
                           "app", "src", "main", "assets", TRAIN_MODE)

def to_np(tensor):
    """Tensor → numpy, compatible with NumPy 2.x."""
    return np.array(tensor.detach().float().cpu().tolist(), dtype=np.float32)

print(f"\n[1] Loading {MODEL} -> {PT_PATH}")
sd = torch.load(PT_PATH, map_location="cpu")
print("    Keys:", list(sd.keys()))

print("\n[2] Building Keras model ...")

def _conv_block_2layer(inp):
    """2 conv blocks dùng chung cho cả 3 model. Đầu ra: [batch, 50, 128]"""
    x = tf.keras.layers.Conv1D(64,  5, padding="same", use_bias=True, name="conv1")(inp)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=True, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool2")(x)
    return x

inp = tf.keras.Input(shape=(200, 9), name="input")
x   = _conv_block_2layer(inp)

if MODEL == "cnn":
    x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=True, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

elif MODEL == "convlstm":
    x = tf.keras.layers.LSTM(128, return_sequences=False, unroll=True, name="lstm")(x)

elif MODEL == "convlstm_bi":
    x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False, unroll=True, name="lstm"),
            merge_mode="concat", name="bilstm")(x)

keras_model = tf.keras.Model(inp, x, name=f"backbone_{MODEL}")
print(f"    Output shape: {keras_model.output_shape}")

print("\n[3] Transferring weights ...")

def set_conv(layer, w_key, b_key):
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

conv_prefix = "encoder" if MODEL == "cnn" else "encoder.conv"

set_conv(keras_model.get_layer("conv1"), f"{conv_prefix}.0.weight", f"{conv_prefix}.0.bias")
set_bn  (keras_model.get_layer("bn1"),   f"{conv_prefix}.1")
set_conv(keras_model.get_layer("conv2"), f"{conv_prefix}.4.weight", f"{conv_prefix}.4.bias")
set_bn  (keras_model.get_layer("bn2"),   f"{conv_prefix}.5")

if MODEL == "cnn":
    set_conv(keras_model.get_layer("conv3"), "encoder.8.weight", "encoder.8.bias")
    set_bn  (keras_model.get_layer("bn3"),   "encoder.9")

elif MODEL == "convlstm":
    lstm_layer = keras_model.get_layer("lstm")
    kernel     = to_np(sd["encoder.lstm.weight_ih_l0"]).T
    recurrent  = to_np(sd["encoder.lstm.weight_hh_l0"]).T
    bias       = (to_np(sd["encoder.lstm.bias_ih_l0"])
                + to_np(sd["encoder.lstm.bias_hh_l0"]))
    lstm_layer.set_weights([kernel, recurrent, bias])

elif MODEL == "convlstm_bi":
    bidir_layer = keras_model.get_layer("bilstm")
    def _lstm_weights(ih_key, hh_key, bih_key, bhh_key):
        return [
            to_np(sd[ih_key]).T,
            to_np(sd[hh_key]).T,
            to_np(sd[bih_key]) + to_np(sd[bhh_key]),
        ]
    fwd = _lstm_weights("encoder.lstm.weight_ih_l0",
                        "encoder.lstm.weight_hh_l0",
                        "encoder.lstm.bias_ih_l0",
                        "encoder.lstm.bias_hh_l0")
    bwd = _lstm_weights("encoder.lstm.weight_ih_l0_reverse",
                        "encoder.lstm.weight_hh_l0_reverse",
                        "encoder.lstm.bias_ih_l0_reverse",
                        "encoder.lstm.bias_hh_l0_reverse")
    bidir_layer.set_weights(fwd + bwd)

print("    Weights OK")

print("\n[4] Smoke test ...")
np.random.seed(0)
x_test  = np.random.randn(1, 200, 9).astype(np.float32)
out     = keras_model.predict(x_test, verbose=0)
assert out.shape == (1, 128), f"Bad output shape: {out.shape}"
print(f"    Output shape: {out.shape}  norm={np.linalg.norm(out):.3f}")

print("\n[5] Converting to TFLite ...")
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = []
tflite_bytes = converter.convert()

with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_bytes)
print(f"    Saved: {OUT_TFLITE}  ({len(tflite_bytes)//1024} KB)")

interp = tf.lite.Interpreter(model_content=tflite_bytes)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"    Input : {inp_d['shape']}  {inp_d['dtype'].__name__}")
print(f"    Output: {out_d['shape']}  {out_d['dtype'].__name__}")

interp.set_tensor(inp_d["index"], x_test)
interp.invoke()
tflite_out = interp.get_tensor(out_d["index"])
diff = float(np.abs(tflite_out - out).max())
print(f"    Keras vs TFLite diff: {diff:.2e}  (must be < 1e-4)")
if diff > 1e-4:
    sys.exit(f"ERROR: diff too large ({diff:.6f})")

import shutil

COPY_LIST = [
    ("backbone.tflite",            OUT_TFLITE),
    ("impostor_pool_inertial.npy", os.path.join(EXPORT_DIR, "impostor_pool_inertial.npy")),
    ("impostor_pool_touch.npy",    os.path.join(EXPORT_DIR, "impostor_pool_touch.npy")),
    ("touch_scaler.json",          os.path.join(EXPORT_DIR, "touch_scaler.json")),
]

print(f"\n[6] Copying to {ASSETS_DIR}:")
os.makedirs(ASSETS_DIR, exist_ok=True)
for dest_name, src in COPY_LIST:
    dst = os.path.join(ASSETS_DIR, dest_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"    OK   {dest_name}")
    else:
        print(f"    SKIP (not found): {src}")

import json, datetime
manifest_path = os.path.join(ASSETS_DIR, "export_manifest.json")
manifest = {}
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

manifest.update({
    "model_file": "backbone.tflite",
    "model_kind": f"backbone_encoder_{MODEL}",
    "source_checkpoint": f"{MODEL}_v2 {TRAIN_MODE}",
    "exported_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "export_pipeline_version": "v5-unified-tflite",
    "context_mode": TRAIN_MODE
})

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"    OK   export_manifest.json (updated)")

print(f"\n=== Done [{MODEL}] -- rebuild app in Android Studio ===")
