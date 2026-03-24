#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""Compare weights from our JSON+bin format vs the actual TFLite model."""
import numpy as np
import json

TFLITE_PATH = "/Users/sven/code/micro-facemesh/weights/face_landmarks_detector.tflite"
WEIGHTS_JSON = "/Users/sven/code/micro-facemesh/weights/face_landmarks_weights.json"
WEIGHTS_BIN = "/Users/sven/code/micro-facemesh/weights/face_landmarks_weights.bin"

# Load TFLite model and extract all tensors
import tensorflow.lite as tflite
interp = tflite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()

tensor_details = interp.get_tensor_details()
print(f"TFLite model has {len(tensor_details)} tensors")

# Build lookup from TFLite
tflite_tensors = {}
for td in tensor_details:
    name = td['name']
    try:
        data = interp.get_tensor(td['index'])
        tflite_tensors[name] = data
    except:
        pass

# Load our JSON+bin weights
meta = json.load(open(WEIGHTS_JSON))
buf = open(WEIGHTS_BIN, 'rb').read()
buf_np = np.frombuffer(buf, dtype=np.float32)

print(f"\nOur weights: {len(meta['keys'])} keys")

# Compare specific important weights
check_keys = [
    ('conv2d_81', 'Conv2D'),   # Stem conv
    ('batch_normalization_132', ''),  # Stem BN/bias
    ('p_re_lu_126', ''),       # Stem PReLU
    ('conv2d_82', 'Conv2D'),   # Stage1 block1 narrow
    ('conv2d_150', 'Conv2D'),  # LM head
    ('conv2d_150', 'BiasAdd'), # LM head bias
    ('conv2d_151', 'Conv2D'),  # Presence head (inner model)
    ('conv2d_151', 'BiasAdd'), # Presence head bias (inner model)
]

for search_terms in check_keys:
    # Find in our weights
    our_idx = None
    for i, k in enumerate(meta['keys']):
        if all(s in k for s in search_terms if s):
            our_idx = i
            break

    if our_idx is None:
        print(f"\n{search_terms}: NOT FOUND in our weights")
        continue

    our_key = meta['keys'][our_idx]
    our_shape = meta['shapes'][our_idx]
    our_offset = meta['offsets'][our_idx]
    our_size = 1
    for s in our_shape:
        our_size *= s
    our_data = np.frombuffer(buf, dtype=np.float32, count=our_size, offset=our_offset)

    # Find in TFLite
    tflite_match = None
    for name, data in tflite_tensors.items():
        if all(s in name for s in search_terms if s):
            tflite_match = (name, data)
            break

    if tflite_match is None:
        # Try shorter search
        for name, data in tflite_tensors.items():
            if search_terms[0] in name:
                tflite_match = (name, data)
                break

    print(f"\n=== {search_terms} ===")
    print(f"  Our key: {our_key[:80]}...")
    print(f"  Our shape: {our_shape}, range: [{our_data.min():.6f}, {our_data.max():.6f}]")
    print(f"  Our first 5: {our_data[:5]}")

    if tflite_match:
        tname, tdata = tflite_match
        tflat = tdata.flatten()
        print(f"  TFLite key: {tname[:80]}...")
        print(f"  TFLite shape: {list(tdata.shape)}, range: [{tflat.min():.6f}, {tflat.max():.6f}]")
        print(f"  TFLite first 5: {tflat[:5]}")

        if our_data.shape == tflat.shape:
            diff = np.abs(our_data - tflat)
            print(f"  Max diff: {diff.max():.8f}, Mean diff: {diff.mean():.8f}")
        else:
            print(f"  Shape mismatch: ours={our_data.shape} vs tflite={tflat.shape}")
    else:
        print(f"  TFLite: NO MATCH FOUND")
        # Print all tflite keys that contain first search term
        matches = [n for n in tflite_tensors.keys() if search_terms[0] in n]
        if matches:
            print(f"  Partial matches: {matches[:5]}")
