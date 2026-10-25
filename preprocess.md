# Preprocess

Prepare **RE10K**, **DL3DV**, and **ACID** datasets for training and evaluation.

## RE10K

Training reads `data/processed_re10k/train`. Download the official trajectories and the train frames from [`mutou0308/RE10K`](https://huggingface.co/datasets/mutou0308/RE10K).

```shell
wget -O RealEstate10K.tar.gz \
  https://storage.cloud.google.com/realestate10k-public-files/RealEstate10K.tar.gz
tar -xzf RealEstate10K.tar.gz

pip install -U "huggingface_hub[cli]"
huggingface-cli download mutou0308/RE10K \
  --repo-type dataset \at
  --include "train_*.zip" \
  --local-dir /path/re10k_frames
```

- Place the converted clips at `data/processed_re10k/train/`. Each zip entry is `<seq_id>/<timestamp>.png`. Match it to `RealEstate10K/train/<seq_id>.txt`, keep timestamps present in both, and sort by timestamp. Write the frames as `00000.png`, `00001.png`, ... . The available zips give 9,487 clips and 1,265,355 frames.
- Each `metadata/<seq_id>.json` has `width`, `height`, and one entry per frame. A line in the `.txt` is `timestamp fx fy cx cy 0 0` followed by 12 row-major world-to-camera values. `fx fy cx cy` are normalized, so `fxfycxcy` is the pixel intrinsics `[fx*W, fy*H, cx*W, cy*H]`. `w2c` is that pose stored as a 4×4 OpenCV world-to-camera matrix.
- The 10 scenes under `benchmark/` use the same conversion. Their frames come from [`DavidYan2001/RealEstate10K`](https://huggingface.co/datasets/DavidYan2001/RealEstate10K) `dataset/test.tar.gz`, paired with `RealEstate10K/test/<seq_id>.txt`. The sequence ids are the filenames in `evaluation/re10k/splits/`, selected by [SEVA](https://github.com/Stability-AI/stable-virtual-camera). Evaluation reads `images/<seq_id>/` and `metadata/<seq_id>.json`.

```text
data/processed_re10k/
├── train/
│   ├── images/
│   │   └── <seq_id>/
│   │       ├── 00000.png
│   │       ├── 00001.png
│   │       └── ...
│   └── metadata/
│       ├── <seq_id>.json
│       └── ...
└── benchmark/
    ├── images/
    │   └── <seq_id>/
    │       ├── 00000.png
    │       ├── 00001.png
    │       └── ...
    └── metadata/
        ├── <seq_id>.json
        └── ...
```

## DL3DV

Training uses the [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) processed DL3DV release, splits `1K` through `7K`.

```shell
pip install -U "huggingface_hub[cli]"
huggingface-cli download zhangify/CUT3R_release \
  --repo-type dataset \
  --include "processed_dl3dv_ours_rgb_cam_1/part.tar.gz.part-*" \
  --local-dir /path/cut3r_dl3dv
cd /path/cut3r_dl3dv/processed_dl3dv_ours_rgb_cam_1
cat part.tar.gz.part-* > part.tar.gz
tar -xzf part.tar.gz
```

- Place the extracted `1K`–`7K` trees at `data/processed_dl3dv/`. The paper's training tree has 883, 999, 867, 668, 971, 993, and 997 scenes in those splits. Each camera file is `frame_XXXXX.npz` with `intrinsic` (3×3) and `pose` (4×4 camera-to-world). Frames start at `frame_00001.png`.

- Scenes whose hash is a directory under `DL3DV-10K-Benchmark/` are excluded from training. Request the official [140-scene benchmark](https://huggingface.co/datasets/DL3DV/DL3DV-10K-Benchmark) and place each scene directory there. Only the directory name is used for exclusion.
- The 10 scenes under `test/` are extracted from `DL3DV-10K-Benchmark/` using the scene hashes in `evaluation/dl3dv/splits/` selected by [SEVA](https://github.com/Stability-AI/stable-virtual-camera). Each file there is `<scene_hash>.json`. Copy that scene directory to `test/<scene_hash>/`. Evaluation reads `gaussian_splat/transforms.json` and `gaussian_splat/images_4/`.

```text
data/processed_dl3dv/
├── 1K/
│   └── <scene_hash>/
│       └── dense/
│           ├── rgb/
│           │   ├── frame_00001.png
│           │   ├── frame_00002.png
│           │   └── ...
│           └── cam/
│               ├── frame_00001.npz
│               ├── frame_00002.npz
│               └── ...
├── 2K/
├── 3K/
├── 4K/
├── 5K/
├── 6K/
├── 7K/
├── DL3DV-10K-Benchmark/
│   ├── <scene_hash>/
│   └── ...
└── test/
    └── <scene_hash>/
        └── gaussian_splat/
            ├── transforms.json
            └── images_4/
                ├── frame_00001.png
                ├── frame_00002.png
                └── ...
```

## ACID

Evaluation reads `data/processed_acid/benchmark`. Download the test split from [`lhmd/acid_torch`](https://huggingface.co/datasets/lhmd/acid_torch) and parse it. The metadata schema matches RE10K.

```shell
pip install -U "huggingface_hub[cli]"
huggingface-cli download lhmd/acid_torch \
  --repo-type dataset \
  --include "test/*" \
  --local-dir /path/acid_torch
```

- The 10 scenes  with substantial viewpoint variation under `benchmark/` are extracted from the parsed sequences using the filenames in `evaluation/acid/splits/`. Evaluation reads `images/<seq_id>/` and `metadata/<seq_id>.json`.

```text
data/processed_acid/
└── benchmark/
    ├── images/
    │   └── <seq_id>/
    │       ├── 00000.jpg
    │       ├── 00001.jpg
    │       └── ...
    └── metadata/
        ├── <seq_id>.json
        └── ...
```
