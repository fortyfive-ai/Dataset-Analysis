# Robot Training Dataset Structures

A practical guide to how robot training data is physically stored on disk: what folders you see, what's inside each file, and how the popular formats compare.

---

## TL;DR — The three things to remember

1. **Every robot dataset records 4 things**: camera footage, robot state, action commands, and a task instruction. Every format is just a different way of slicing those 4 things into files.
2. **LeRobot (Parquet + MP4) has won 2025-2026.** Almost every new project and every major Chinese humanoid company that publishes data uses it: Physical Intelligence, NVIDIA GR00T, Unitree, Robotera, Galaxea, AgiBot (2026 release), Fourier (via converter).
3. **HDF5 is still the academic / sim default.** RLDS is Google's legacy stack with stalled growth. Custom binary formats (AgiBot's older HDF5+tar, 1X's tokenized .bin) exist for very large or very specialized cases.

---

## 1. The 4 elements every dataset stores

| Element | What it is | Example |
|---|---|---|
| 🎥 **Visual** | Camera streams (and sometimes depth) | head cam, two wrist cams, RGB at 30 fps |
| 🦾 **State** | What the robot senses about itself right now | joint angles, end-effector pose, gripper opening, head/waist pose |
| 🎮 **Action** | What the controller is telling the robot to do next | target joint angles, target end-effector pose |
| 📝 **Task** | A natural-language description of the goal | "place the red block in the bin" |

The difference between formats is **which file each of these 4 elements goes into**:

| Format | Visual | State | Action | Task |
|---|---|---|---|---|
| **LeRobot** | MP4 (separate) | Parquet column | Parquet column | tasks.parquet + index |
| **HDF5 (RoboMimic)** | array inside HDF5 | array inside HDF5 | array inside HDF5 | JSON string in attrs |
| **HDF5 (ALOHA)** | raw pixel array inside HDF5 | array inside HDF5 | array inside HDF5 | usually none, filename only |
| **RLDS / TFRecord** | bytes inside record | bytes inside record | bytes inside record | `instruction` field in record |
| **AgiBot (HDF5+tar)** | separate MP4 + depth PNGs | deeply nested HDF5 | deeply nested HDF5 | separate `task_*.json` with frame ranges |
| **1X custom** | tokenized `.bin` | `.bin` array | `.bin` array | metadata.json |

---

## 2. The formats, ranked by adoption

### 🥇 #1 — LeRobot (Parquet + MP4)

**Who uses it:** Hugging Face, Physical Intelligence (openpi), NVIDIA GR00T-N1, Unitree (40+ datasets on HF), Robotera, Galaxea AI, Fourier (via converter), AgiBot (2026 release), DROID (76k Franka trajectories), and the broad community on HF Hub.

**What it looks like on disk:**

```
my_dataset/
├── data/chunk-000/file-000.parquet           ← numbers (state, action, timestamps)
├── videos/observation.images.cam_high/
│   └── chunk-000/file-000.mp4                ← visual (MP4 video)
└── meta/
    ├── info.json                              ← dataset description
    ├── stats.json                             ← per-feature mean/std
    ├── tasks.parquet                          ← task strings ↔ task_index
    └── episodes/chunk-000/file-000.parquet    ← episode index (which frames, which mp4 segment)
```

**The smart part:** numbers and pixels live in separate files. A single Parquet file holds rows for many episodes, and a single MP4 holds video for many episodes concatenated together. Episode boundaries are reconstructed from the metadata index, not from filenames.

**Pros**
- ✅ MP4 compression cuts storage roughly **10×** vs raw HDF5 pixels
- ✅ Parquet is columnar — read only the `action` column for fast loading
- ✅ Streamable from HuggingFace Hub (`StreamingLeRobotDataset`) — no full download required
- ✅ Scales to millions of episodes without filesystem pressure
- ✅ Excellent ecosystem: ACT, Diffusion Policy, π0, π0.5, GR00T all consume it natively

**Cons**
- ❌ MP4 decode adds CPU/GPU work at read time vs reading raw arrays
- ❌ Fast-evolving spec (v1 → v2 → v2.1 → v3.0 in ~12 months)
- ❌ `observation.state` is a flat vector — can't represent AgiBot-style nested state without flattening

**Why people choose it:**
> "I'm building a VLA model on a large dataset, I need streaming, I want a small disk footprint, and I want my code to interop with the major open-source policies." — the default reason in 2025+.

---

### 🥈 #2 — HDF5 (RoboMimic style — one file holds everything)

**Who uses it:** NVIDIA Isaac Lab (`record_demos.py` writes this format), MimicGen, RoboCasa, robosuite, Diffusion Policy (loader supports it).

**What it looks like:**

```
my_dataset/
└── demo.hdf5                       ← the entire dataset is one file
    ├── /data
    │   ├── attrs                   ← global metadata (env_args as JSON)
    │   ├── /demo_0
    │   │   ├── attrs               ← num_samples, model_file
    │   │   ├── /states             (N, D)   simulator states
    │   │   ├── /actions            (N, A)   normalized to [-1, 1]
    │   │   ├── /rewards            (N,)
    │   │   ├── /dones              (N,)
    │   │   ├── /obs/agentview_image
    │   │   └── /obs/robot0_eye_in_hand_image
    │   ├── /demo_1
    │   └── /demo_2
    └── /mask
        ├── /train                  ← which demos are train split
        └── /valid                  ← which demos are val split
```

**Pros**
- ✅ One file ships everything — easy to distribute
- ✅ Cross-language (C/C++/Java/Python all read HDF5)
- ✅ Random access by demo path (`/data/demo_42/actions`)
- ✅ Tightly integrated with the MuJoCo / Isaac simulation stack
- ✅ Mature tooling, used in academia for years

**Cons**
- ❌ **No native video codec** — images stored as raw uint8 arrays, file size balloons
- ❌ Single-writer concurrency limit
- ❌ No streaming — must download the entire file before reading
- ❌ Awkward at >1 TB scale

**Why people choose it:**
> "I'm doing imitation learning in simulation, datasets are in the thousands of demos, and I want to plug straight into robosuite or Isaac Lab." — the sim-research default.

---

### 🥉 #3 — HDF5 (ALOHA style — one file per episode)

**Who uses it:** ALOHA, Mobile ALOHA, many Stanford teleop projects.

**What it looks like:**

```
my_dataset/
├── episode_0.hdf5
├── episode_1.hdf5
├── episode_2.hdf5
└── ...
```

Inside each `.hdf5`:

```
episode_0.hdf5
├── /images
│   ├── /cam_high           (500, 480, 640, 3) uint8     ← raw pixels, no codec
│   ├── /cam_low            (500, 480, 640, 3) uint8
│   ├── /cam_left_wrist     (500, 480, 640, 3) uint8
│   └── /cam_right_wrist    (500, 480, 640, 3) uint8
├── /qpos                   (500, 14) float64             ← joint positions
├── /qvel                   (500, 14) float64             ← joint velocities
└── /action                 (500, 14) float64             ← target joint positions
```

**Pros**
- ✅ Each episode is independent — corruption in one file doesn't break others
- ✅ Convenient for live recording (no shared write lock)
- ✅ Filename-based addressing is intuitive

**Cons**
- ❌ Raw pixel storage means 4 cameras × 1 minute can easily be several GB
- ❌ Hundreds of thousands of small files crush most filesystems
- ❌ No standard schema — every project invents its own field names

**Why people choose it:**
> "I'm running real-time teleoperation and need to write each episode as it finishes, independently."

---

### #4 — RLDS / TFRecord (the Google stack)

**Who uses it:** Google DeepMind (RT-1, RT-2, RT-X), the Open X-Embodiment consortium (34 labs, 60 datasets unified, >1M trajectories), BC-Z.

**What it looks like:**

```
my_dataset/
├── dataset_info.json
├── features.json
└── my_dataset-train.tfrecord-00000-of-00010
    my_dataset-train.tfrecord-00001-of-00010
    ...
    my_dataset-train.tfrecord-00009-of-00010
```

Each `.tfrecord` is a sequential binary stream of records — think of it as a tape:

```
record #1:   observation { image, instruction }, action, is_first=True
record #2:   ...
record #N:   ..., is_last=True
record #N+1: is_first=True       ← next episode begins
```

Episode boundaries are marked by the `is_first` and `is_last` flags inside each record, not by file structure.

**Pros**
- ✅ Unifies 60 datasets from 34 labs under one schema (Open X-Embodiment's value proposition)
- ✅ Deep TensorFlow / TFDS integration
- ✅ Strict schema enforcement — no missing fields
- ✅ Scales well on TPUs with parallel readers

**Cons**
- ❌ Requires the TF library to read — can't open with anything else
- ❌ Sequential reads are fast, **random access is awkward**
- ❌ "Cross-embodiment" alignment is coarse — coordinate frames not actually unified across robots
- ❌ Growth has stalled — almost no new projects pick this in 2025+

**Why people choose it:**
> "I want to mix data from many labs, and we're already on Google infra / TFDS." — primarily a legacy and ecosystem reason.

---

### #5 — AgiBot HDF5 + tar shards (massive-scale real-world data)

**Who uses it:** AgiBot World Alpha and Beta (the world's largest open robot dataset — **1,003,672 trajectories, 2,976 hours, 43.8 TB**).

**What it looks like:**

```
AgiBotWorld/
├── observations/327/648642/        ← task_id / episode_id
│   ├── depth/                      ← one PNG per frame
│   └── videos/{head, hand_left, hand_right}.mp4
├── parameters/327/648642/camera/   ← intrinsics + extrinsics JSON
├── proprio_stats/327/648642/proprio_stats.h5    ← state + action HDF5
└── task_info/task_327.json         ← per-episode language annotations
```

To make millions of files transferable, everything is bundled into ~48 GB tar shards:

```
observations/327/648642-685032.tar    ~48 GB
proprio_stats/648533-923022.tar       ~48 GB
```

The `proprio_stats.h5` schema is unusually rich — **6 categories of state** (joint, end-effector, effector, head, waist, robot), each with sub-fields for position/velocity/effort. This is far more structured than typical robot datasets.

The `task_info/task_*.json` carries **frame-range-level language annotations** — a single episode can be split into multiple sub-tasks ("pick up onion" frames 0-435, "place in bag" frames 435-619).

**Pros**
- ✅ Handles PB-scale distribution (tar shards make HuggingFace transfers efficient)
- ✅ Extremely rich state schema covering whole-body modalities
- ✅ Frame-level task annotations enable hierarchical learning
- ✅ The only published dataset of this scale

**Cons**
- ❌ Requires custom parsing — no off-the-shelf loader
- ❌ Deeply nested HDF5 has a learning curve
- ❌ AgiBot themselves migrated to LeRobot for the 2026 release

**Why people choose it:**
> "I had hundreds of thousands of real-robot trajectories to release in 2024, and LeRobot wasn't mature enough yet." — primarily a **timing** issue.

---

### #6 — 1X custom binary (single-vendor, world-model focused)

**Who uses it:** 1X Technologies (EVE humanoid, world-model training).

**What it looks like:**

```
worldmodel/train_v2.0/
├── video_*.bin           ← visual already tokenized via NVIDIA Cosmos
├── states_*.bin          ← 25-D float32 state arrays
├── segment_idx_*.bin     ← episode boundary indices
└── metadata.json
```

The state vector is fixed at 25 dimensions: 21 joint angles + 2 hand-closure scalars + linear velocity + angular velocity.

**Pros**
- ✅ **Tiny on disk** — tokenized video can be ~10× smaller than MP4
- ✅ Near-zero preprocessing at training time — model consumes tokens directly
- ✅ Purpose-built for world-model training

**Cons**
- ❌ Original pixels are **lost** — tokenization is one-way
- ❌ Single-vendor, **zero ecosystem**
- ❌ Changing the tokenizer means regenerating the entire dataset

**Why people choose it:**
> "We're training a world model. Whether the original pixels can be recovered doesn't matter to us." — extreme specialization.

---

## 3. Side-by-side comparison

| Rank | Format | Scale | Disk size | Load speed | Ecosystem | Best fit |
|---|---|---|---|---|---|---|
| 🥇 | **LeRobot** | millions | small (MP4) | fast (columnar + streaming) | **excellent** | modern VLA training |
| 🥈 | **HDF5 (RoboMimic)** | tens of thousands | medium | medium | strong (sim research) | sim-based imitation learning |
| 🥉 | **HDF5 (ALOHA)** | thousands | large | medium | medium | live teleop recording |
| 4 | **RLDS / TFRecord** | millions (OXE) | medium | fast seq / slow random | declining | Google ecosystem compat |
| 5 | **AgiBot HDF5+tar** | millions | large | medium | single-vendor (migrating) | massive real-world releases |
| 6 | **1X custom** | hundreds of thousands | **tiny** | **very fast** | none | world-model training |

---

## 4. Picking a format — decision tree

```
What's your project?

├── Large-scale VLA training, scaling to millions of episodes
│   └─→ ✅ LeRobot
│
├── Sim-based imitation learning, tens of thousands of demos
│   └─→ ✅ HDF5 (RoboMimic style)
│
├── Live teleop recording, each episode saved independently
│   └─→ ✅ HDF5 (ALOHA style)
│
├── Need to mix with Open X-Embodiment / Google stack
│   └─→ ✅ RLDS / TFRecord
│
├── Training a world model, pixels can be tokenized upfront
│   └─→ Custom binary (see 1X for reference)
│
└── Not sure?
    └─→ Pick LeRobot. You'll almost certainly be right.
```

---

## 5. The trend

```
2021 ────────► 2023 ────────► 2024 ────────► 2025-2026
RLDS released   ALOHA          LeRobot v1     LeRobot v3
(Google)        HDF5 widespread released      becomes the de-facto standard
                                              ↑
                          Almost everyone is migrating here
                          (AgiBot, Unitree, Pi, NVIDIA, Galaxea, Robotera...)
```

**For anyone building a robot data platform, this implies a clear priority order:**

1. **LeRobot is mandatory** — both read and write, supporting versions v1 / v2.1 / v3.0
2. **HDF5 (both styles) is mandatory** — academic legacy plus AgiBot-scale real data
3. **rosbag / MCAP → LeRobot conversion** is a high-value pain point (raw robot logs aren't in this list but show up everywhere in practice)
4. **RLDS** is medium priority (large but stagnant)
5. **Custom binary** is low priority (single-vendor)

---

## What this analysis does not cover

- Quantitative benchmarks (read throughput, on-disk size per hour) — published numbers were not surfaced.
- Internal formats of companies that haven't disclosed them (Tesla, Figure, Boston Dynamics, XPeng, etc.).
- ROS bag (`.bag`) and MCAP — common for raw robot logs but distinct from training datasets.
- DeepMind's RT-2 / Gemini Robotics internal pipeline beyond what Open X-Embodiment covers.

## Sources

Primary sources cited inline throughout: official repos (huggingface/lerobot, OpenDriveLab/AgiBot-World, unitreerobotics/unitree_lerobot, ARISE-Initiative/robomimic, google-research/rlds, 1x-technologies/1xgpt), papers (Open X-Embodiment, AgiBot World Colosseo, RLDS, RoboMimic), and HuggingFace dataset cards. Items that could not be verified against a primary source are marked `[UNVERIFIED]`; items companies have not publicly disclosed are marked `[UNDISCLOSED]`.
