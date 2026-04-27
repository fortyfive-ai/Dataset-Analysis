# Robot Training Dataset Structures — A Fact-Only Analysis

Survey of dataset formats used to train robot policies. The goal is to keep this fact-only — every claim is sourced where possible, items the research could not verify are marked `[UNVERIFIED]`, and items companies have not publicly disclosed are marked `[UNDISCLOSED]`. Properties listed under each format are technical characteristics stated in (or directly inferable from) the cited primary sources.

Compiled 2026-04-27. Five parallel research agents covered: (1) RLDS / Open X-Embodiment, (2) HDF5-based formats, (3) LeRobot Parquet+MP4, (4) commercial humanoid companies, (5) simulation/synthetic data formats. A sixth agent fact-checked the synthesis; corrections were applied before this version.

---

## 1. RLDS / Open X-Embodiment (TFRecord-based)

### What it is
RLDS (Reinforcement Learning Datasets) is "an ecosystem for recording, replaying, manipulating, annotating and sharing data in the context of Sequential Decision Making" [Ramos et al. 2021, [arXiv:2111.02767](https://arxiv.org/abs/2111.02767)]. Built on `tf.data.Dataset` and TFDS, serialized to **TFRecord** files [Open X-Embodiment paper, [arXiv:2310.08864](https://arxiv.org/abs/2310.08864); [github.com/google-research/rlds](https://github.com/google-research/rlds)].

### Schema
- **Episode**: a `tf.data.Dataset` of Steps + optional metadata (`episode_id`, `agent_id`, `environment_config`, `experiment_id`, `invalid`).
- **Step (required fields)**: `is_first`, `is_last`.
- **Step (optional)**: `observation`, `action`, `reward`, `discount`, `is_terminal`, custom metadata.
- All steps within a dataset must share identical fields.
- `is_terminal` distinguishes true termination from truncation: `is_last=True` with `is_terminal=False` ⇒ truncated.

Source: [github.com/google-research/rlds README](https://github.com/google-research/rlds).

### Open X-Embodiment specifics
- 60 datasets unified, 34 contributing labs, >1M real robot trajectories, 22 embodiments, 527 skills, 160,266 tasks [[Open X-Embodiment project page](https://robotics-transformer-x.github.io/); [arXiv:2310.08864](https://arxiv.org/abs/2310.08864)].
- **Action space**: 7-D vector (x, y, z, roll, pitch, yaw, gripper), discretized in RT-X into 256 bins across 8 dimensions including episode-termination [[arxiv.org/html/2310.08864v3](https://arxiv.org/html/2310.08864v3)]. Unused dims are zero-filled.
- **Observation**: one canonical RGB camera per dataset + task string. Wrist/in-hand cameras and depth are present in some source datasets but **not consumed** by the released RT-X model [[github.com/google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment) README].
- **Cross-embodiment alignment**: "coarse" — actions normalized per dataset before discretization, then de-normalized per embodiment at inference. Coordinate frames are not aligned across robots [[arxiv.org/html/2310.08864v3](https://arxiv.org/html/2310.08864v3)].

### Adopters
Google / Google DeepMind (RT-1, RT-2, RT-X, QT-Opt, Language Table); UC Berkeley (Bridge, Autolab UR5, Cable Routing); University of Freiburg (TACO); KAIST (Jaco Play); Stanford / UT Austin / NYU (RoboTurk, NYU VINN, Austin VIOLA, TOTO). Full per-institution attribution lives in the project spreadsheet linked from the [open_x_embodiment README](https://github.com/google-deepmind/open_x_embodiment).

### Stated reasons for choosing RLDS
- "Standard and lossless format" with "explicit … contents and meaning of each field" [Ramos et al. 2021 abstract; [Google Research blog](https://research.google/blog/rlds-an-ecosystem-to-generate-share-and-use-datasets-in-reinforcement-learning/)].
- Integration with TFDS "facilitates the sharing of RL datasets with the research community" [Ramos et al. 2021].
- "Accommodates the various action spaces and input modalities of different robot setups" [arxiv.org/html/2310.08864v3].
- Supports "efficient, parallelized data loading in all major deep learning frameworks" [arxiv.org/html/2310.08864v3].

### Objective technical properties
- TFRecord serialization (sequential record container).
- Built on `tf.data.Dataset` / TFDS catalog.
- Preserves temporal ordering rather than randomizing interactions.
- All steps in a dataset must share identical fields (rigid schema).
- Cross-embodiment unification is "coarse" per the paper — actions are normalized per-dataset; no shared coordinate frame across robots.

---

## 2. HDF5-based formats (ALOHA, RoboMimic, MimicGen)

### What HDF5 is
Hierarchical, self-describing binary format from the HDF Group: groups, n-D array datasets, arbitrary attribute metadata, "no limit on the number or size of data objects" [[hdfgroup.org/solutions/hdf5](https://www.hdfgroup.org/solutions/hdf5/)]. In the robot datasets surveyed below, an episode is represented as one HDF5 group containing per-timestep datasets and attributes for env metadata.

### ALOHA / Mobile ALOHA
Schema per [Trossen ALOHA docs](https://docs.trossenrobotics.com/aloha_docs/2.0/operation/data_collection.html):

```
Stationary ALOHA                     Mobile ALOHA
/images                              /images
  /cam_high       (480,640,3) u8       /cam_high       (480,640,3) u8
  /cam_low        (480,640,3) u8       /cam_left_wrist (480,640,3) u8
  /cam_left_wrist (480,640,3) u8       /cam_right_wrist(480,640,3) u8
  /cam_right_wrist(480,640,3) u8     /qpos        (14,) f64
/qpos        (14,) f64                /qvel        (14,) f64
/qvel        (14,) f64                /action      (14,) f64
/action      (14,) f64                /base_action (2,)  f64
```

- Files: `episode_<idx>.hdf5`.
- The 14-D `qpos`/`qvel`/`action` vectors correspond to the dual-arm setup (two 7-D arms, including grippers) `[UNVERIFIED — exact 14-DoF breakdown not quoted from a primary source here]`.
- 50 Hz recording confirmed by `DT = 0.02` in [tonyzhaozh/act constants.py](https://github.com/tonyzhaozh/act); ACT policy "directly predicts joint positions at 50 Hz" [[ALOHA project page](https://tonyzhaozh.github.io/aloha/)].
- Mobile ALOHA adds 2-D `base_action` (mobile base velocity).

### RoboMimic (Mandlekar et al. 2021)
Schema per [robomimic dataset overview](https://robomimic.github.io/docs/datasets/overview.html):

```
data/                               (group)
  attrs: total                      (int)
  attrs: env_args                   (JSON: env_name, env_type, env_kwargs)
  demo_0/
    attrs: num_samples
    attrs: model_file               (MJCF XML, robosuite only)
    states     (N, D)               MuJoCo states
    actions    (N, A)               normalized to [-1, 1]
    rewards    (N,)
    dones      (N,)
    obs/                            per-key datasets; images uint8 HWC
    next_obs/
mask/                               optional filter keys (train/valid)
```

Filter keys "enable arbitrary splitting of a dataset into sub-groups." Common image obs: `agentview_image`, `robot0_eye_in_hand_image`.

### Other HDF5 adopters
- **MimicGen** (Mandlekar et al. 2023): same schema as RoboMimic; runnable with `robomimic train.py`. 48,000+ demos / 12 tasks [[NVlabs/mimicgen](https://github.com/NVlabs/mimicgen)].
- **Diffusion Policy** (Chi et al. 2023): native format is **Zarr** (`ReplayBuffer`), but loads RoboMimic HDF5 [[real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)].
- **BC-Z** (Jang et al. 2021): distributed via TFDS, so backed by **TFRecord, not HDF5** [[tensorflow.org/datasets/catalog/bc_z](https://www.tensorflow.org/datasets/catalog/bc_z)].

### Stated reasons
ALOHA and RoboMimic project docs do not explicitly state a rationale for HDF5 in surfaced documentation. The HDF Group cites self-describing files, hierarchical organization, integrated metadata, portability, and "high-performance I/O" as general HDF5 properties.

### Objective technical properties
- Single self-describing file holds heterogeneous arrays + attribute metadata.
- Cross-language bindings (h5py, HDF5 C/C++/Fortran/Java).
- No native video codec: ALOHA stores RGB as raw `uint8` `(T,480,640,3)` arrays; RoboMimic stores images as raw `uint8` HWC.
- Chunking, compression, partial reads, single-writer concurrency are standard HDF5 capabilities `[UNVERIFIED on the specific docs page fetched]`.

---

## 3. LeRobot (Parquet + MP4, Hugging Face)

### Layout (v3.0 — current `CODEBASE_VERSION`)
Per [src/lerobot/datasets/lerobot_dataset.py](https://github.com/huggingface/lerobot) docstring and [docs/source/lerobot-dataset-v3.mdx](https://github.com/huggingface/lerobot):

```
data/chunk-XXX/file-YYY.parquet            tabular frames; many episodes per file
videos/<obs.image.cam>/chunk-XXX/file-YYY.mp4
meta/info.json                             schema, fps, codebase_version, path templates
meta/stats.json                            global mean/std/min/max
meta/tasks.parquet                         task strings <-> task_index
meta/episodes/chunk-XXX/file-YYY.parquet   per-episode lengths / tasks / offsets
```

### Schema fields
From [src/lerobot/utils/constants.py](https://github.com/huggingface/lerobot):
- `timestamp` (f32), `frame_index` (i64), `episode_index` (i64), `task_index` (i64), `index`.
- Optional RL keys: `next.reward`, `next.done`, `next.truncated`.
- User-defined: `observation.state`, `observation.images.<camera>`, `action`.

Verified live on `huggingface.co/datasets/lerobot/pusht`: `action, episode_index, frame_index, next.done, next.reward, next.success, observation.image, observation.state, task_index, timestamp`.

### Versions
v1 → v2 (per-episode Parquet+MP4) → v2.1 (per-episode stats) → **v3.0** (file-based chunks, Parquet metadata, streaming). Converter: `src/lerobot/scripts/convert_dataset_v21_to_v30.py`.

### Stated reasons (LeRobot docs)
From [docs/source/lerobot-dataset-v3.mdx](https://github.com/huggingface/lerobot):
- Tabular: "Low-dimensional, high-frequency signals … stored in Apache Parquet. Access is memory-mapped or streamed via the `datasets` stack."
- Visual: "Camera frames concatenated and encoded into MP4. Frames from the same episode are grouped; videos are sharded per camera for practical sizes."
- Scaling: "To scale to millions of episodes, tabular rows and video frames from multiple episodes are concatenated into larger files. Episode-specific views are reconstructed via metadata, not file boundaries."
- Default video codec: `libsvtav1` (AV1); also h264/hevc and hardware encoders.

### Adopters
- Hugging Face LeRobot library + `huggingface.co/lerobot` org datasets (pusht, aloha_*, etc.).
- DROID 1.0.1 (76k+ Franka trajectories) — ported via `examples/port_datasets/port_droid.py`.
- yaak-ai/L2D-v3.
- Community datasets tagged `LeRobot` on HF Hub (HF blog ["LeRobot Community Datasets: The 'ImageNet' of Robotics"](https://huggingface.co/blog/lerobot-datasets) highlights SO-100 and Koch arms among community contributions).
- **Physical Intelligence (Pi)**: openpi uses the LeRobot dataset format as its canonical input pipeline (LIBERO converter shipped) [[github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)]. Whether Pi's own internal data is natively LeRobot vs. converted is `[UNVERIFIED]`.
- **NVIDIA GR00T-N1** (2025) — released GR00T-N1 dataset on HF in LeRobot schema [[arXiv:2503.14734](https://arxiv.org/abs/2503.14734); HF `nvidia/GR00T-N1-2B`].
- Tesla, Figure: `[UNVERIFIED]` — no citation found.

### Conversions in the repo
- v2.1 → v3.0: `convert_dataset_v21_to_v30.py`.
- RLDS → LeRobot: `examples/port_datasets/port_droid.py` with SLURM helpers.
- HDF5 / ROS bag → LeRobot: `[UNVERIFIED]` no first-party converter; community ports exist on Hub.

### Objective technical properties
- MP4 decode required at read time (default `torchcodec`, fallback `pyav` / torchvision).
- Parquet is columnar → projection (read only `action` etc.) and PyArrow memory-mapping.
- HF Hub streaming via `StreamingLeRobotDataset` — no full download required.
- v3 consolidates many episodes per file → fewer filesystem entries; trade-off is that episode boundaries must be resolved via `meta/episodes/` rather than filenames.
- Video stored via codec (default AV1 / `libsvtav1`) rather than raw frames `[UNVERIFIED — no published size benchmark cited here]`.

---

## 4. Commercial humanoid companies

Of the companies surveyed below, only **Physical Intelligence** (LeRobot) and **1X** (custom binary shards) publish runnable dataset specs. The rest are `[UNDISCLOSED]` at the schema level in the sources reviewed — only architecture-level statements and aggregate hours/modalities are public.

### Tesla (Optimus)
- Format: `[UNDISCLOSED]`. No published schema, paper, or repo.
- Modalities: originally mocap suit + VR headset joint trajectories paired with onboard vision; **shifted in mid-2025 to a five-camera helmet+backpack rig recording egocentric video only** [[Yahoo / Business Insider](https://finance.yahoo.com/news/tesla-shifted-optimus-training-strategy-092502233.html); [eWeek](https://www.eweek.com/news/tesla-optimus-robot-training/)].
- Collection: human teleoperators in mocap suits + VR (50+ hires, 1.70–1.80 m height requirement) [[IoT World Today](https://www.iotworldtoday.com/robotics/tesla-pays-48-an-hour-for-motion-capture-to-develop-humanoid-robot)]; vision-only egocentric since June 2025.
- Stated reason for the shift: "scale data collection more quickly" by removing the suit.
- Open releases: none.

### Figure AI
- Format: `[UNDISCLOSED]`. Only architecture-level disclosure: Helix paired-image + state + action stream [[Figure Helix post](https://www.figure.ai/news/helix)].
- Modalities (confirmed): monocular onboard images; state = wrist pose + finger positions; action = **35-DoF at 200 Hz** (wrist poses, finger flex/abduction, torso, head); plus a synthetic "percentage task completion" channel.
- Collection: ~500 hours teleoperated, multi-robot, multi-operator. Project Go-Big adds passively-collected egocentric human video from Brookfield homes [[Project Go-Big](https://www.figure.ai/news/project-go-big)].
- Stated reason: 500 h is "<5% the size of previously collected VLA datasets" [[Helix post](https://www.figure.ai/news/helix)]. Helix also describes VLM-based hindsight auto-labeling ("What instruction would you have given the robot…").
- Open releases: none.

### 1X Technologies
- Format (**publicly documented**): custom binary shards `video_{shard}.bin`, `states_{shard}.bin`, `segment_idx_{shard}.bin`, `metadata.json` [[github.com/1x-technologies/1xgpt](https://github.com/1x-technologies/1xgpt); [HF dataset card](https://huggingface.co/datasets/1x-technologies/worldmodel)]. Not LeRobot, not Parquet, not HDF5.
- Modalities: egocentric video from EVE. v2.0 = NVIDIA Cosmos discrete DV8×8×8 video tokens at 30 Hz; v1.1 = 16×16 MAGVIT2 patches at 2 Hz over 256×256. State/action vector = 25 f32 dims (21 joint angles, 2 hand-closure scalars, linear vel, angular vel).
- Collection: teleoperation on EVE robots in homes/offices. Internal corpus: "thousands of hours"; public release: >100 hours (27.3 GB).
- Stated reason: tokenized representation chosen because "raw logit tensors would be prohibitively large"; world-model framing "directly from raw sensor data" avoids manual asset creation [[1X World Model post](https://www.1x.tech/discover/1x-world-model)].
- Open releases: yes — worldmodel dataset on HF + 1xgpt training code + tokenizer checkpoints.

### Physical Intelligence (Pi)
- Format (**publicly documented**): openpi's input pipeline accepts the **LeRobot dataset format** and ships a LIBERO converter [[github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)]. Whether Pi's own internal training corpus is natively in LeRobot format vs. converted is `[UNVERIFIED]`.
- Modalities: multiple cameras (exterior + wrist) + proprioceptive state + language → action chunks. Eight embodiments: UR5e, bimanual UR5e, Franka, bimanual Trossen, bimanual ARX, Mobile Trossen, Mobile Fibocom [[π0 blog](https://www.pi.website/blog/pi0)].
- Collection: teleoperation across Pi's own fleet + Open X-Embodiment + internet pretraining. Exact hours `[UNDISCLOSED]`.
- Stated reason: cross-embodiment training breadth; LeRobot adoption gives interop with HF tooling.
- Open releases: openpi weights (π0, π0-FAST, π0.5 for DROID, LIBERO, ALOHA towel-folding/tupperware). No full proprietary dataset released.

### Boston Dynamics (Atlas, Spot)
- Format: `[UNDISCLOSED]`.
- Modalities: March 2025 Atlas RL demo trained on **mocap-suit trajectories** [[BD on X, 2025-03-19](https://x.com/BostonDynamics/status/1902359785678500355)]. TRI Large Behavior Model collaboration uses Atlas teleop demos + Open X-Embodiment [[RAI press](https://rai-inst.com/resources/press-release/boston-dynamics-atlas-partnership/); [TechTalks](https://bdtechtalks.com/2025/08/25/boston-dynamics-large-behavior-models/)].
- Stated reason: build "a shared reinforcement learning training pipeline" for electric Atlas via the RAI partnership.
- Open releases: none for Atlas.

### Agility Robotics (Digit)
- Format: `[UNDISCLOSED]`.
- Modalities: whole-body-control foundation model trained on **synthetic trajectories generated in NVIDIA Isaac Sim** by uniform random sampling of workspace pose targets [[Agility post](https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model)].
- Collection: teleop demos + sim + RL refinement (qualitative); whole-body model is pure sim.
- Stated reason: human-mocap datasets like AMASS / LAFAN1 "don't cover the entire desired manipulation workspace."
- Open releases: none.

### Sanctuary AI (Phoenix)
- Format: `[UNDISCLOSED]`.
- Modalities: depth + RGB cameras (Gen 8 increased FoV/resolution); tactile = 7-cell micro-barometer pads at finger tips, 5 mN sensitivity [[Sanctuary press](https://www.businesswire.com/news/home/20250226807548/en/Sanctuary-AI-Equips-General-Purpose-Robots-with-New-Touch-Sensors-for-Performing-Highly-Dexterous-Tasks); [Robot Report](https://www.therobotreport.com/sanctuary-ai-integrates-tactile-sensors-into-phoenix-general-purpose-robots/)].
- Collection: teleoperated "pilots" generate demonstrations consumed by Sanctuary's "Carbon" control system.
- Stated reason: "highest quality, highest fidelity training data" via human pilots; tactile added for richer training signal.
- Open releases: none.

### Apptronik (Apollo)
- Format / modalities: `[UNDISCLOSED]`.
- Collection: qualitative "data-collection engine" / "flywheel" [[Automate.org](https://www.automate.org/robotics/industry-insights/this-years-model-apptroniks-next-apollo-is-nearly-ready-for-its-closeup)]. Public partnership with Google DeepMind on Gemini Robotics, but no spec disclosed.
- Open releases: none.

### Skild AI
- Format: `[UNDISCLOSED]`.
- Modalities: "human-operated robot data" + "millions of public videos" + simulation rollouts [[Skild blog](https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain); [NVIDIA case study](https://www.nvidia.com/en-gb/case-studies/skild-ai/)].
- Stated reason: Skild claims "1,000× more data points than competing models" via being morphology-agnostic ("omni-bodied") so quadruped/humanoid/tabletop/mobile + human video can be pooled.
- Open releases: none.

### Covariant
- Format: `[UNDISCLOSED]` at file level. Architecturally: all modalities tokenized into a shared space for autoregressive next-token prediction by RFM-1 (8B params) [[Covariant RFM-1 post](https://covariant.ai/insights/introducing-rfm-1-giving-robots-human-like-reasoning-capabilities/)].
- Modalities (confirmed): "still images, video from various angles, station and task descriptions, sensor data from motor encoders and pressure sensors" + suction-cup strength + joint angles + force readings. World-model output runs at ~512×512 / ~5 fps.
- Collection: telemetry from Covariant's deployed warehouse pick-and-place fleet ("tens of millions of trajectories") + internet pretraining. Covariant's foundation-model team was reported as largely acquihired by Amazon in August 2024 `[UNVERIFIED — no citation in the source briefing]`.
- Open releases: none.

### Disclosure summary

| Company | File format public? | Open dataset? | Open weights? |
|---|---|---|---|
| Tesla Optimus | No | No | No |
| Figure AI | No | No | No |
| 1X | **Yes** (custom .bin shards) | **Yes** (>100 h on HF) | Partial (1xgpt) |
| Physical Intelligence | **Yes** (LeRobot input pipeline; internal data format `[UNVERIFIED]`) | Partial (LIBERO/DROID converters) | **Yes** (openpi) |
| Boston Dynamics | No | No | No |
| Agility | No | No | No |
| Sanctuary AI | No | No | No |
| Apptronik | No | No | No |
| Skild AI | No | No | No |
| Covariant | No | No | No |

---

## 5. Simulation / synthetic data formats

### NVIDIA Isaac Lab / Isaac Sim / GR00T
- Scene format: **OpenUSD** [[Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/)].
- Demonstration recording (`record_demos.py`) writes trajectories to **HDF5 in robomimic format** [Isaac Lab "Imitation Learning" tutorial].
- **GR00T-N1** (Bjorck et al. 2025, [arXiv:2503.14734](https://arxiv.org/abs/2503.14734)) describes a "data pyramid" (web video + sim + real teleop). Released GR00T-N1 dataset is in **LeRobot schema** on HF (`nvidia/GR00T-N1-2B`).

### MuJoCo MJCF
- MJCF is XML scene/model format for MuJoCo (bodies, joints, actuators, contacts) [MuJoCo docs].
- **No built-in trajectory container**; trajectories are logged externally.
- MuJoCo MPC logs rollouts via protobuf/JSON [google-deepmind/mujoco_mpc README].
- robosuite / robomimic (which wrap MuJoCo) standardize demonstrations as **HDF5**.

### MimicGen (Mandlekar et al., CoRL 2023)
- Takes a small set of human demos in robomimic HDF5; segments by object-centric subtasks; replays transformed end-effector trajectories [[arXiv:2310.17596](https://arxiv.org/abs/2310.17596)].
- Output stored in the **same robomimic HDF5 schema** as input. Released datasets (core, object, robot, large_interpolation) on HF (`amandlek/mimicgen_datasets`).

### RoboCasa (Nasiriany et al., RSS 2024)
- Scenes built in **MuJoCo MJCF** with kitchen environments composed of procedurally arranged AI-generated assets [[arXiv:2406.02523](https://arxiv.org/abs/2406.02523)].
- Demonstrations (human + MimicGen-augmented) stored in **robomimic HDF5**.
- 100K trajectories across 100 tasks.

### Other procedural / play data
- **RoboTurk** (Mandlekar et al. 2018): mobile-phone teleop; **HDF5 + MP4** per demo.
- **robosuite**: MJCF scenes; bundled `collect_human_demonstrations.py` outputs **HDF5**.
- **AI Habitat**: **GLB/PLY** meshes + scene config in JSON (`.scene_instance.json`); episodes as **JSON.gz**.

### Web / internet video as training data
- **Ego4D** (Grauman et al., CVPR 2022): **MP4** + **JSON** annotations (narrations, action labels, hand/object boxes) [[arXiv:2110.07058](https://arxiv.org/abs/2110.07058); ego4d-data.org].
- **Something-Something V2** (Goyal et al. 2017): **WebM** + **JSON** [[arXiv:1706.04261](https://arxiv.org/abs/1706.04261)].
- **R3M** (Nair et al. 2022) pretrains on Ego4D MP4s with time-contrastive + language losses [[arXiv:2203.12601](https://arxiv.org/abs/2203.12601)].
- **VC-1** (Majumdar et al., NeurIPS 2023) trains on CortexBench (Ego4D, ImageNet, …) [[arXiv:2303.18240](https://arxiv.org/abs/2303.18240)].
- **Voltron** (Karamcheti et al., ICML 2023) pretrains on Something-Something-v2 + language [[arXiv:2302.12766](https://arxiv.org/abs/2302.12766)].

### Stated reasons for each format
- **OpenUSD (Isaac)**: NVIDIA cites composition, layering, references, DCC interop [[Omniverse "Why USD"](https://openusd.org); openusd.org overview].
- **MJCF**: chosen for fast contact-rich simulation; designed for MuJoCo's compute model [MuJoCo docs].
- **HDF5 (robomimic / MimicGen / RoboCasa)**: hierarchical grouping, partial loading, compression for large demo sets [robomimic docs].
- **LeRobot Parquet+MP4 (GR00T)**: columnar Parquet for tabular data + MP4 for video efficiency [LeRobot docs].

### Objective technical properties
- USD supports composition, references, variants, layer overrides — designed for scene reuse.
- Isaac Lab writes trajectories to HDF5 (not USD) [Isaac Lab "Imitation Learning" tutorial]. Whether this is because USD is suboptimal for time-series numeric data is `[UNVERIFIED — not stated explicitly in the surfaced docs]`.
- MJCF has no native trajectory container; most surveyed MuJoCo-based imitation-learning projects (robosuite, robomimic, MimicGen, RoboCasa) layer HDF5 on top, while MuJoCo MPC logs rollouts via protobuf/JSON.
- HDF5 supports chunking, compression (gzip/lzf), partial reads; lacks columnar query.
- Parquet is columnar with predicate pushdown; LeRobot stores video in MP4 sidecars rather than as Parquet columns.
- MP4 + JSON (Ego4D, SSv2) decouples pixel data from labels, enabling re-annotation without re-encoding.

---

## 6. Cross-format comparison

| Format | Container | Video storage | Tabular storage | Cross-language | Streaming | Used by |
|---|---|---|---|---|---|---|
| RLDS | TFRecord | inside record (encoded bytes) | inside record | Python (TF) | tf.data | Open X-Embodiment, RT-1/2/X, BC-Z |
| HDF5 (robomimic-style) | single .hdf5 | raw uint8 arrays | datasets in groups | C/C++/Java/Python/Fortran | h5py partial reads | ALOHA, RoboMimic, MimicGen, RoboCasa, Isaac Lab `record_demos.py` |
| LeRobot v3 | dir of Parquet+MP4+JSON | MP4 (AV1/H264/HEVC) | Parquet (columnar) | Python (Arrow ecosystem) | HF Hub `StreamingLeRobotDataset` | LeRobot, openpi (input pipeline; Pi internal data `[UNVERIFIED]`), GR00T-N1, DROID port |
| 1X custom | `.bin` shards | tokenized (DV8×8×8 or MAGVIT2) | `.bin` state shards | Python | sequential | 1X worldmodel |
| OpenUSD | `.usd*` | (scene only) | (scene only) | C++/Python | USD layers | Isaac Sim/Lab scene description |
| MJCF | XML | (model only) | (model only) | C/Python | n/a | MuJoCo / robosuite scenes |

---

## What this analysis does not cover

- Quantitative benchmarks (read throughput, on-disk size per hour) for any of these formats — the research did not surface published numbers.
- Internal formats of companies marked `[UNDISCLOSED]`.
- ROS bag (`.bag`) and the newer MCAP format, which are common for raw robot logs but were not the focus of the source briefings.
- DeepMind's RT-2 / Gemini Robotics internal data pipeline beyond what the Open X-Embodiment paper covers.

## Sources

All inline citations link to primary sources (papers, official repos, official docs, company blog posts). Items the agents could not verify against a primary source are tagged `[UNVERIFIED]`; items where the company has explicitly not disclosed information are tagged `[UNDISCLOSED]`.
