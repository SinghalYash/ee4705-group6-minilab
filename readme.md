# EE4705 MiniLab 1.3 – Group 6: LLM + YOLO Powered Quadruped

A simulated quadruped robot that accepts typed English commands, converts them into structured robot actions using an LLM, executes motion autonomously, and uses an onboard RGB camera with YOLO-based perception for object search and approach.

```text
Instruction ──► Task 3: LLM Parser ──► Executor ──► Task 2 / Task 4 ──► Feedback
  (English)        structured JSON       queue       motion / YOLO       terminal
                                                       control
```

**Scene:** a MuJoCo quadruped with three coloured chairs (green, red and blue) and a red stop sign, defined in `task_2/scene/task2_scene.xml`. The robot's `dog_front_camera` is the image source used for Task 4 perception.

**Group report:** [EE4705 MiniLab 1.3 Report](https://docs.google.com/document/d/1iv5LpOmHX5K73lbYQxbs57AS8vZbt1TNYQ6d3tNpbnE/edit?usp=sharing)

---

## Contents

1. [Project overview](#1-project-overview)
2. [Setup](#2-setup)
3. [Running the integrated system](#3-running-the-integrated-system)
4. [Task 2 – Platform, camera, scene and motion skills](#4-task-2--platform-camera-scene-and-motion-skills)
5. [Task 3 – Typed English commands via an LLM parser](#5-task-3--typed-english-commands-via-an-llm-parser)
6. [Task 4 – YOLO object search and approach](#6-task-4--yolo-object-search-and-approach)
7. [Evaluation results](#7-evaluation-results)
8. [Scene and configuration files](#8-scene-and-configuration-files)
9. [Results and evidence files](#9-results-and-evidence-files)
10. [Troubleshooting](#10-troubleshooting)
11. [Contributions](#11-contributions)
12. [References](#12-references)

---

## 1. Project overview

The project implements the complete MiniLab pipeline:

- a MuJoCo quadruped driven by a pre-trained reinforcement-learning locomotion policy;
- an onboard `320 × 240` RGB front camera;
- reusable timed-motion and closed-loop-turn skills;
- an LLM parser that maps free-form English instructions to a validated JSON command schema;
- autonomous execution of single-step and multi-step commands;
- YOLO object detection with colour grounding;
- closed-loop search and approach for requested objects.

Tasks 3 and 4 use the same platform, scene, onboard camera and motion-skills API developed in Task 2.

### 1.1 Two-rate architecture

The LLM is deliberately kept outside the fast locomotion loop:

| Component | Rate |
|---|---:|
| MuJoCo physics + PD torque control | 200 Hz |
| ONNX locomotion policy | 50 Hz |
| Onboard front-camera perception | 15 Hz |
| LLM command parser | Once per user command |

```text
Slow loop:
English instruction → LLM → validated structured actions

Fast local loop:
camera / perception → motion skills → RL policy → PD control → MuJoCo
```

The walking policy receives a 46-dimensional observation. Six consecutive observations are retained, giving the `276`-dimensional ONNX policy input. The policy outputs 12 actions that are converted into target joint positions and tracked by the PD controller.

The ONNX policy uses IsaacGym joint ordering (`FL, FR, RL, RR`), while the MuJoCo model uses (`FL, FR, RR, RL`), so the rear-leg groups are remapped before and after policy inference.

---

## 2. Setup

### 2.1 Environment

The final development environment used:

| Item | Configuration |
|---|---|
| Simulator | MuJoCo |
| Python | 3.12.14 |
| Front camera | `dog_front_camera` |
| Camera resolution | `320 × 240` |
| Perception rate | 15 Hz |

### 2.2 Main dependencies

| Package | Used for |
|---|---|
| `mujoco` | Physics simulation and rendering |
| `numpy` | Numerical operations |
| `onnxruntime` | RL locomotion policy inference |
| `pyyaml` | YAML configuration |
| `Pillow` | Image handling |
| `ultralytics` | YOLO object detection |
| `opencv-python` | Image processing and evidence generation |
| `openai` | Cloud LLM interface |
| `ollama` | Local LLM evaluation baseline |
| `SpeechRecognition` | Optional speech-related support |

### 2.3 Install

From the repository root:

```bash
conda activate ee4705
python -m pip install -e quadruped_mujoco
pip install mujoco numpy onnxruntime pyyaml pillow ultralytics opencv-python openai ollama SpeechRecognition
```

The `quadruped_mujoco` directory contains the locally installed `runtime_control` package used by the simulator and browser interface.

### 2.4 LLM configuration

API keys are supplied through environment variables and **must never be committed to the repository**.

#### OpenAI

The Task 3 cloud parser supports `gpt-5-mini`.

**Windows CMD**
```cmd
set LLM_PROVIDER=openai
set OPENAI_API_KEY=<your-api-key>
```

**macOS / Linux**
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="<your-api-key>"
```

#### Alibaba Cloud DashScope

The parser also supports the Alibaba Cloud OpenAI-compatible endpoint with `qwen3-vl-flash`.

**Windows CMD**
```cmd
set LLM_PROVIDER=qwen
set DASHSCOPE_API_KEY=<your-api-key>
```

**macOS / Linux**
```bash
export LLM_PROVIDER=qwen
export DASHSCOPE_API_KEY="<your-api-key>"
```

#### Local Ollama baseline

`task_3/ollama_parser.py` provides a local evaluation parser using:

```text
qwen3:4b
```

No cloud API key is required for the Ollama baseline.

---

## 3. Running the integrated system

The integrated Task 2 + Task 3 + Task 4 system is launched through `task_4/run_integrated.py`.

### 3.1 Native MuJoCo viewer

```bash
python task_4/run_integrated.py --executor task3
```

### 3.2 Browser control panel

```bash
python task_4/run_integrated.py --executor task3 --gui
```

The integrated execution path is:

```text
Typed English command
        │
        ▼
Task 3 LLM parser
        │
        ▼
Validated JSON actions
        │
        ▼
Task 3 executor
        │
        ├──────── move / turn ───────► Task 2 MotionSkills
        │
        └──────── goto_object ───────► Task 4 search + approach
                                              │
                                              ▼
                                      Task 2 onboard camera
                                              │
                                              ▼
                                      YOLO + colour grounding
                                              │
                                              ▼
                                      closed-loop velocity
```

Example motion command:

```text
walk forward for three seconds, then turn back
```

Example object-search command:

```text
go to the green chair
```

### 3.3 Required terminal logs

The integrated system prints the main assessment logs:

| Tag | Purpose |
|---|---|
| `[CMD]` | Parsed/validated Task 3 command |
| `[EXEC]` | Action currently being executed |
| `[DONE]` | Task 3 action sequence completed |
| `[TURN]` | Task 2 closed-loop turn result |
| `[SEARCH]` | Task 4 target search |
| `[DETECT]` | YOLO + colour-grounding detection |
| `[FOUND]` | Target confirmed within the required distance |
| `[MISSION]` | Final Task 4 mission status |

---

## 4. Task 2 – Platform, camera, scene and motion skills

**Owner:** Chan Ping-Shuen Savannah

### 4.1 Run Task 2 platform

Native MuJoCo viewer:

```bash
python quadruped_mujoco/eg/play.py
```

Browser control panel:

```bash
python quadruped_mujoco/eg/play.py --gui
```

### 4.2 Manual controls

| Key | Action |
|---|---|
| `W / S` | Forward / backward |
| `A / D` | Strafe |
| `Q / E` | Manual turn |
| `R / F` | Body height |
| `T` | Reset |
| `X` | Emergency stop |
| `C` | Capture onboard front-camera frame |

### 4.3 Motion-skill shortcuts

The native MuJoCo viewer and browser interface expose the same shortcuts for directly testing the reusable Task 2 motion skills:

| Key | Skill |
|---|---|
| `M` | Timed forward move |
| `J` | Closed-loop `+90°` turn |
| `L` | Closed-loop `-90°` turn |
| `B` | Closed-loop `+180°` turn |

The underlying API is implemented in `task_2/motion_skills.py`:

```python
move(vx, vy, wz, duration)
set_velocity(vx, vy, wz)
stop()
turn(angle_deg)
get_base_pose()
```

`move()` uses a queue of timed velocity commands. `set_velocity()` provides continuous commands for Task 4 visual servoing. `turn()` reads the robot's measured yaw from the MuJoCo simulation state rather than relying on open-loop timing.

A completed closed-loop turn prints a line such as:

```text
[TURN] target=90.0 deg actual=87.0 deg final_error=3.0 deg
```

### 4.4 Onboard camera pipeline

The front-camera pipeline is implemented in:

```text
task_2/camera_pipeline.py
```

Configuration:

| Parameter | Value |
|---|---|
| Camera | `dog_front_camera` |
| Resolution | `320 × 240` |
| Perception frequency | 15 Hz |

The public interface:

```python
get_latest_frame()
```

returns:

```text
(rgb, simulation_time)
```

where `rgb` is the latest `HxWx3` RGB image and the timestamp identifies newly rendered perception frames.

Task 4 uses this onboard front-camera stream as its image source.

### 4.5 Custom object scene

The custom scene is:

```text
task_2/scene/task2_scene.xml
```

It contains:

| Object | COCO class | Colour |
|---|---|---|
| Chair 1 | `chair` | Green |
| Chair 2 | `chair` | Red |
| Chair 3 | `chair` | Blue |
| Stop sign | `stop sign` | Red |

The multiple chairs provide same-class colour-disambiguation cases.

Ground-truth object locations are stored in:

```text
objects.yaml
```

These positions are used only for final-distance logging and evaluation. They are **not used by the controller for steering**.

### 4.6 YOLO scene verification

Run:

```bash
python task_2/test_yolo.py
```

The script runs YOLO on a rendered camera frame and saves annotated results.

Evidence is stored under:

```text
task_2/evidence/
task_2/yolo_test/
```

---

## 5. Task 3 – Typed English commands via an LLM parser

**Owner:** Muhammad Irfan Bin Abdullah

Task 3 converts free-form English instructions into a validated list of structured robot actions.

### 5.1 Command schema

The JSON schema is defined in:

```text
task_3/command_schema.py
```

Supported actions:

| Action | Main fields | Executed by |
|---|---|---|
| `move` | `vx`, `vy`, `wz`, `duration` | Task 2 timed-move skill |
| `turn` | `angle_deg` | Task 2 closed-loop turn |
| `goto_object` | `class`, `color` | Task 4 search and approach |
| `stop` | — | Task 3 executor |
| `chat` | `reply` | Chat interface |

Example input:

```text
walk forward for three seconds, then turn back
```

produces an ordered action sequence equivalent to:

```json
{
  "accepted": true,
  "actions": [
    {
      "action": "move",
      "vx": 0.8,
      "vy": 0.0,
      "wz": 0.0,
      "duration": 3.0
    },
    {
      "action": "turn",
      "angle_deg": 180.0
    }
  ]
}
```

### 5.2 Parser and chat interface

Main files:

| File | Purpose |
|---|---|
| `task_3/llm_parser.py` | Cloud LLM parser |
| `task_3/command_schema.py` | Structured command schema |
| `task_3/ollama_parser.py` | Local Ollama evaluation parser |
| `task_3/chat_interface.py` | Multi-turn terminal interface |
| `task_3/executor.py` | Sequential autonomous execution |
| `task_3/evaluation.py` | Parser evaluation |

The parser supports:

- single-step commands;
- multi-step commands;
- paraphrases;
- invalid/out-of-scope rejection;
- non-English rejection;
- clarification/chat responses.

The chat interface runs separately from the simulation loop so that waiting for the user or LLM does not block MuJoCo.

---

## 6. Task 4 – YOLO object search and approach

**Owner:** Singhal Yash

Task 4 connects `goto_object(class, color)` to the Task 2 onboard camera and motion interface.

### 6.1 Main files

| File | Purpose |
|---|---|
| `task_4/perception.py` | YOLO detection and colour grounding |
| `task_4/approach.py` | Closed-loop search/approach controller |
| `task_4/goto_object.py` | Task 4 mission entry point |
| `task_4/robot_adapter.py` | Adapter between Task 2 and Task 4 |
| `task_4/check_detection.py` | Detection/colour verification |
| `task_4/run_integrated.py` | Task 2 + 3 + 4 integration |
| `task_4/run_eval.py` | Task 4 evaluation |

### 6.2 Perception

YOLO detects the requested COCO class from the live Task 2 camera stream. Colour is classified separately from pixels inside the detected bounding box.

Example:

```text
[DETECT] class=chair color=green conf=0.72 bbox=[212,140,318,352]
```

### 6.3 Search and approach

When the requested object is not visible, the robot rotates and repeatedly processes new onboard-camera frames:

```text
[SEARCH] target not visible, rotating
```

Once a matching object is acquired, the controller:

1. compares the bounding-box centre with the image centre;
2. turns to reduce horizontal image error;
3. commands forward velocity;
4. estimates target range;
5. slows as it approaches;
6. stops and confirms the requested target.

A mission is successful only when the target is confirmed and satisfies the required proximity criterion.

Example:

```text
[FOUND] class=chair color=green t=14.2 s d=0.61 m
[MISSION] status=SUCCESS
```

The required planar trunk-to-object distance is:

```text
d <= 0.80 m
```

Ground-truth positions from `objects.yaml` are used only to calculate `d` for logging and evaluation.

---

## 7. Evaluation results

### 7.1 Task 3 parser evaluation

Run:

```bash
python task_3/evaluation.py openai
```

or:

```bash
python task_3/evaluation.py ollama
```

Results:

| Parser | Correct | Accuracy |
|---|---:|---:|
| OpenAI | 19/20 | 95% |
| Ollama | 19/20 | 95% |

The 20-command evaluation covers basic commands, multi-step commands, paraphrases and invalid/out-of-scope requests.

Detailed results:

```text
task_3/results/openai_results.csv
task_3/results/ollama_results.csv
```

The OpenAI run had one failure on an invalid request that combined unsupported behaviour with a movement command. The Ollama run had one failure in which a non-English command was interpreted as a valid movement command.

### 7.2 Task 4 search-and-approach evaluation

Run:

```bash
python task_4/run_eval.py
```

The final evaluation includes different targets and starting poses, same-class colour disambiguation, targets that are initially outside the camera view, longer-distance starts and an absent-target negative test.

| Metric | Near starts (2.1–4.6 m) | Far starts (5.1–5.6 m) | All positive trials |
|---|---:|---:|---:|
| Detection | 11/11 (100%) | 4/4 (100%) | **15/15 (100%)** |
| Correct grounding | 11/11 (100%) | 3/4 (75%) | **14/15 (93%)** |
| Approach success (`SUCCESS` and `d ≤ 0.80 m`) | 11/11 (100%) | 3/4 (75%) | **14/15 (93%)** |
| `SUCCESS` reported with `d > 0.80 m` | 0/11 | 0/4 | **0/15** |

Additional metrics:

| Metric | Result |
|---|---:|
| Correct rejection of absent target | 1/1 (100%) |
| Mean final distance of successful trials | 0.58 m |
| Successful final-distance range | 0.50–0.79 m |
| Mean time to found | 12.9 s |
| Time-to-found range | 4.8–29.7 s |

One positive trial failed: the blue-chair target from a 5.41 m start timed out before successful approach.

Detailed results:

```text
task_4/results/eval_summary.md
```

---

## 8. Scene and configuration files

| File | Purpose |
|---|---|
| `task_2/scene/task2_scene.xml` | Custom Task 2 object scene |
| `objects.yaml` | Ground-truth object classes, colours and positions |
| `quadruped_mujoco/eg/dog/xml/dog_terrain.xml` | Quadruped MuJoCo model |
| `quadruped_mujoco/eg/model_3400.onnx` | Pre-trained locomotion policy |
| `quadruped_mujoco/eg/play.py` | Main simulation and Task 2 integration |
| `interfaces.py` | Shared Task 2/3/4 robot interface |

---

## 9. Results, Evidence and Demonstration Videos

| Path | Contents |
|---|---|
| `task_2/evidence/Task 2_Demo.mp4` | Task 2 demonstration: scene, onboard camera, timed motion and closed-loop turning |
| `task_2/evidence/` | Task 2 detection and supporting evidence |
| `task_2/yolo_test/` | YOLO annotated test outputs |
| `task_3/results/Video_Task 3.mp4` | Task 3 demonstration: typed English commands, multi-step autonomous execution and command rejection |
| `task_3/results/openai_results.csv` | OpenAI parser evaluation |
| `task_3/results/ollama_results.csv` | Ollama/Qwen parser evaluation |
| `Video task4.mp4` | Task 4 demonstration: YOLO detection, colour grounding, autonomous search and object approach |
| `task_4/results/eval_summary.md` | Task 4 quantitative evaluation summary |
| `task_4/results/frames/` | Saved Task 4 evaluation frames |
| `task_5/evidence/Video_Bonus.mp4` | Bonus demonstration: speech input, VLM/VQA, YOLO–VLM comparison and multi-goal navigation |
| `task_5/evidence/bbox_comparison.png` | YOLO versus VLM bounding-box comparison |

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: task_2`, `task_3` or `task_4` | Run from the repository root and activate the correct environment |
| `ModuleNotFoundError: runtime_control` | Run `python -m pip install -e quadruped_mujoco` |
| `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` missing | Set the required environment variable in the current terminal |
| LLM call fails | Check the API key, provider selection, internet connection and quota |
| MuJoCo viewer does not open | Check that `mujoco` is installed and no stale simulation process remains |
| Browser GUI does not open | Check that the GUI port is available and restart the simulator |
| YOLO does not detect the object | Check the rendered onboard-camera view, object size/visibility and supplied mesh/texture |
| Task 4 reports timeout | Inspect `[SEARCH]` / `[DETECT]` logs and the corresponding evaluation frame |
| API key was accidentally committed | Revoke/replace the key immediately and remove it from repository history |

For headless Linux execution:

```bash
export MUJOCO_GL=egl
```

---

## 11. Contributions

| Task | Owner | Main files |
|---|---|---|
| **Task 1 – Research and architecture** | **All** | Group report |
| **Task 2 – Platform, scene, camera and motion skills** | **Chan Ping-Shuen Savannah** | `quadruped_mujoco/eg/play.py`, `task_2/camera_pipeline.py`, `task_2/motion_skills.py`, `task_2/scene/task2_scene.xml`, `objects.yaml` |
| **Task 3 – LLM command parser and execution** | **Muhammad Irfan Bin Abdullah** | `task_3/llm_parser.py`, `task_3/command_schema.py`, `task_3/ollama_parser.py`, `task_3/chat_interface.py`, `task_3/executor.py`, `task_3/evaluation.py` |
| **Task 4 – Perception, search and approach** | **Singhal Yash** | `task_4/perception.py`, `task_4/approach.py`, `task_4/goto_object.py`, `task_4/robot_adapter.py`, `task_4/run_integrated.py`, `task_4/run_eval.py` |
| **Task 5 – Integration, report and submission** | **All** | Final system, report and demonstration evidence |

The shared interoperability contract is defined in:

```text
interfaces.py
```


## 12. References

### Platform and software

- [MuJoCo](https://mujoco.org/)
- [Example `quadruped_mujoco` platform](https://github.com/aoqianz/quadruped_mujoco)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenAI API](https://platform.openai.com/docs)
- [Alibaba Cloud Model Studio](https://www.alibabacloud.com/help/en/model-studio/)
- [Ollama](https://ollama.com/)

### Research references

1. E. Todorov, T. Erez, and Y. Tassa, **“MuJoCo: A Physics Engine for Model-Based Control,”** IEEE/RSJ IROS, 2012.
2. N. Rudin, D. Hoeller, P. Reist, and M. Hutter, **“Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning,”** CoRL, 2021.
3. T.-Y. Lin et al., **“Microsoft COCO: Common Objects in Context,”** ECCV, 2014.
4. M. Ahn et al., **“Do As I Can, Not As I Say: Grounding Language in Robotic Affordances,”** CoRL, 2022.
5. J. Liang et al., **“Code as Policies: Language Model Programs for Embodied Control,”** IEEE ICRA, 2023.
6. A. Brohan et al., **“RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control,”** CoRL, 2023.


