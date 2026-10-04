EE4705 MiniLab 1.3 – Group 6: LLM + YOLO Powered Quadruped

A simulated quadruped robot that accepts typed English commands, converts them into structured robot actions using an LLM, executes motion autonomously, and uses an onboard camera with YOLO-based perception for object search and approach.

```text
                     ┌──────────────────────────────┐
                     │       User (English)          │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │      Task 3: LLM Parser      │
                     │   English → structured JSON  │
                     └──────────────┬───────────────┘
                                    │
                         move / turn / goto_object
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │     Task 2: Motion Skills    │
                     │  timed move / closed-loop    │
                     │        turn / velocity      │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │      MuJoCo Quadruped        │
                     │   RL policy + PD control     │
                     └──────────────┬───────────────┘
                                    │
                         onboard front camera
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │    Task 4: YOLO Perception   │
                     │ detection + colour grounding │
                     │    search + visual approach  │
                     └──────────────────────────────┘

Tasks 3 and 4 use the same platform, scene, onboard camera and motion-skills API developed in Task 2.
Contents
1. Project Overview
2. Group Work Allocation
3. Repository Layout
4. Platform and Control Pipeline
5. Environment and Dependencies
6. Installation
7. API Keys and LLM Configuration
8. Task 2 – Platform, Camera, Scene and Motion Skills
9. Task 3 – English Command Parsing
10. Task 3 – Parser Evaluation
11. Task 4 – YOLO Object Search and Approach
12. Task 4 – Evaluation
13. Integrated Task 3 + Task 4 System
14. Scene and Configuration Files
15. Key Files by Student
16. Results Summary
17. Troubleshooting
18. References
1. Project Overview
This project implements the EE4705 MiniLab 1.3 system:
- a MuJoCo quadruped with a pre-trained reinforcement-learning locomotion policy;
- a 320 × 240 onboard RGB front camera;
- reusable motion skills for timed movement and closed-loop turning;
- an LLM parser that converts English instructions into structured commands;
- YOLO-based object detection with colour grounding;
- autonomous search and approach for requested objects.
The project uses a two-rate architecture:
Slow loop:
English command → LLM → structured robot actions

Fast local loop:
camera / perception → motion skills → RL policy → PD control → MuJoCo

The LLM is therefore kept outside the fast locomotion loop. Motion execution, perception and robot control remain local.
2. Group Work Allocation
Task	Student	Main responsibility
Task 1	All	Language-model comparison, learned locomotion and two-rate architecture
Task 2	Chan Ping-Shuen Savannah	MuJoCo platform, onboard camera, custom scene and motion skills
Task 3	Muhammad Irfan Bin Abdullah	LLM command parsing, JSON schema, chat interface, execution and parser evaluation
Task 4	Singhal Yash	YOLO perception, colour grounding, object search/approach and integrated evaluation
Task 5	All	Final integration, report, reproducibility and submission


3. Repository Layout
ee4705-group6-minilab/
│
├── quadruped_mujoco/
│   ├── eg/
│   │   ├── play.py
│   │   ├── model_3400.onnx
│   │   └── dog/
│   │       └── ...
│   │
│   └── src/runtime_control/
│       ├── runtime.py
│       ├── panel.py
│       ├── integration.py
│       ├── map_manager.py
│       ├── control.py
│       └── maps/
│
├── task_2/
│   ├── camera_pipeline.py
│   ├── motion_skills.py
│   ├── test_yolo.py
│   ├── front_camera_test.png
│   ├── scene/
│   │   └── task2_scene.xml
│   └── evidence/
│
├── task_3/
│   ├── llm_parser.py
│   ├── command_schema.py
│   ├── ollama_parser.py
│   ├── chat_interface.py
│   ├── executor.py
│   └── results/
│
├── task_4/
│   ├── perception.py
│   ├── approach.py
│   ├── goto_object.py
│   ├── robot_adapter.py
│   ├── check_detection.py
│   ├── run_integrated.py
│   ├── run_eval.py
│   └── results/
│
├── interfaces.py
├── objects.yaml
└── readme.md

4. Platform and Control Pipeline
The project uses the quadruped_mujoco platform.
Main control rates
Component	Rate
MuJoCo physics + PD control	200 Hz
RL locomotion policy	50 Hz
Onboard front-camera perception	15 Hz
LLM parser	Once per user command


The RL policy receives a 46-dimensional observation containing:
- commanded velocity (vx, vy, wz);
- body angular velocity;
- projected gravity;
- 12 joint-position deviations;
- 12 joint velocities;
- previous 12-dimensional action;
- commanded body height.
Six consecutive observations are retained, producing the 276-dimensional ONNX input.
The policy outputs 12 actions corresponding to the quadruped's actuated joints. These actions are scaled and combined with the nominal configuration to produce target joint positions. A PD controller then generates the joint torques applied to MuJoCo.
The policy was trained in simulation and transferred to MuJoCo without retraining. The rear-left/rear-right groups require joint-order remapping between the IsaacGym policy convention and the MuJoCo model convention.
5. Environment and Dependencies
Python environment
The final development environment used:
Python 3.12.14
Conda environment: ee4705

Main dependencies
mujoco
numpy
onnxruntime
pyyaml
pillow
ultralytics
opencv-python
openai
ollama
SpeechRecognition

The runtime-control package is installed locally from:
quadruped_mujoco/

The main system runs the MuJoCo simulation, ONNX policy and YOLO locally. Only the LLM parser requires a cloud API when using OpenAI or Alibaba Cloud.
6. Installation
From the repository root:
conda activate ee4705
python -m pip install -e quadruped_mujoco

Install the remaining dependencies:
pip install mujoco numpy onnxruntime pyyaml pillow ultralytics opencv-python openai ollama SpeechRecognition

The YOLO model weights are loaded by Ultralytics when required.
7. API Keys and LLM Configuration
API keys must never be committed to the repository.
The Task 3 parser supports the following cloud configuration:
OpenAI
Model: gpt-5-mini
Environment variable: OPENAI_API_KEY

Windows:
set LLM_PROVIDER=openai
set OPENAI_API_KEY=<your-api-key>

Linux/macOS:
export LLM_PROVIDER=openai
export OPENAI_API_KEY="<your-api-key>"

Alibaba Cloud DashScope
Model: qwen3-vl-flash
Environment variable: DASHSCOPE_API_KEY

Windows:
set LLM_PROVIDER=qwen
set DASHSCOPE_API_KEY=<your-api-key>

Linux/macOS:
export LLM_PROVIDER=qwen
export DASHSCOPE_API_KEY="<your-api-key>"

Local Ollama
Ollama is also provided as a local evaluation baseline:
Model: qwen3:4b
Implementation: task_3/ollama_parser.py

8. Task 2 – Platform, Camera, Scene and Motion Skills
8.1 Native MuJoCo viewer
Run:
python quadruped_mujoco/eg/play.py

This opens the native MuJoCo viewer.
8.2 Browser control panel
Run:
python quadruped_mujoco/eg/play.py --gui

The browser control panel is available at the runtime-control web interface.
8.3 Manual controls
W/S   forward/back
A/D   strafe
Q/E   turn
R/F   body height
T     reset
X     emergency stop
C     capture onboard front-camera frame

8.4 Task 2 motion-skill shortcuts
Both the native viewer and browser interface support the Task 2 motion-skill shortcuts:
M     timed forward move
J     closed-loop +90° turn
L     closed-loop -90° turn
B     closed-loop +180° turn

These call the same underlying MotionSkills API used by Tasks 3 and 4.
8.5 Motion skills API
Implemented in:
task_2/motion_skills.py

Main methods:
move(vx, vy, wz, duration)set_velocity(vx, vy, wz)stop()turn(angle_deg)get_base_pose()


move() places timed velocity commands into a queue.
set_velocity() provides continuous velocity control for visual servoing.
turn(angle_deg) uses measured robot yaw from the MuJoCo simulation state rather than relying on open-loop timing.
8.6 Onboard camera
Implemented in:
task_2/camera_pipeline.py

Final configuration:
Camera: dog_front_camera
Resolution: 320 × 240
Frequency: 15 Hz

The latest camera frame is exposed through:
get_latest_frame()


which returns:
(rgb, simulation_time)

Task 4 uses this onboard front-camera stream as its image source.
8.7 Custom scene
The Task 2 object scene is:
task_2/scene/task2_scene.xml

Ground-truth object information is stored in:
objects.yaml

The current scene contains:
green chair
red chair
blue chair
red stop sign

The multiple chairs provide same-class colour disambiguation.
Ground-truth positions are used only for distance logging and evaluation. They are not used by the controller to steer toward objects.
8.8 YOLO verification
Run:
python task_2/test_yolo.py

The script tests rendered camera images with YOLO and saves annotated detection results under:
task_2/yolo_test/

Evidence images are also stored under:
task_2/evidence/

9. Task 3 – English Command Parsing
Task 3 converts a typed English instruction into a list of validated structured actions.
Command schema
Defined in:
task_3/command_schema.py

Supported actions:
move
turn
goto_object
stop
chat

Example multi-step command:
walk forward for three seconds, then turn back

becomes an ordered sequence equivalent to:
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

Parser
Main parser:
task_3/llm_parser.py

The parser is LLM-based rather than keyword-only.
It handles:
- single-step commands;
- multi-step commands;
- natural-language paraphrases;
- invalid or unsafe requests;
- non-English requests;
- ordinary conversational replies.
Accepted commands print:
[CMD] actions=...

Rejected commands print:
[CMD] rejected reason=...

Chat interface
Implemented in:
task_3/chat_interface.py

After an accepted command, the robot begins autonomous execution without an additional teleoperation trigger.
Execution is handled in:
task_3/executor.py

and uses the Task 2 MotionSkills API.
The chat loop runs in a separate thread so that the simulation does not block while waiting for the user or LLM.
10. Task 3 – Parser Evaluation
Run the OpenAI evaluation:
python task_3/evaluation.py openai

Run the Ollama evaluation:
python task_3/evaluation.py ollama

Results are stored in:
task_3/results/openai_results.csv
task_3/results/ollama_results.csv

The evaluation contains 20 utterances covering:
- basic commands;
- multi-step commands;
- paraphrases;
- invalid/out-of-scope requests.
Final recorded result
Parser	Correct	Accuracy
OpenAI	19/20	95%
Ollama	19/20	95%


The OpenAI evaluation had one failure on an invalid command that combined an unsupported request with a movement instruction.
The Ollama evaluation had one failure where a non-English command was interpreted as a valid movement request.
These failure cases are analysed in the final report.
11. Task 4 – YOLO Object Search and Approach
Task 4 connects the Task 3 goto_object command to the Task 2 onboard camera and motion skills.
Main files
task_4/perception.py
task_4/approach.py
task_4/goto_object.py
task_4/robot_adapter.py

Perception pipeline
YOLO detects the requested COCO class from the live Task 2 front-camera frame.
Colour is determined separately from image pixels inside the detected bounding box.
The system prints detections in the form:
[DETECT] class=chair color=green conf=0.72 bbox=[...]

Search and approach
When the target is not visible:
[SEARCH]

is printed and the robot rotates in place until a candidate target is detected.
After detection:
1. the target bounding-box centre is compared with the camera centre;
2. yaw is adjusted to reduce the horizontal error;
3. the robot walks forward;
4. the object range is estimated;
5. the robot stops when the target is sufficiently close;
6. the detection is confirmed before declaring success.
A target is considered found only when:
- the live onboard camera detects the requested class and colour;
- the planar trunk-to-object distance is at most 0.80 m;
- a [FOUND] line is printed.
Example:
[FOUND] class=chair color=green t=14.2 s d=0.61 m
[MISSION] status=SUCCESS

Ground-truth object positions from objects.yaml are used only for distance logging and evaluation.
12. Task 4 – Evaluation
Run the full evaluation:
python task_4/run_eval.py

The evaluation uses different target objects and robot starting poses, including:
- targets initially outside the camera view;
- same-class disambiguation such as green vs. red chairs;
- different starting positions;
- an absent-object negative test;
- longer-distance starting conditions.
The evaluation summary is stored in:
task_4/results/eval_summary.md

Final recorded results
Metric	Result
Positive trials	15
Detection accuracy	15/15 (100%)
Grounding accuracy	14/15 (93%)
Approach success	14/15 (93%)
Correct rejection of absent target	1/1 (100%)
Mean final distance of successful trials	0.58 m
Successful final-distance range	0.50–0.79 m
Mean time to found	12.9 s


One positive trial failed by timeout:
Target: blue chair
Start distance: approximately 5.41 m
Result: FAIL (timeout)
Final distance: approximately 5.38 m

13. Integrated Task 3 + Task 4 System
The integrated system connects the Task 3 command interface to the Task 4 object-search controller while reusing the Task 2 platform.
Native MuJoCo viewer
python task_4/run_integrated.py --executor task3

Browser control panel
python task_4/run_integrated.py --executor task3 --gui

The resulting pipeline is:
Typed English command
        ↓
Task 3 LLM parser
        ↓
Structured JSON command
        ↓
Task 3 executor
        ↓
Task 2 MotionSkills / Task 4 goto_object
        ↓
MuJoCo quadruped
        ↓
Task 2 onboard camera
        ↓
YOLO + colour grounding
        ↓
Closed-loop object search and approach

Task 3 terminal logs include:
[CMD]
[EXEC]
[DONE]

Task 2 turns print:
[TURN]

Task 4 prints:
[SEARCH]
[DETECT]
[FOUND]
[MISSION]

These terminal outputs are used as evidence in the demonstration videos.
14. Scene and Configuration Files
File	Purpose
task_2/scene/task2_scene.xml	Custom Task 2 object scene
objects.yaml	Ground-truth object classes, colours and positions
quadruped_mujoco/eg/dog/xml/dog_terrain.xml	Quadruped MuJoCo model
quadruped_mujoco/eg/model_3400.onnx	Pre-trained locomotion policy
quadruped_mujoco/eg/play.py	Main MuJoCo simulation and runtime integration
interfaces.py	Shared Task 2/3/4 robot interface


15. Key Files by Student
Student A — Chan Ping-Shuen Savannah
Task 2:
quadruped_mujoco/eg/play.py
task_2/camera_pipeline.py
task_2/motion_skills.py
task_2/scene/task2_scene.xml
objects.yaml

Responsibilities:
- MuJoCo platform integration;
- onboard front-camera pipeline;
- custom object scene;
- timed move API;
- continuous velocity API;
- closed-loop turn API.
Student B — Muhammad Irfan Bin Abdullah
Task 3:
task_3/llm_parser.py
task_3/command_schema.py
task_3/ollama_parser.py
task_3/chat_interface.py
task_3/executor.py
task_3/evaluation.py

Responsibilities:
- structured JSON command schema;
- LLM-based command parsing;
- multi-turn terminal interface;
- autonomous command execution;
- parser evaluation.
Student C — Singhal Yash
Task 4:
task_4/perception.py
task_4/approach.py
task_4/goto_object.py
task_4/robot_adapter.py
task_4/check_detection.py
task_4/run_integrated.py
task_4/run_eval.py

Responsibilities:
- YOLO object detection;
- colour grounding;
- object-search controller;
- visual approach;
- Task 3 integration;
- Task 4 evaluation.
Shared interface
interfaces.py

defines the interface used to connect the platform, motion skills, perception and higher-level execution.
16. Results Summary
Task 2
- MuJoCo native viewer verified.
- Browser GUI verified.
- Three or more terrain maps available through the runtime system.
- Three onboard camera perspectives available.
- dog_front_camera used for Task 4 perception.
- Camera stream configured at 320 × 240 and 15 Hz.
- Timed move implemented through a queue.
- Closed-loop turns implemented using measured simulation yaw.
- Motion-skill keyboard shortcuts tested in both native and browser interfaces.
Task 3
- Structured LLM output using a fixed JSON command schema.
- Single-step and multi-step English commands supported.
- Paraphrases supported.
- Invalid and out-of-scope commands rejected.
- 20-command evaluation completed with two LLM services.
- OpenAI accuracy: 95%.
- Ollama accuracy: 95%.
Task 4
- YOLO class detection implemented.
- Colour grounding implemented from bounding-box pixels.
- Closed-loop search and visual approach implemented.
- Same-class colour disambiguation supported.
- 15 positive trials evaluated.
- Detection: 100%.
- Grounding: 93%.
- Approach success: 93%.
- Absent-target rejection: 100%.
17. Troubleshooting
Problem	Fix
ModuleNotFoundError for task_2, task_3 or task_4	Run commands from the repository root and ensure the Conda environment is activated
DASHSCOPE_API_KEY or OPENAI_API_KEY missing	Set the required environment variable in the current terminal
LLM call fails	Check the API key, internet connection and available provider quota
MuJoCo viewer does not start	Check that the MuJoCo package is installed and the simulation is not already running
Browser GUI does not open	Ensure the requested GUI port is available
YOLO does not detect an object	Use the supplied scene meshes and test a sufficiently large rendered image
Task 4 fails to find an object	Check that the target is within the scene, the onboard front camera is active, and the requested colour/class is correct
Task 4 reports timeout	Inspect the [SEARCH], [DETECT] and [FOUND] logs and the evaluation frame saved under task_4/results/frames/
API key accidentally appears in code	Remove it immediately and replace the credential; keys must not be committed to Git


For headless Linux execution, MuJoCo can be configured with:
export MUJOCO_GL=egl

18. References
Platform and software
- MuJoCo: https://mujoco.org
- Example quadruped platform: https://github.com/aoqianz/quadruped_mujoco
- Ultralytics YOLO: https://docs.ultralytics.com
- ONNX Runtime: https://onnxruntime.ai
- OpenAI API documentation: https://platform.openai.com/docs
- Alibaba Cloud Model Studio: https://www.alibabacloud.com/help/en/model-studio/
- Ollama: https://ollama.com
Research references
- E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” IEEE/RSJ IROS, 2012.
- N. Rudin, D. Hoeller, P. Reist, and M. Hutter, “Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning,” CoRL, 2021.
- T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” ECCV, 2014.
- M. Ahn et al., “Do As I Can, Not As I Say: Grounding Language in Robotic Affordances,” CoRL, 2022.
- J. Liang et al., “Code as Policies: Language Model Programs for Embodied Control,” IEEE ICRA, 2023.
- A. Brohan et al., “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control,” CoRL, 2023.