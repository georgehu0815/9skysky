export type ExperimentId = "dance" | "swing" | "running" | "stilts";

export interface RewardTermDefinition {
  key: string;
  label: string;
  description: string;
  defaultWeight: number;
  penalty?: boolean;
  editable?: boolean;
}

export interface RewardDefinition {
  summary: string;
  formula: string;
  terms: RewardTermDefinition[];
}

export interface ExperimentDefinition {
  id: ExperimentId;
  title: string;
  shortTitle: string;
  goal: string;
  description: string;
  task: string;
  result: string;
  metricLabel: string;
  metricKey: string;
  metricSuffix: string;
  preview: string;
  poster: string;
  video: string;
  evidence: string[];
  artifactStem: string;
  defaultRunName: string;
  fullTimesteps: number;
  fullEnvs: number;
  maxEpisodeSeconds: number;
  ppo: {
    learningRate: number;
    gamma: number;
    clipCoefficient: number;
    updateEpochs: number;
    entropyCoefficient: number;
    maxGradNorm: number;
  };
  guideAnchor: string;
  guidance: {
    requiredInput: string;
    steps: string[];
    distinctions: string[];
    output: string;
  };
  reward: RewardDefinition;
  controls?: {
    stiltHeightCm?: number;
    stiltBlend?: number;
    stiltMassKg?: number;
  };
}

const REWARDS: Record<ExperimentId, RewardDefinition> = {
  dance: {
    summary:
      "PPO is paid for matching the authored pose and timing while remaining upright, travelling, and avoiding unsafe contacts or wasteful actuator use.",
    formula:
      "reward = pose tracking + rotation tracking + balance + travel - slip - spin - impacts - joint limits - motor effort",
    terms: [
      { key: "pose_match", label: "Pose match", defaultWeight: 4, description: "Match the clip's joint pose at the current frame." },
      { key: "rotation_match", label: "Rotation timing", defaultWeight: 4, description: "Match the clip's body rotation at the current time." },
      { key: "stick_it", label: "Stick the landing", defaultWeight: 5, description: "Finish a one-shot trick standing on both feet." },
      { key: "on_feet", label: "Stay on feet", defaultWeight: 5, description: "Remain upright, tall, and in foot contact during a looping dance." },
      { key: "travel", label: "Forward travel", defaultWeight: 3, description: "Move forward instead of performing the loop in place." },
      { key: "no_slip", label: "Foot slip", defaultWeight: 0.5, penalty: true, description: "Charge planted feet that skate across the floor." },
      { key: "no_spin", label: "Unwanted spin", defaultWeight: 0.3, penalty: true, description: "Charge yaw motion when the dance should travel straight." },
      { key: "gentle_head", label: "Head impacts", defaultWeight: 1, penalty: true, description: "Charge head-floor impacts by severity." },
      { key: "soft_landings", label: "Hard landings", defaultWeight: 0.75, penalty: true, description: "Charge excessive landing impact." },
      { key: "no_limit_parking", label: "Joint limits", defaultWeight: 1, penalty: true, description: "Charge joints held against their end stops." },
      { key: "save_energy", label: "Motor effort", defaultWeight: 0.5, penalty: true, description: "Charge excessive actuator torque." },
    ],
  },
  swing: {
    summary:
      "PPO is paid for expanding the bidirectional swing frontier and gaining controlled height and energy, while geometry and actuator penalties keep the motion planar and physically valid.",
    formula:
      "reward = frontier progress + height + late height + energy - lateral motion - misalignment - string violations - unsafe actions",
    terms: [
      { key: "swing_peak_progress", label: "New arc frontier", defaultWeight: 224, description: "Pay only for reaching a new peak on either side of the swing." },
      { key: "swing_height", label: "Swing height", defaultWeight: 8, description: "Reward height throughout the episode." },
      { key: "swing_late_height", label: "Late height", defaultWeight: 24, description: "Increase the value of height later in the attempt." },
      { key: "swing_energy", label: "Pendulum energy", defaultWeight: 0.5, description: "Reward useful height and angular speed." },
      { key: "swing_lateral_penalty", label: "Sideways offset", defaultWeight: 3, penalty: true, description: "Charge lateral displacement from the swing plane." },
      { key: "swing_lateral_barrier_penalty", label: "Lateral barrier", defaultWeight: 8, penalty: true, description: "Strongly charge motion beyond the safe lateral band." },
      { key: "swing_lateral_velocity_penalty", label: "Sideways velocity", defaultWeight: 3, penalty: true, description: "Charge fast sideways movement." },
      { key: "swing_out_of_plane_penalty", label: "Out-of-plane rotation", defaultWeight: 1, penalty: true, description: "Charge roll and yaw angular velocity." },
      { key: "swing_alignment_penalty", label: "Attachment alignment", defaultWeight: 18, penalty: true, description: "Keep the two attachment axes aligned." },
      { key: "swing_alignment_barrier_penalty", label: "Alignment barrier", defaultWeight: 4, penalty: true, description: "Strongly charge alignment beyond the valid range." },
      { key: "string_slack_penalty", label: "String slack", defaultWeight: 8, penalty: true, description: "Charge slack or unequal string lengths." },
      { key: "string_extension_penalty", label: "String extension", defaultWeight: 12, penalty: true, description: "Charge strings stretched beyond their intended length." },
      { key: "invalid_episode_penalty", label: "Invalid geometry", defaultWeight: 10, penalty: true, description: "Charge an episode after persistent mechanism invalidity." },
      { key: "action_rate_penalty", label: "Action changes", defaultWeight: 0.03, penalty: true, description: "Charge abrupt changes in policy actions." },
      { key: "joint_torque_penalty", label: "Joint torque", defaultWeight: 0.001, penalty: true, description: "Charge excessive actuator force." },
      { key: "joint_limit_penalty", label: "Joint limits", defaultWeight: 1, penalty: true, description: "Charge motion beyond joint limits." },
    ],
  },
  running: {
    summary:
      "PPO follows a forward-speed command, earns controlled flight and posture rewards, and pays for foot, actuator, and body-motion errors.",
    formula:
      "reward = speed tracking + turn tracking + air time + posture - clearance error - slip - jerk - joint limits - body roll",
    terms: [
      { key: "keep_pace", label: "Match speed", defaultWeight: 4, description: "Match the commanded body-frame velocity." },
      { key: "track_turn", label: "Match turn rate", defaultWeight: 2, description: "Match the commanded yaw rate." },
      { key: "air_time", label: "Running flight", defaultWeight: 3, description: "Reward feet spending a useful duration in the air." },
      { key: "flight", label: "Both-feet flight", defaultWeight: 0, description: "Optional bounded reward for both feet off the ground, gated by upright posture and command-directed body speed; disabled by default." },
      { key: "stay_upright", label: "Stay upright", defaultWeight: 2, description: "Keep the body upright while permitting running lean." },
      { key: "pose", label: "Running pose", defaultWeight: 1, description: "Track a speed-appropriate leg posture." },
      { key: "head_up", label: "Head posture", defaultWeight: 3.5, description: "Keep the head aligned with its command." },
      { key: "foot_clearance", label: "Foot clearance", defaultWeight: 2, penalty: true, description: "Charge a moving foot at the wrong height." },
      { key: "plant_the_foot", label: "Planted-foot slip", defaultWeight: 0.1, penalty: true, description: "Charge a planted foot that skids." },
      { key: "smooth_moves", label: "Action smoothness", defaultWeight: 1, penalty: true, description: "Charge jerky actions as the gait curriculum advances." },
      { key: "no_limit_parking", label: "Joint limits", defaultWeight: 1, penalty: true, description: "Charge joints held against their end stops." },
      { key: "calm_roll", label: "Body roll rate", defaultWeight: 0.025, penalty: true, description: "Charge excessive trunk roll and pitch velocity." },
    ],
  },
  stilts: {
    summary:
      "PPO follows walking commands while prioritizing upright balance on the selected stilts, controlled foot flight, head posture, and smooth body motion.",
    formula:
      "reward = linear tracking + turn tracking + upright balance + foot air time + head pose + optional leg pose - action changes - body angular velocity",
    terms: [
      { key: "track_lin_vel", label: "Match walking speed", defaultWeight: 2.5, description: "Match the requested linear velocity." },
      { key: "track_ang_vel", label: "Match turn rate", defaultWeight: 1, description: "Match the requested yaw rate." },
      { key: "upright", label: "Upright balance", defaultWeight: 3, description: "Prioritize a level trunk on the raised contacts." },
      { key: "pose", label: "Leg pose", defaultWeight: 0, description: "Optional pull toward the default leg pose; disabled by default." },
      { key: "head_pose", label: "Head pose", defaultWeight: 0.25, description: "Track the requested head posture without dominating balance." },
      { key: "feet_air_time", label: "Foot air time", defaultWeight: 2, description: "Reward controlled swing phases while walking." },
      { key: "action_rate_penalty", label: "Action changes", defaultWeight: 0.2, penalty: true, editable: false, description: "Apply 20% of the walking curriculum's dynamic action-rate penalty." },
      { key: "ang_vel_xy_penalty", label: "Body angular velocity", defaultWeight: 0.05, penalty: true, description: "Charge excessive roll and pitch velocity." },
    ],
  },
};

export const EXPERIMENTS: readonly ExperimentDefinition[] = [
  {
    id: "dance",
    title: "Dance imitation",
    shortTitle: "Dance",
    goal: "Learn a smooth, repeatable two-minute dance from an authored motion loop.",
    description:
      "A clip-conditioned PPO policy learns the 120 BPM choreography while preserving balance and the deployable observation contract.",
    task: "RLX imitate · dance-120bpm",
    result: "Local Studio baseline with deterministic rollout and visual review.",
    metricLabel: "Mean return",
    metricKey: "mean_return",
    metricSuffix: "",
    preview: "/experiments/dance/preview.mp4",
    poster: "/experiments/dance/poster.png",
    video: "/experiments/dance/preview.mp4",
    evidence: ["61 observations", "14 actions", "50 Hz", "2-minute render supported"],
    artifactStem: "dance",
    defaultRunName: "dance-studio",
    fullTimesteps: 1_000_000,
    fullEnvs: 16,
    maxEpisodeSeconds: 4,
    ppo: {
      learningRate: 0.0003,
      gamma: 0.99,
      clipCoefficient: 0.2,
      updateEpochs: 5,
      entropyCoefficient: 0.01,
      maxGradNorm: 0.5,
    },
    guideAnchor: "dance-imitation",
    guidance: {
      requiredInput:
        "One saved animation clip containing the poses and timing for the choreography. A velocity command is not required.",
      steps: [
        "Open Dance imitation.",
        "Use Animate to create or edit the dance clip.",
        "Save the clip.",
        "Select that clip for Dance training.",
        "Run Pipeline smoke first.",
        "Switch to Full training and select Start RLX.",
        "Watch Reward history.",
        "Run Evaluate.",
        "Run Render to generate the two-minute video.",
        "Review the video before accepting the policy.",
      ],
      distinctions: [
        "Clip means “perform these poses at this timing.”",
        "Command means “move at this speed or turn rate.” Dance imitation does not require a command.",
        "PPO is the learning algorithm. It learns from the selected recipe’s observations and reward.",
        "The clip supplies the choreography, while PPO learns balance, contacts, momentum, and actuator actions.",
      ],
      output:
        "A checkpoint, deterministic evaluation record, two-minute rollout video, contact sheet, metadata, and deployable ONNX policy.",
    },
    reward: REWARDS.dance,
  },
  {
    id: "swing",
    title: "Self-pumped swing",
    shortTitle: "Swing",
    goal: "Start still and discover coordinated body motion that pumps the swing.",
    description:
      "A mechanism-aware RLX task rewards bidirectional arc growth while guarding string tension, lateral motion, and attachment alignment.",
    task: "RLX swing · tension-only strings",
    result: "Published reference reaches a 173.20° strict full span.",
    metricLabel: "Peak-to-peak span",
    metricKey: "swing_span_deg",
    metricSuffix: "°",
    preview: "/experiments/swing/preview.gif",
    poster: "/experiments/swing/poster.jpg",
    video: "/experiments/swing/full.mp4",
    evidence: ["173.20° reference span", "71/100 strict seeds", "0.7 action scale"],
    artifactStem: "swing",
    defaultRunName: "swing-studio-01",
    fullTimesteps: 4_000_000,
    fullEnvs: 16,
    maxEpisodeSeconds: 24,
    ppo: {
      learningRate: 0.0001,
      gamma: 0.995,
      clipCoefficient: 0.1,
      updateEpochs: 3,
      entropyCoefficient: 0.002,
      maxGradNorm: 1,
    },
    guideAnchor: "self-pumped-swing",
    guidance: {
      requiredInput:
        "No clip and no velocity command. The swing mechanism, initial still state, and pumping rewards define what PPO must discover.",
      steps: [
        "Open Self-pumped swing.",
        "Confirm the swing mechanism recipe is selected; do not add a dance clip or movement command.",
        "Run Pipeline smoke first to verify the mechanism, observations, rewards, and artifact path.",
        "Switch to Full training and select Start RLX.",
        "Watch Reward history and the measured swing span as training progresses.",
        "Run Full evaluation: the default skill target is a symmetric 150° span (75° each side), all 24 seconds, valid geometry, and tensioned strings in every episode. Pipeline smoke does not assess skill.",
        "Run Render to generate the complete swing rollout.",
        "Review the video for a full controlled arc, tensioned strings, and low sideways drift.",
        "Accept the policy only after the evaluation and visual review both pass.",
      ],
      distinctions: [
        "Clip is not used because PPO must discover the pumping motion.",
        "Command is not used because the objective is mechanism motion, not a requested speed or turn rate.",
        "PPO learns from swing angle growth and safety rewards while choosing the 14 actuator actions.",
      ],
      output:
        "A swing checkpoint, strict-span evaluation, full rollout video, contact sheet, metadata, and deployable ONNX policy.",
    },
    reward: REWARDS.swing,
  },
  {
    id: "running",
    title: "Fast running",
    shortTitle: "Running",
    goal: "Discover a fast forward gait, then harden it against delay and disturbances.",
    description:
      "A speed curriculum prioritizes forward progress, controlled flight, low drift, and stable recovery under local MuJoCo simulation.",
    task: "RLX run · forward speed curriculum",
    result: "Published reference: 1.651 m/s nominal and 1.612 m/s under stress.",
    metricLabel: "Forward speed",
    metricKey: "forward_speed_m_s",
    metricSuffix: " m/s",
    preview: "/experiments/running/preview.gif",
    poster: "/experiments/running/poster.jpg",
    video: "/experiments/running/full.mp4",
    evidence: ["1.651 m/s nominal", "1.612 m/s stressed", "98.44% stressed survival"],
    artifactStem: "running",
    defaultRunName: "running-studio",
    fullTimesteps: 4_000_000,
    fullEnvs: 16,
    maxEpisodeSeconds: 12,
    ppo: {
      learningRate: 0.0003,
      gamma: 0.99,
      clipCoefficient: 0.2,
      updateEpochs: 5,
      entropyCoefficient: 0.01,
      maxGradNorm: 0.5,
    },
    guideAnchor: "fast-running",
    guidance: {
      requiredInput:
        "No animation clip. The environment supplies forward-speed commands and a speed curriculum for PPO to follow.",
      steps: [
        "Open Fast running.",
        "Confirm the forward-speed recipe is selected; no dance clip is needed.",
        "Run Pipeline smoke first to verify commands, observations, rewards, and artifact output.",
        "Switch to Full training and select Start RLX.",
        "Watch Reward history, forward speed, survival, and drift.",
        "Run Evaluate under nominal conditions and under backlash plus disturbance stress.",
        "Run Render to generate a sustained running rollout.",
        "Review the video for forward progress, controlled flight, low sideways drift, and recovery without falls.",
        "Accept the policy only after the speed, stability, and visual checks pass.",
      ],
      distinctions: [
        "Clip is not used because running is command-driven rather than pose-timed choreography.",
        "Command means the target forward speed sampled by the environment.",
        "PPO learns the gait, balance, contacts, momentum, and actuator actions that satisfy those commands.",
      ],
      output:
        "A running checkpoint, nominal and stressed evaluation metrics, rollout video, contact sheet, metadata, and deployable ONNX policy.",
    },
    reward: REWARDS.running,
  },
  {
    id: "stilts",
    title: "Stilt walking",
    shortTitle: "Stilts",
    goal: "Learn stable locomotion on a selected stilt height and support shape.",
    description:
      "A morphology-specific RLX recipe trains balance and forward motion on explicit stilt contact geometry, beginning with the 10 cm blend-0.50 setup.",
    task: "RLX stilts · fixed morphology",
    result: "Published 10 cm reference: 0.14055 m/s, 100% survival, 3.28° median max tilt.",
    metricLabel: "Forward speed",
    metricKey: "forward_speed_m_s",
    metricSuffix: " m/s",
    preview: "/experiments/stilts/preview.gif",
    poster: "/experiments/stilts/poster.jpg",
    video: "/experiments/stilts/full.mp4",
    evidence: ["10 cm shown", "0.50 support blend", "8 released heights"],
    artifactStem: "stilts",
    defaultRunName: "stilts-10cm-studio",
    fullTimesteps: 4_000_000,
    fullEnvs: 16,
    maxEpisodeSeconds: 12,
    ppo: {
      learningRate: 0.0003,
      gamma: 0.99,
      clipCoefficient: 0.2,
      updateEpochs: 5,
      entropyCoefficient: 0.01,
      maxGradNorm: 0.5,
    },
    guideAnchor: "stilt-walking",
    guidance: {
      requiredInput:
        "No animation clip. Choose the stilt height, support blend, and mass; the environment then supplies walking velocity commands.",
      steps: [
        "Open Stilt walking.",
        "Choose the stilt height, support blend, and mass you want to train.",
        "Run Pipeline smoke first to verify that exact morphology, commands, rewards, and artifact path.",
        "Switch to Full training and select Start RLX.",
        "Watch Reward history, forward speed, survival, tilt, and contact stability.",
        "Run Evaluate with the same stilt height, blend, and mass used for training.",
        "Run Render to generate the stilt-walking rollout.",
        "Review the video for stable contacts, controlled tilt, forward progress, and recovery without falls.",
        "Accept the policy only for the morphology recorded in its metadata.",
      ],
      distinctions: [
        "Clip is not used because stilt walking is command-driven locomotion.",
        "Command means the requested walking speed or turn rate supplied by the environment.",
        "PPO learns balance and actuator actions for one explicit stilt morphology; changing the hardware shape requires matching training and evaluation.",
      ],
      output:
        "A morphology-specific checkpoint, evaluation record, rollout video, contact sheet, metadata with stilt dimensions, and deployable ONNX policy.",
    },
    reward: REWARDS.stilts,
    controls: {
      stiltHeightCm: 10,
      stiltBlend: 0.5,
      stiltMassKg: 0.029,
    },
  },
] as const;

export function getExperiment(id: unknown): ExperimentDefinition {
  return (
    EXPERIMENTS.find((experiment) => experiment.id === id) ?? EXPERIMENTS[0]
  );
}

export function isExperimentId(value: unknown): value is ExperimentId {
  return EXPERIMENTS.some((experiment) => experiment.id === value);
}

export function defaultRewardWeights(
  experimentId: ExperimentId
): Record<string, number> {
  return Object.fromEntries(
    getExperiment(experimentId).reward.terms
      .filter((term) => term.editable !== false)
      .map((term) => [term.key, term.defaultWeight])
  );
}
