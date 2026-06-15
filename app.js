const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const ctx = overlay.getContext("2d");

const startButton = document.querySelector("#startButton");
const soundOnlyButton = document.querySelector("#soundOnlyButton");
const addRepButton = document.querySelector("#addRepButton");
const resetButton = document.querySelector("#resetButton");
const depthSlider = document.querySelector("#depthSlider");
const targetRepsInput = document.querySelector("#targetRepsInput");
const modeButtons = document.querySelectorAll(".mode-button");
const presenceStatus = document.querySelector("#presenceStatus");
const phaseText = document.querySelector("#phaseText");
const repCountEl = document.querySelector("#repCount");
const depthScoreEl = document.querySelector("#depthScore");
const layerCountEl = document.querySelector("#layerCount");
const goalProgressEl = document.querySelector("#goalProgress");
const soundRows = document.querySelectorAll(".sound-row");

const state = {
  mode: "auto",
  reps: 0,
  depth: 0.65,
  targetReps: 8,
  layerCount: 1,
  phase: "idle",
  inStation: false,
  workoutReady: false,
  readyReason: "기구 밖",
  exercise: "idle",
  direction: "top",
  baseline: null,
  peakDepth: 0,
  poseReady: false,
  fallbackPrevious: null,
  fallbackMotion: 0,
  lastChimeAt: 0,
};

let poseLandmarker = null;
let lastVideoTime = -1;
let audio = null;
let animationFrame = null;

const POSE_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const WASM_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

startButton.addEventListener("click", startExperience);
soundOnlyButton.addEventListener("click", startSoundOnly);
addRepButton.addEventListener("click", addTestRep);
depthSlider.addEventListener("input", () => {
  state.depth = Number(depthSlider.value) / 100;
  updateUI();
});
targetRepsInput.addEventListener("input", () => {
  state.targetReps = clamp(Math.round(Number(targetRepsInput.value) || 1), 1, 30);
  targetRepsInput.value = String(state.targetReps);
  syncDopamineToGoal();
  updateUI();
});
resetButton.addEventListener("click", resetSession);

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    state.baseline = null;
    modeButtons.forEach((item) => item.classList.toggle("active", item === button));
  });
});

async function startExperience() {
  startButton.disabled = true;
  startButton.textContent = "시작 중";
  phaseText.textContent = "권한 요청";

  try {
    await startCamera();
    if (!audio) {
      audio = createMountainAudio();
    }
    await audio.start();
    audio.keepMountainBed();
    await setupPose();
    resizeCanvas();
    startButton.textContent = "실행 중";
    soundOnlyButton.disabled = true;
    phaseText.textContent = "기구 감지";
    animationFrame = requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    startButton.disabled = false;
    startButton.textContent = "다시 시작";
    phaseText.textContent = "권한 확인";
    presenceStatus.textContent = "오류";
  }
}

async function startSoundOnly() {
  soundOnlyButton.disabled = true;
  soundOnlyButton.textContent = "산 배경음 재생 중";
  phaseText.textContent = "사운드 테스트";
  presenceStatus.textContent = "사운드만";

  try {
    if (!audio) {
      audio = createMountainAudio();
    }
    await audio.start();
    audio.keepMountainBed();
    addRepButton.disabled = false;
  } catch (error) {
    console.error(error);
    soundOnlyButton.disabled = false;
    soundOnlyButton.textContent = "산 배경음만 테스트";
    phaseText.textContent = "사운드 오류";
    presenceStatus.textContent = "오류";
  }
}

async function addTestRep() {
  if (!audio) {
    await startSoundOnly();
  }

  if (!audio) return;

  state.depth = Number(depthSlider.value) / 100;
  state.reps += 1;
  phaseText.textContent = "도파민 레이어";
  presenceStatus.textContent = "사운드만";
  syncDopamineToGoal();
  updateUI();
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();
}

async function setupPose() {
  try {
    const vision = await import(POSE_CDN);
    const resolver = await vision.FilesetResolver.forVisionTasks(WASM_CDN);
    poseLandmarker = await vision.PoseLandmarker.createFromOptions(resolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.45,
      minPosePresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
    });
    state.poseReady = true;
  } catch (error) {
    console.warn("Pose model failed, using motion fallback.", error);
    state.poseReady = false;
  }
}

function loop() {
  resizeCanvas();
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (state.poseReady && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = poseLandmarker.detectForVideo(video, performance.now());
    handlePose(result.landmarks?.[0] || null);
  } else if (!state.poseReady) {
    handleFallbackMotion();
  }

  updateSound();
  updateUI();
  animationFrame = requestAnimationFrame(loop);
}

function handlePose(landmarks) {
  if (!landmarks) {
    finishInteraction();
    return;
  }

  const points = pickLandmarks(landmarks);
  const stationPoints = [
    points.nose,
    points.leftShoulder,
    points.rightShoulder,
    points.leftElbow,
    points.rightElbow,
    points.leftWrist,
    points.rightWrist,
    points.leftHip,
    points.rightHip,
  ];
  const visibility = stationPoints.reduce((sum, point) => sum + point.visibility, 0) / stationPoints.length;
  const centerX = (points.leftShoulder.x + points.rightShoulder.x + points.leftHip.x + points.rightHip.x) / 4;
  const shoulderWidth = distance(points.leftShoulder, points.rightShoulder);
  const torsoHeight =
    (distance(points.leftShoulder, points.leftHip) + distance(points.rightShoulder, points.rightHip)) / 2;
  const stationConfidence = visibility > 0.48 && shoulderWidth > 0.08 && torsoHeight > 0.14;

  state.inStation = stationConfidence && centerX > 0.18 && centerX < 0.82;

  drawSkeleton(points, state.inStation);

  if (!state.inStation) {
    finishInteraction();
    return;
  }

  const readiness = getWorkoutReadiness(points);
  state.workoutReady = readiness.ready;
  state.readyReason = readiness.reason;

  if (!readiness.ready) {
    if (readiness.ended) {
      finishInteraction();
      return;
    }
    state.exercise = "ready";
    state.depth = Math.max(0, state.depth - 0.045);
    state.layerCount = 1;
    state.baseline = null;
    state.peakDepth = 0;
    state.direction = "top";
    return;
  }

  if (state.phase === "idle") {
    startInteractionSession();
  }

  state.exercise = readiness.exercise;

  const movementY = getMovementSignal(points, state.exercise);
  processRepSignal(movementY);
}

function pickLandmarks(landmarks) {
  return {
    nose: landmarks[0],
    leftShoulder: landmarks[11],
    rightShoulder: landmarks[12],
    leftElbow: landmarks[13],
    rightElbow: landmarks[14],
    leftWrist: landmarks[15],
    rightWrist: landmarks[16],
    leftHip: landmarks[23],
    rightHip: landmarks[24],
    leftKnee: landmarks[25],
    rightKnee: landmarks[26],
    leftAnkle: landmarks[27],
    rightAnkle: landmarks[28],
  };
}

function getWorkoutReadiness(points) {
  const wristY = (points.leftWrist.y + points.rightWrist.y) / 2;
  const wristX = (points.leftWrist.x + points.rightWrist.x) / 2;
  const shoulderY = (points.leftShoulder.y + points.rightShoulder.y) / 2;
  const shoulderX = (points.leftShoulder.x + points.rightShoulder.x) / 2;
  const hipY = (points.leftHip.y + points.rightHip.y) / 2;
  const hipX = (points.leftHip.x + points.rightHip.x) / 2;
  const ankleY = (points.leftAnkle.y + points.rightAnkle.y) / 2;
  const shoulderWidth = distance(points.leftShoulder, points.rightShoulder);
  const wristWidth = distance(points.leftWrist, points.rightWrist);
  const lowerBodyVisible =
    points.leftKnee.visibility > 0.42 &&
    points.rightKnee.visibility > 0.42 &&
    points.leftAnkle.visibility > 0.42 &&
    points.rightAnkle.visibility > 0.42;
  const fullBodyVisible = lowerBodyVisible && ankleY - points.nose.y > 0.42;
  const feetOffGround = fullBodyVisible && ankleY < 0.93;
  const elbowAngle =
    (angle(points.leftShoulder, points.leftElbow, points.leftWrist) +
      angle(points.rightShoulder, points.rightElbow, points.rightWrist)) /
    2;
  const handsOverBarLike =
    wristY < shoulderY - 0.13 &&
    wristWidth > shoulderWidth * 0.75 &&
    Math.abs(wristX - shoulderX) < shoulderWidth * 0.75;
  const dipSupportLike =
    wristY > shoulderY - 0.03 &&
    wristY < hipY + 0.13 &&
    Math.abs(wristX - hipX) < shoulderWidth * 0.95 &&
    elbowAngle < 158;

  if (!lowerBodyVisible) {
    return { ready: false, exercise: "idle", reason: "다리까지 보여야 함", ended: false };
  }

  if (!feetOffGround) {
    return { ready: false, exercise: "idle", reason: "발이 땅에 닿음", ended: true };
  }

  if (state.mode === "pullup") {
    return { ready: true, exercise: "pullup", reason: handsOverBarLike ? "턱걸이 준비" : "공중 대기" };
  }

  if (state.mode === "dip") {
    return { ready: true, exercise: "dip", reason: dipSupportLike ? "딥스 준비" : "공중 대기" };
  }

  if (handsOverBarLike) {
    return { ready: true, exercise: "pullup", reason: "턱걸이 준비" };
  }

  if (dipSupportLike) {
    return { ready: true, exercise: "dip", reason: "딥스 준비" };
  }

  return { ready: true, exercise: state.exercise === "dip" ? "dip" : "pullup", reason: "공중 대기" };
}

function getMovementSignal(points, exercise) {
  if (exercise === "dip") {
    return (points.leftShoulder.y + points.rightShoulder.y) / 2;
  }
  return points.nose.y;
}

function processRepSignal(signalY) {
  if (state.baseline === null) {
    state.baseline = signalY;
    return;
  }

  state.baseline = state.baseline * 0.985 + signalY * 0.015;

  const rawDepth =
    state.exercise === "dip"
      ? Math.max(0, signalY - state.baseline)
      : Math.max(0, state.baseline - signalY);

  state.depth = clamp(rawDepth / 0.18, 0, 1);
  state.peakDepth = Math.max(state.peakDepth, state.depth);

  if (state.phase === "ready" && state.depth > 0.25) {
    state.phase = "exercising";
  }

  if (state.direction === "top" && state.depth > 0.58) {
    state.direction = "bottom";
  }

  if (state.direction === "bottom" && state.depth < 0.24) {
    if (state.peakDepth > 0.58) {
      state.reps += 1;
      triggerRepAccent(state.peakDepth);
    }
    state.peakDepth = 0;
    state.direction = "top";
  }

  if (state.phase === "exercising" && state.reps === 0) {
    state.layerCount = 1;
  }
}

function handleFallbackMotion() {
  const sampleSize = 80;
  const temp = document.createElement("canvas");
  temp.width = sampleSize;
  temp.height = sampleSize;
  const tempCtx = temp.getContext("2d");
  tempCtx.drawImage(video, 0, 0, sampleSize, sampleSize);
  const frame = tempCtx.getImageData(0, 0, sampleSize, sampleSize).data;

  if (!state.fallbackPrevious) {
    state.fallbackPrevious = frame;
    return;
  }

  let diff = 0;
  for (let index = 0; index < frame.length; index += 16) {
    diff += Math.abs(frame[index] - state.fallbackPrevious[index]);
  }

  state.fallbackPrevious = frame;
  state.fallbackMotion = state.fallbackMotion * 0.86 + (diff / 70000) * 0.14;
  state.inStation = false;
  state.workoutReady = false;
  state.readyReason = "포즈 모델 필요";
  state.exercise = "idle";
  state.depth = clamp(state.fallbackMotion * 2.8, 0, 1);
  state.layerCount = 1;

  drawFallbackField();
}

function markResting() {
  state.inStation = false;
  state.workoutReady = false;
  state.readyReason = "기구 밖";
  state.exercise = "idle";
  state.depth = Number(depthSlider.value) / 100;
  state.layerCount = 1;
  state.baseline = null;
  state.peakDepth = 0;
  state.direction = "top";
}

function startInteractionSession() {
  resetWorkoutProgress();
  state.phase = "ready";
  state.workoutReady = true;
  state.readyReason = "공중 대기";
  if (audio) {
    audio.keepMountainBed();
  }
}

function finishInteraction() {
  if (state.phase !== "idle" || state.reps > 0 || state.layerCount > 1) {
    resetWorkoutProgress();
    if (audio) {
      audio.fadeOutDopamine();
      audio.keepMountainBed();
    }
  }

  markResting();
  state.phase = "idle";
}

function resetWorkoutProgress() {
  state.reps = 0;
  state.depth = Number(depthSlider.value) / 100;
  state.layerCount = 1;
  state.baseline = null;
  state.peakDepth = 0;
  state.direction = "top";
}

function drawSkeleton(points, active) {
  const color = active ? "#74d48a" : "#ee716a";
  const pairs = [
    ["leftShoulder", "rightShoulder"],
    ["leftShoulder", "leftElbow"],
    ["leftElbow", "leftWrist"],
    ["rightShoulder", "rightElbow"],
    ["rightElbow", "rightWrist"],
    ["leftShoulder", "leftHip"],
    ["rightShoulder", "rightHip"],
    ["leftHip", "rightHip"],
    ["leftHip", "leftKnee"],
    ["rightHip", "rightKnee"],
    ["leftKnee", "leftAnkle"],
    ["rightKnee", "rightAnkle"],
  ];

  ctx.lineWidth = 4;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;

  pairs.forEach(([from, to]) => {
    drawLine(points[from], points[to]);
  });

  Object.values(points).forEach((point) => {
    const x = point.x * overlay.width;
    const y = point.y * overlay.height;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawLine(from, to) {
  ctx.beginPath();
  ctx.moveTo(from.x * overlay.width, from.y * overlay.height);
  ctx.lineTo(to.x * overlay.width, to.y * overlay.height);
  ctx.stroke();
}

function drawFallbackField() {
  const bars = 18;
  ctx.strokeStyle = "rgba(116, 212, 138, 0.64)";
  ctx.lineWidth = 2;
  for (let index = 0; index < bars; index += 1) {
    const x = (overlay.width / bars) * index;
    const height = overlay.height * (0.12 + state.depth * Math.sin(index * 1.7) * 0.16 + state.depth * 0.34);
    ctx.beginPath();
    ctx.moveTo(x, overlay.height);
    ctx.lineTo(x, overlay.height - height);
    ctx.stroke();
  }
}

function updateSound() {
  if (!audio) return;
  audio.keepMountainBed();
}

function triggerRepAccent(depth) {
  if (!audio || performance.now() - state.lastChimeAt < 600) return;
  state.lastChimeAt = performance.now();
  state.depth = depth;
  syncDopamineToGoal();
}

function syncDopamineToGoal() {
  state.targetReps = clamp(Math.round(Number(targetRepsInput.value) || state.targetReps), 1, 30);
  const progress = getDopamineProgress();
  const intensity = getDopamineIntensity();
  const desiredLayerCount = getDesiredLayerCount();
  state.layerCount = 1 + desiredLayerCount;

  if (audio) {
    audio.setDopamineLayerCount(desiredLayerCount, state.depth, intensity);
  }
}

function getDesiredLayerCount() {
  if (state.reps === 0) return 0;
  const dopamineLayerCount = soundRows.length - 1;
  if (state.targetReps <= 1) return dopamineLayerCount;

  const postFirstProgress = clamp((state.reps - 1) / (state.targetReps - 1), 0, 1);
  const backLoadedProgress = Math.pow(postFirstProgress, 1.8);
  return Math.min(dopamineLayerCount, 1 + Math.floor(backLoadedProgress * (dopamineLayerCount - 1)));
}

function getDopamineProgress() {
  return clamp(state.reps / state.targetReps, 0, 1);
}

function getDopamineIntensity() {
  if (state.reps === 0) return 0;
  const progress = getDopamineProgress();
  return clamp(0.22 + Math.pow(progress, 1.85) * 0.78, 0, 1);
}

function updateUI() {
  presenceStatus.textContent =
    state.phase === "exercising"
      ? "운동 중"
      : state.phase === "ready"
        ? "준비 완료"
        : "숲 대기";
  repCountEl.textContent = String(state.reps);
  depthScoreEl.textContent = `${Math.round(state.depth * 100)}%`;
  layerCountEl.textContent = String(state.layerCount);
  goalProgressEl.textContent = `${Math.round(getDopamineProgress() * 100)}%`;

  const labels = {
    idle: "산 사운드",
    ready: state.readyReason,
    pullup: "턱걸이",
    dip: "딥스",
    motion: "움직임",
  };
  phaseText.textContent =
    state.phase === "ready"
      ? "인터랙션 준비"
      : state.phase === "exercising"
        ? labels[state.exercise] || "운동 중"
        : labels[state.exercise] || "산 사운드";

  soundRows.forEach((row, index) => {
    row.classList.toggle("active", index < state.layerCount);
  });
}

function resetSession() {
  state.targetReps = clamp(Math.round(Number(targetRepsInput.value) || 8), 1, 30);
  resetWorkoutProgress();
  state.phase = "idle";
  if (audio) {
    audio.fadeOutDopamine();
    audio.keepMountainBed();
  }
  updateUI();
}

function resizeCanvas() {
  const rect = overlay.getBoundingClientRect();
  const width = Math.round(rect.width * window.devicePixelRatio);
  const height = Math.round(rect.height * window.devicePixelRatio);
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
}

function createMountainAudio() {
  const context = new AudioContext();
  const master = context.createGain();
  const dopamineBus = context.createGain();
  const hallBus = context.createGain();
  const hall = context.createConvolver();
  const forestGain = context.createGain();
  const windGain = context.createGain();
  const waterGain = context.createGain();
  const pulseGain = context.createGain();

  master.gain.value = 0.58;
  dopamineBus.gain.value = 3.1;
  hallBus.gain.value = 0.32;
  hall.buffer = createHallImpulse(context, 2.8);
  forestGain.gain.value = 0;
  windGain.gain.value = 0;
  waterGain.gain.value = 0;
  pulseGain.gain.value = 0;

  forestGain.connect(master);
  windGain.connect(master);
  waterGain.connect(master);
  pulseGain.connect(master);
  dopamineBus.connect(master);
  dopamineBus.connect(hall).connect(hallBus).connect(master);
  master.connect(context.destination);

  let forestSource = null;
  let forestBufferPromise = null;

  const wind = noiseSource(context, 2);
  const windFilter = context.createBiquadFilter();
  windFilter.type = "lowpass";
  windFilter.frequency.value = 420;
  wind.connect(windFilter).connect(windGain);

  const water = noiseSource(context, 1);
  const waterFilter = context.createBiquadFilter();
  waterFilter.type = "highpass";
  waterFilter.frequency.value = 1200;
  water.connect(waterFilter).connect(waterGain);

  const pulse = context.createOscillator();
  pulse.type = "sine";
  pulse.frequency.value = 92;
  pulse.connect(pulseGain);

  const lfo = context.createOscillator();
  const lfoGain = context.createGain();
  lfo.frequency.value = 0.08;
  lfoGain.gain.value = 170;
  lfo.connect(lfoGain).connect(windFilter.frequency);

  wind.start();
  water.start();
  pulse.start();
  lfo.start();

  const dopamineLayers = [];
  const layerRecipes = [
    { name: "strings", part: "strings", gain: 0.19, filter: 1250 },
    { name: "brass", part: "brass", gain: 0.22, filter: 1400 },
    { name: "bass", part: "timpani", gain: 0.18, filter: 320 },
    { name: "choir", part: "choir", gain: 0.16, filter: 1550 },
    { name: "timpani", part: "drums", gain: 0.19, filter: 420 },
  ];

  return {
    async start() {
      if (context.state !== "running") await context.resume();
      await startForestLoop();
    },
    keepMountainBed() {
      ramp(context, master.gain, 0.58, 1.2);
      ramp(context, forestGain.gain, 0.45, 1.2);
      ramp(context, windGain.gain, 0, 1.2);
      ramp(context, waterGain.gain, 0, 1.2);
      ramp(context, pulseGain.gain, 0, 1.2);
    },
    setDopamineLayerCount(count, depth, progress) {
      const targetCount = clamp(count, 0, layerRecipes.length);
      while (dopamineLayers.length < targetCount) {
        this.addDopamineLayer(depth);
      }

      while (dopamineLayers.length > targetCount) {
        const layer = dopamineLayers.pop();
        ramp(context, layer.gain.gain, 0, 0.6);
        layer.stop();
      }

      dopamineLayers.forEach((layer, index) => {
        const recipe = layerRecipes[index];
        const grandeur = 0.45 + depth * 0.85;
        const progressLift = 0.65 + progress * 0.65;
        ramp(context, layer.gain.gain, recipe.gain * grandeur * progressLift, 1.1);
      });
    },
    addDopamineLayer(depth) {
      const recipe = layerRecipes[dopamineLayers.length % layerRecipes.length];
      const grandeur = 0.45 + depth * 0.85;
      const layer = createMarchLayer(context, dopamineBus, recipe, depth);
      ramp(context, layer.gain.gain, recipe.gain * grandeur, 1.4);
      dopamineLayers.push(layer);
    },
    resetDopamine() {
      dopamineLayers.splice(0).forEach((layer) => {
        ramp(context, layer.gain.gain, 0, 0.6);
        layer.stop();
      });
    },
    fadeOutDopamine() {
      dopamineLayers.splice(0).forEach((layer) => {
        ramp(context, layer.gain.gain, 0, 1.8);
        window.setTimeout(() => layer.stop(), 1900);
      });
    },
    setPresence(active) {
      void active;
    },
    setDepth(depth) {
      void depth;
    },
    setLayers(layerCount) {
      void layerCount;
    },
    chime(depth) {
      [523.25, 659.25, 783.99 + depth * 130].forEach((frequency, index) => {
        const osc = context.createOscillator();
        const gain = context.createGain();
        osc.type = "triangle";
        osc.frequency.value = frequency;
        gain.gain.setValueAtTime(0, context.currentTime);
        gain.gain.linearRampToValueAtTime(0.12, context.currentTime + 0.02 + index * 0.03);
        gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.75 + index * 0.08);
        osc.connect(gain).connect(master);
        osc.start(context.currentTime + index * 0.03);
        osc.stop(context.currentTime + 1.1 + index * 0.08);
      });
    },
  };

  async function startForestLoop() {
    if (forestSource) return;

    try {
      if (!forestBufferPromise) {
        forestBufferPromise = loadAudioBuffer(context, "./forest_loop2.wav");
      }
      const buffer = await forestBufferPromise;
      forestSource = context.createBufferSource();
      forestSource.buffer = buffer;
      forestSource.loop = true;
      forestSource.connect(forestGain);
      forestSource.start();
      ramp(context, forestGain.gain, 0.45, 2.6);
    } catch (error) {
      console.warn("forest_loop2.wav failed, using generated forest fallback.", error);
      forestSource = noiseSource(context, 2);
      const forestFilter = context.createBiquadFilter();
      forestFilter.type = "bandpass";
      forestFilter.frequency.value = 620;
      forestFilter.Q.value = 0.7;
      forestSource.connect(forestFilter).connect(forestGain);
      forestSource.start();
    }
  }
}

async function loadAudioBuffer(context, url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Audio load failed: ${response.status} ${url}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  return context.decodeAudioData(arrayBuffer);
}

function noiseSource(context, seconds) {
  const bufferSize = context.sampleRate * seconds;
  const buffer = context.createBuffer(1, bufferSize, context.sampleRate);
  const data = buffer.getChannelData(0);

  for (let index = 0; index < bufferSize; index += 1) {
    data[index] = Math.random() * 2 - 1;
  }

  const source = context.createBufferSource();
  source.buffer = buffer;
  source.loop = true;
  return source;
}

function createMarchLayer(context, destination, recipe, depth) {
  const gain = context.createGain();
  const filter = context.createBiquadFilter();
  const tempo = 118;
  const beat = 60 / tempo;
  const bar = beat * 4;
  const layer = { gain, timers: [], nodes: [], stop };
  const grandeur = 0.75 + depth * 0.8;

  gain.gain.value = 0;
  filter.type = "lowpass";
  filter.frequency.value = recipe.filter * grandeur;
  filter.Q.value = recipe.part === "brass" ? 0.9 : 0.65;
  gain.connect(filter).connect(destination);
  createSustainedPart(context, gain, recipe.part, depth, layer.nodes);

  let nextStart = context.currentTime + 0.08;
  scheduleMarchBar(context, gain, recipe.part, nextStart, beat, depth);
  nextStart += bar;

  const timer = window.setInterval(() => {
    scheduleMarchBar(context, gain, recipe.part, nextStart, beat, depth);
    nextStart += bar;
  }, bar * 1000);

  layer.timers.push(timer);
  return layer;

  function stop() {
    layer.timers.forEach((timerId) => window.clearInterval(timerId));
    layer.nodes.forEach((node) => {
      if (typeof node.stop === "function") {
        node.stop(context.currentTime + 0.7);
      }
    });
  }
}

function createSustainedPart(context, destination, part, depth, nodes) {
  const now = context.currentTime;
  const c2 = 65.41;
  const g2 = 98;
  const c3 = 130.81;
  const e3 = 164.81;
  const g3 = 196;
  const c4 = 261.63;
  const e4 = 329.63;
  const g4 = 392;
  const grandeur = 0.9 + depth * 0.8;
  const parts = {
    timpani: { type: "sine", notes: [c2, g2, c3], gain: 0.42 },
    strings: { type: "triangle", notes: [c3, g3, c4, e4], gain: 0.38 },
    brass: { type: "sawtooth", notes: [c3, e3, g3, c4], gain: 0.34 },
    choir: { type: "triangle", notes: [g2, c3, e3, g3, c4], gain: 0.28 },
    drums: { type: "sine", notes: [c2, g2], gain: 0.36 },
  };
  const config = parts[part] || parts.strings;
  const tremolo = context.createOscillator();
  const tremoloGain = context.createGain();

  tremolo.frequency.value = part === "strings" || part === "choir" ? 0.65 : 1.8;
  tremoloGain.gain.value = config.gain * 0.08 * grandeur;
  tremolo.connect(tremoloGain).connect(destination.gain);
  tremolo.start(now);
  nodes.push(tremolo, tremoloGain);

  config.notes.forEach((frequency, index) => {
    const osc = context.createOscillator();
    const voiceGain = context.createGain();
    const filter = context.createBiquadFilter();

    osc.type = config.type;
    osc.frequency.value = frequency;
    osc.detune.value = (index - config.notes.length / 2) * 2.5;
    voiceGain.gain.value = (config.gain * grandeur) / config.notes.length;
    filter.type = "lowpass";
    filter.frequency.value = part === "brass" ? 1450 : 980 + depth * 620;
    filter.Q.value = 0.5;
    osc.connect(filter).connect(voiceGain).connect(destination);
    osc.start(now);
    nodes.push(osc, voiceGain, filter);
  });
}

function scheduleMarchBar(context, destination, part, start, beat, depth) {
  const c2 = 65.41;
  const g2 = 98;
  const c3 = 130.81;
  const e3 = 164.81;
  const g3 = 196;
  const c4 = 261.63;
  const e4 = 329.63;
  const g4 = 392;

  if (part === "timpani") {
    [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5].forEach((step, index) => {
      const note = index % 2 === 0 ? c2 : g2;
      playTimpaniHit(context, destination, start + step * beat, note, 0.18 + depth * 0.15);
    });
    return;
  }

  if (part === "strings") {
    [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5].forEach((step, index) => {
      const note = [c3, g3, c4, g3][index % 4];
      playShortVoice(context, destination, start + step * beat, note, "triangle", 0.08, 0.16);
    });
    return;
  }

  if (part === "brass") {
    playOrchestraStab(context, destination, start + 0.02, [c3, e3, g3, c4], 0.2 + depth * 0.12);
    playOrchestraStab(context, destination, start + 1.5 * beat, [g2, c3, e3, g3], 0.17 + depth * 0.1);
    playOrchestraStab(context, destination, start + 2.45 * beat, [c3, e3, g3, c4], 0.24 + depth * 0.14);
    return;
  }

  if (part === "choir") {
    playSoftChord(context, destination, start, [c3, e3, g3, c4], 0.07 + depth * 0.045, barDuration(beat));
    playSoftChord(context, destination, start + 2 * beat, [g2, c3, e3, g3], 0.06 + depth * 0.04, barDuration(beat) / 2);
    return;
  }

  if (part === "drums") {
    [0, 1, 2, 3].forEach((step) => {
      playTimpaniHit(context, destination, start + step * beat, c2, 0.25 + depth * 0.18);
    });
    playCymbalSwell(context, destination, start + 2.8 * beat, 0.07 + depth * 0.06);
  }
}

function barDuration(beat) {
  return beat * 1.85;
}

function playShortVoice(context, destination, startTime, frequency, type, gainAmount, duration) {
  const osc = context.createOscillator();
  const gain = context.createGain();

  osc.type = type;
  osc.frequency.value = frequency;
  gain.gain.setValueAtTime(0.0001, startTime);
  gain.gain.exponentialRampToValueAtTime(gainAmount, startTime + 0.018);
  gain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);
  osc.connect(gain).connect(destination);
  osc.start(startTime);
  osc.stop(startTime + duration + 0.04);
}

function playSoftChord(context, destination, startTime, notes, gainAmount, duration) {
  const chordGain = context.createGain();
  chordGain.gain.setValueAtTime(0.0001, startTime);
  chordGain.gain.exponentialRampToValueAtTime(gainAmount, startTime + 0.18);
  chordGain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);
  chordGain.connect(destination);

  notes.forEach((frequency, index) => {
    const osc = context.createOscillator();
    const voiceGain = context.createGain();
    osc.type = "triangle";
    osc.frequency.value = frequency;
    osc.detune.value = (index - 1.5) * 2;
    voiceGain.gain.value = 1 / notes.length;
    osc.connect(voiceGain).connect(chordGain);
    osc.start(startTime);
    osc.stop(startTime + duration + 0.08);
  });
}

function playOrchestraStab(context, destination, startTime, notes, gainAmount) {
  const stabGain = context.createGain();
  const bodyFilter = context.createBiquadFilter();
  const biteFilter = context.createBiquadFilter();

  stabGain.gain.setValueAtTime(0.0001, startTime);
  stabGain.gain.exponentialRampToValueAtTime(gainAmount, startTime + 0.035);
  stabGain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.72);

  bodyFilter.type = "lowpass";
  bodyFilter.frequency.setValueAtTime(1250, startTime);
  bodyFilter.frequency.exponentialRampToValueAtTime(620, startTime + 0.55);
  bodyFilter.Q.value = 0.7;

  biteFilter.type = "peaking";
  biteFilter.frequency.value = 740;
  biteFilter.gain.value = 3.5;
  biteFilter.Q.value = 0.9;

  stabGain.connect(bodyFilter).connect(biteFilter).connect(destination);

  notes.forEach((frequency, index) => {
    [-5, 0, 4].forEach((detune, detuneIndex) => {
      const osc = context.createOscillator();
      const voiceGain = context.createGain();
      osc.type = detuneIndex === 1 ? "triangle" : "sawtooth";
      osc.frequency.value = frequency;
      osc.detune.value = detune + index * 1.5;
      voiceGain.gain.value = 1 / (notes.length * 3);
      osc.connect(voiceGain).connect(stabGain);
      osc.start(startTime);
      osc.stop(startTime + 0.78);
    });
  });
}

function playTimpaniHit(context, destination, startTime, frequency, gainAmount) {
  const drum = context.createOscillator();
  const drumGain = context.createGain();
  const drumFilter = context.createBiquadFilter();

  drum.type = "sine";
  drum.frequency.setValueAtTime(frequency, startTime);
  drum.frequency.exponentialRampToValueAtTime(42, startTime + 0.38);

  drumFilter.type = "lowpass";
  drumFilter.frequency.value = 260;
  drumFilter.Q.value = 1.3;

  drumGain.gain.setValueAtTime(0.0001, startTime);
  drumGain.gain.exponentialRampToValueAtTime(gainAmount, startTime + 0.018);
  drumGain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.72);

  drum.connect(drumFilter).connect(drumGain).connect(destination);
  drum.start(startTime);
  drum.stop(startTime + 0.78);
}

function playCymbalSwell(context, destination, startTime, gainAmount) {
  const cymbal = noiseSource(context, 0.9);
  const cymbalGain = context.createGain();
  const highpass = context.createBiquadFilter();
  const lowpass = context.createBiquadFilter();

  highpass.type = "highpass";
  highpass.frequency.value = 3600;
  lowpass.type = "lowpass";
  lowpass.frequency.value = 9200;

  cymbalGain.gain.setValueAtTime(0.0001, startTime);
  cymbalGain.gain.exponentialRampToValueAtTime(gainAmount, startTime + 0.04);
  cymbalGain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.78);

  cymbal.connect(highpass).connect(lowpass).connect(cymbalGain).connect(destination);
  cymbal.start(startTime);
  cymbal.stop(startTime + 0.86);
}

function createHallImpulse(context, seconds) {
  const length = Math.floor(context.sampleRate * seconds);
  const impulse = context.createBuffer(2, length, context.sampleRate);

  for (let channel = 0; channel < impulse.numberOfChannels; channel += 1) {
    const data = impulse.getChannelData(channel);
    for (let index = 0; index < length; index += 1) {
      const decay = Math.pow(1 - index / length, 2.2);
      data[index] = (Math.random() * 2 - 1) * decay * 0.5;
    }
  }

  return impulse;
}

function ramp(context, param, value, seconds) {
  param.cancelScheduledValues(context.currentTime);
  param.linearRampToValueAtTime(value, context.currentTime + seconds);
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const cross = ab.x * cb.y - ab.y * cb.x;
  return Math.abs((Math.atan2(cross, dot) * 180) / Math.PI);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

window.addEventListener("beforeunload", () => {
  if (animationFrame) cancelAnimationFrame(animationFrame);
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach((track) => track.stop());
});
