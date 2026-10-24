/* =============================================================================
   WebGL Drone Simulator  -  Three.js r128 (classic script, global THREE)

   World frame (simulation):  X = East,  Y = Up,  Z = North
   Scene mapping (Three.js):  (x, y, -z)   -> right-handed, North = -Z on screen
   Heading: 0 deg = North, positive = clockwise (seen from above)
   Command vector: [v_x, v_y, v_z, yaw_rate, action]
       v_x forward m/s, v_y right m/s (body frame), v_z up m/s, yaw_rate deg/s,
       action 0 HOVER | 1 TAKEOFF (2 m) | 2 LAND | 3 RESET
   ============================================================================= */
(function () {
  'use strict';

  // ---------------------------------------------------------------- config ---
  const CONFIG = {
    wsPort: 8765,
    cmdTimeout: 0.5,        // s without a vector -> taper to hover
    takeoffAlt: 2.0,        // m
    groundSize: 50,         // m (50 x 50)
    geofence: 24.5,         // m, half-extent the drone may travel
    maxAlt: 40,             // m
    maxHSpeed: 8,           // m/s
    maxVSpeed: 4,           // m/s
    maxYawRate: 180,        // deg/s
    velTau: 0.25,           // s, velocity response time constant
    yawTau: 0.20,           // s
    stopTau: 0.15,          // s, faster response while tapering after a timeout
    tiltTau: 0.12,          // s
    maxTilt: 22,            // deg of visual pitch/roll at tiltRefSpeed
    tiltRefSpeed: 5,        // m/s
    groundOffset: 0.14,     // m, body origin height above the ground when landed
    manualSpeed: 4,         // m/s
    manualClimb: 2.5,       // m/s
    manualYaw: 90,          // deg/s
    followDist: 4.5,
    followHeight: 1.9
  };

  const DEG = Math.PI / 180;
  const $ = (id) => document.getElementById(id);
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const nowSec = () => performance.now() / 1000;
  const smoothing = (dt, tau) => 1 - Math.exp(-dt / tau);

  function fatal(message) {
    const loading = $('loading');
    if (loading) { loading.hidden = true; }
    $('fatal-msg').textContent = message;
    $('fatal').hidden = false;
  }

  if (!window.THREE || !THREE.OrbitControls) {
    fatal('Three.js or OrbitControls failed to load.');
    return;
  }

  // -------------------------------------------------------------- renderer ---
  const viewport = $('viewport');
  let renderer = null;
  const rendererAttempts = [
    { antialias: true, powerPreference: 'high-performance' },
    { antialias: false, powerPreference: 'default' }
  ];
  for (let i = 0; i < rendererAttempts.length && !renderer; i++) {
    try { renderer = new THREE.WebGLRenderer(rendererAttempts[i]); } catch (err) { renderer = null; }
  }
  if (!renderer) {
    fatal('WebGL is not available in this browser. Enable hardware acceleration or try Chrome/Safari/Firefox.');
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x1a1d24, 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  viewport.appendChild(renderer.domElement);

  renderer.domElement.addEventListener('webglcontextlost', function (e) {
    e.preventDefault();
    $('ctx-lost').hidden = false;
  });
  renderer.domElement.addEventListener('webglcontextrestored', function () {
    $('ctx-lost').hidden = true;
  });

  // ----------------------------------------------------------- scene/camera ---
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1d24);
  scene.fog = new THREE.Fog(0x1a1d24, 45, 140);

  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 400);
  camera.position.set(6, 4.5, 8);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.5;
  controls.maxDistance = 70;
  controls.maxPolarAngle = Math.PI / 2 - 0.03;   // never go below the ground
  controls.target.set(0, 0.5, 0);
  controls.update();

  // --------------------------------------------------------------- lighting ---
  const hemi = new THREE.HemisphereLight(0xa9c1e6, 0x20242d, 0.75);
  scene.add(hemi);

  const sun = new THREE.DirectionalLight(0xffffff, 0.95);
  sun.position.set(14, 26, 10);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -32;
  sun.shadow.camera.right = 32;
  sun.shadow.camera.top = 32;
  sun.shadow.camera.bottom = -32;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 90;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.02;
  scene.add(sun);

  // ------------------------------------------------------------ ground/grid ---
  function makeCheckerTexture() {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const g = c.getContext('2d');
    g.fillStyle = '#2b303b'; g.fillRect(0, 0, 256, 256);
    g.fillStyle = '#242832'; g.fillRect(0, 0, 128, 128); g.fillRect(128, 128, 128, 128);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(CONFIG.groundSize / 2, CONFIG.groundSize / 2);   // 1 m cells
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return tex;
  }

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(CONFIG.groundSize, CONFIG.groundSize),
    new THREE.MeshStandardMaterial({ map: makeCheckerTexture(), roughness: 0.95, metalness: 0.0 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const grid = new THREE.GridHelper(CONFIG.groundSize, CONFIG.groundSize / 5, 0x5b6b85, 0x3a4252);
  grid.position.y = 0.012;
  scene.add(grid);

  // Boundary frame
  (function () {
    const h = CONFIG.groundSize / 2;
    const pts = [
      new THREE.Vector3(-h, 0.02, -h), new THREE.Vector3(h, 0.02, -h),
      new THREE.Vector3(h, 0.02, h), new THREE.Vector3(-h, 0.02, h)
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    scene.add(new THREE.LineLoop(geo, new THREE.LineBasicMaterial({ color: 0x19d3e6 })));
  })();

  // ------------------------------------------------------------------ labels ---
  function makeLabel(text, color, scale) {
    const c = document.createElement('canvas');
    c.width = 256; c.height = 64;
    const g = c.getContext('2d');
    g.font = 'bold 34px -apple-system, Helvetica, Arial, sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillStyle = color;
    g.fillText(text, 128, 34);
    const tex = new THREE.CanvasTexture(c);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    sprite.scale.set(4 * scale, 1 * scale, 1);
    return sprite;
  }

  // ------------------------------------------------------------ axes (origin) ---
  // Sim X (East) -> scene +X, sim Y (Up) -> scene +Y, sim Z (North) -> scene -Z
  (function () {
    const origin = new THREE.Vector3(0, 0.03, 0);
    const defs = [
      { dir: new THREE.Vector3(1, 0, 0), color: 0xff5468, css: '#ff5468', text: 'X (East)' },
      { dir: new THREE.Vector3(0, 1, 0), color: 0x3ddc84, css: '#3ddc84', text: 'Y (Up)' },
      { dir: new THREE.Vector3(0, 0, -1), color: 0x4c9aff, css: '#4c9aff', text: 'Z (North)' }
    ];
    defs.forEach(function (d) {
      scene.add(new THREE.ArrowHelper(d.dir, origin, 3, d.color, 0.45, 0.22));
      const label = makeLabel(d.text, d.css, 0.7);
      label.position.copy(origin).addScaledVector(d.dir, 3.7);
      label.position.y = Math.max(label.position.y, 0.6);   // keep sprite fully above the ground plane
      scene.add(label);
    });
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.08, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    dot.position.copy(origin);
    scene.add(dot);

    // Compass letters on the edge of the field
    const e = CONFIG.groundSize / 2 + 1.6;
    [['N', 0, -e, '#ff5468'], ['S', 0, e, '#7d8aa0'], ['E', e, 0, '#7d8aa0'], ['W', -e, 0, '#7d8aa0']].forEach(function (c) {
      const s = makeLabel(c[0], c[3], 1.1);
      s.position.set(c[1], 0.8, c[2]);
      scene.add(s);
    });
  })();

  // ------------------------------------------------------------ drone model ---
  // Hierarchy:  droneRoot (position + yaw)  ->  tiltGroup (visual pitch/roll)  ->  parts
  // Drone nose points to local -Z (= North at yaw 0).
  const droneRoot = new THREE.Group();
  const tiltGroup = new THREE.Group();
  droneRoot.add(tiltGroup);
  scene.add(droneRoot);

  const MAT = {
    chassis: new THREE.MeshStandardMaterial({ color: 0x2a2f3a, metalness: 0.75, roughness: 0.35 }),
    top: new THREE.MeshStandardMaterial({ color: 0x19d3e6, metalness: 0.4, roughness: 0.35, emissive: 0x053c42 }),
    carbon: new THREE.MeshStandardMaterial({ color: 0x16181d, metalness: 0.3, roughness: 0.6 }),
    motor: new THREE.MeshStandardMaterial({ color: 0x3b4150, metalness: 0.85, roughness: 0.3 }),
    frontHub: new THREE.MeshStandardMaterial({ color: 0xff7a1a, metalness: 0.3, roughness: 0.4, emissive: 0x4a2000 }),
    rearHub: new THREE.MeshStandardMaterial({ color: 0xdfe6f2, metalness: 0.3, roughness: 0.4 }),
    nose: new THREE.MeshStandardMaterial({ color: 0xff3d3d, metalness: 0.2, roughness: 0.4, emissive: 0x5a0a0a }),
    blade: new THREE.MeshStandardMaterial({ color: 0x0d0f13, metalness: 0.1, roughness: 0.6 }),
    disc: new THREE.MeshBasicMaterial({
      color: 0x9fe9ff, transparent: true, opacity: 0.2, side: THREE.DoubleSide, depthWrite: false
    }),
    lens: new THREE.MeshStandardMaterial({ color: 0x0a2a30, metalness: 0.9, roughness: 0.1, emissive: 0x0b6c78 })
  };

  function addMesh(parent, geometry, material, x, y, z, cast) {
    const m = new THREE.Mesh(geometry, material);
    m.position.set(x, y, z);
    m.castShadow = cast !== false;
    m.receiveShadow = false;
    parent.add(m);
    return m;
  }

  // Chassis: dark body, cyan top plate, battery pack underneath
  addMesh(tiltGroup, new THREE.BoxGeometry(0.30, 0.09, 0.46), MAT.chassis, 0, 0, 0);
  addMesh(tiltGroup, new THREE.BoxGeometry(0.26, 0.02, 0.38), MAT.top, 0, 0.055, 0.01);
  addMesh(tiltGroup, new THREE.BoxGeometry(0.20, 0.05, 0.28), MAT.carbon, 0, -0.07, 0.02);

  // Colored nose: red cone pointing forward (-Z) + orange front bar
  const nose = addMesh(tiltGroup, new THREE.ConeGeometry(0.06, 0.16, 20), MAT.nose, 0, 0.005, -0.31);
  nose.rotation.x = -Math.PI / 2;
  addMesh(tiltGroup, new THREE.BoxGeometry(0.22, 0.02, 0.03), MAT.frontHub, 0, 0.065, -0.2);

  // Camera gimbal
  addMesh(tiltGroup, new THREE.SphereGeometry(0.045, 16, 12), MAT.chassis, 0, -0.065, -0.2);
  addMesh(tiltGroup, new THREE.SphereGeometry(0.022, 12, 10), MAT.lens, 0, -0.07, -0.235, false);

  // Landing skids
  [-0.13, 0.13].forEach(function (sx) {
    addMesh(tiltGroup, new THREE.BoxGeometry(0.022, 0.022, 0.42), MAT.carbon, sx, -0.125, 0);
    [-0.14, 0.14].forEach(function (sz) {
      addMesh(tiltGroup, new THREE.BoxGeometry(0.018, 0.09, 0.018), MAT.carbon, sx, -0.085, sz);
    });
  });

  // X-configuration: two crossed carbon arms (motor-to-motor 1.1 m) + 4 motors/rotors
  [Math.PI / 4, -Math.PI / 4].forEach(function (ry) {
    const arm = addMesh(tiltGroup, new THREE.BoxGeometry(1.1, 0.03, 0.05), MAT.carbon, 0, 0.0, 0);
    arm.rotation.y = ry;
  });

  const MOTOR_OFFSET = 0.389;
  const rotors = [];
  [
    { x: -MOTOR_OFFSET, z: -MOTOR_OFFSET, front: true, dir: 1 },    // front-left
    { x: MOTOR_OFFSET, z: -MOTOR_OFFSET, front: true, dir: -1 },    // front-right
    { x: -MOTOR_OFFSET, z: MOTOR_OFFSET, front: false, dir: -1 },   // rear-left
    { x: MOTOR_OFFSET, z: MOTOR_OFFSET, front: false, dir: 1 }      // rear-right
  ].forEach(function (m) {
    addMesh(tiltGroup, new THREE.CylinderGeometry(0.055, 0.065, 0.07, 20), MAT.motor, m.x, 0.04, m.z);

    const spinner = new THREE.Group();
    spinner.position.set(m.x, 0.092, m.z);
    tiltGroup.add(spinner);

    addMesh(spinner, new THREE.CylinderGeometry(0.02, 0.02, 0.025, 12), m.front ? MAT.frontHub : MAT.rearHub, 0, 0.005, 0, false);
    const blade = addMesh(spinner, new THREE.BoxGeometry(0.40, 0.006, 0.032), MAT.blade, 0, 0, 0, false);
    blade.castShadow = true;
    addMesh(spinner, new THREE.CylinderGeometry(0.205, 0.205, 0.004, 40), MAT.disc, 0, 0.004, 0, false);

    rotors.push({ node: spinner, dir: m.dir });
  });

  // Ground marker: ring under the drone + vertical altitude line
  const groundMarker = new THREE.Mesh(
    new THREE.RingGeometry(0.28, 0.34, 40),
    new THREE.MeshBasicMaterial({ color: 0x19d3e6, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false })
  );
  groundMarker.rotation.x = -Math.PI / 2;
  scene.add(groundMarker);

  const altLineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0)]);
  const altLine = new THREE.Line(altLineGeo, new THREE.LineBasicMaterial({ color: 0x19d3e6, transparent: true, opacity: 0.35 }));
  scene.add(altLine);

  // ------------------------------------------------------- simulation state ---
  const ACTION = { HOVER: 0, TAKEOFF: 1, LAND: 2, RESET: 3 };

  const state = {
    x: 0, y: CONFIG.groundOffset, z: 0,   // East, Up, North (m)
    heading: 0,                           // deg, 0 = North, clockwise positive
    bvx: 0, bvy: 0, bvz: 0,               // smoothed body-frame velocity (fwd, right, up) m/s
    wx: 0, wz: 0,                         // world velocity East / North m/s
    yawRate: 0,                           // deg/s
    pitch: 0, roll: 0,                    // visual tilt, deg (+pitch = nose down, +roll = right side down)
    rotorSpeed: 0,                        // rad/s
    mode: 'LANDED'                        // LANDED | TAKEOFF | FLIGHT | LANDING
  };

  // Latest command from WebSocket
  const cmd = { vx: 0, vy: 0, vz: 0, yaw: 0, action: 0, receivedAt: -Infinity, raw: null };
  let lastRemoteAction = ACTION.HOVER;

  // Manual override
  const manual = { enabled: false, keys: Object.create(null) };

  function resetSimulation() {
    state.x = 0; state.y = CONFIG.groundOffset; state.z = 0;
    state.heading = 0;
    state.bvx = state.bvy = state.bvz = 0;
    state.wx = state.wz = 0;
    state.yawRate = 0;
    state.pitch = state.roll = 0;
    state.mode = 'LANDED';
    cmd.vx = cmd.vy = cmd.vz = cmd.yaw = 0;
  }

  /** Scene-space height of the body origin when hovering at takeoffAlt metres above the ground. */
  function takeoffY() { return CONFIG.takeoffAlt + CONFIG.groundOffset; }

  function triggerAction(action) {
    if (action === ACTION.RESET) {
      resetSimulation();
    } else if (action === ACTION.TAKEOFF) {
      if (state.mode !== 'TAKEOFF' && state.y < takeoffY() - 0.03) { state.mode = 'TAKEOFF'; }
    } else if (action === ACTION.LAND) {
      if (state.mode !== 'LANDED') { state.mode = 'LANDING'; }
    }
    // ACTION.HOVER never cancels a latched takeoff/landing; it is plain velocity control.
  }

  /** Accepts [v_x, v_y, v_z, yaw_rate, action]; returns true when valid. */
  function handleVector(v) {
    if (!Array.isArray(v) || v.length < 5) { return false; }
    const n = [];
    for (let i = 0; i < 5; i++) {
      const value = Number(v[i]);
      if (!Number.isFinite(value)) { return false; }
      n.push(value);
    }
    const t = nowSec();
    cmd.vx = clamp(n[0], -CONFIG.maxHSpeed, CONFIG.maxHSpeed);
    cmd.vy = clamp(n[1], -CONFIG.maxHSpeed, CONFIG.maxHSpeed);
    cmd.vz = clamp(n[2], -CONFIG.maxVSpeed, CONFIG.maxVSpeed);
    cmd.yaw = clamp(n[3], -CONFIG.maxYawRate, CONFIG.maxYawRate);
    cmd.action = Math.round(n[4]);
    cmd.raw = n;

    // Actions are edge-triggered. After a silent gap the same action counts as new again.
    if (t - cmd.receivedAt > CONFIG.cmdTimeout) { lastRemoteAction = ACTION.HOVER; }
    cmd.receivedAt = t;
    if (cmd.action !== lastRemoteAction) {
      lastRemoteAction = cmd.action;
      if (!manual.enabled) { triggerAction(cmd.action); }
    }
    return true;
  }

  function statusLabel() {
    if (state.mode === 'LANDED') { return 'LANDED'; }
    if (state.mode === 'TAKEOFF') { return 'TAKEOFF'; }
    if (state.mode === 'LANDING') { return 'LANDING'; }
    const speed = Math.sqrt(state.bvx * state.bvx + state.bvy * state.bvy + state.bvz * state.bvz);
    return (speed < 0.1 && Math.abs(state.yawRate) < 2) ? 'HOVER' : 'FLYING';
  }

  /** Advance the simulation by dt seconds (frame-rate independent). */
  function stepSimulation(dt, t) {
    const ground = CONFIG.groundOffset;

    // 1. Commanded body-frame velocity -------------------------------------------------
    let tvx = 0, tvy = 0, tvz = 0, tyaw = 0;
    let velTau = CONFIG.velTau;
    let yawTau = CONFIG.yawTau;

    if (manual.enabled) {
      const k = manual.keys;
      tvx = ((k.KeyW ? 1 : 0) - (k.KeyS ? 1 : 0)) * CONFIG.manualSpeed;
      tvy = ((k.KeyD ? 1 : 0) - (k.KeyA ? 1 : 0)) * CONFIG.manualSpeed;
      tvz = ((k.Space ? 1 : 0) - ((k.ShiftLeft || k.ShiftRight) ? 1 : 0)) * CONFIG.manualClimb;
      tyaw = ((k.KeyE ? 1 : 0) - (k.KeyQ ? 1 : 0)) * CONFIG.manualYaw;
    } else if (t - cmd.receivedAt <= CONFIG.cmdTimeout) {
      tvx = cmd.vx; tvy = cmd.vy; tvz = cmd.vz; tyaw = cmd.yaw;
    } else {
      // SAFETY TIMEOUT: no fresh vector within 0.5 s -> taper everything to zero (hover).
      velTau = CONFIG.stopTau;
      yawTau = CONFIG.stopTau;
    }

    // 2. Mode logic (takeoff / landing autopilots, liftoff, pilot override) -------------
    if (state.mode === 'LANDED' && tvz > 0.05) {
      state.mode = 'FLIGHT';                    // climb command lifts the drone off
    }
    if ((state.mode === 'TAKEOFF' || state.mode === 'LANDING') && Math.abs(tvz) > 0.05) {
      state.mode = 'FLIGHT';                    // explicit vertical command overrides autopilot
    }
    if (state.mode === 'TAKEOFF') {
      const err = takeoffY() - state.y;
      if (err <= 0.03) {
        state.y = takeoffY();
        state.bvz = 0;
        state.mode = 'FLIGHT';
        tvz = 0;
      } else {
        tvz = clamp(err * 1.5, 0.12, 1.5);
      }
    } else if (state.mode === 'LANDING') {
      tvz = -clamp(0.3 + 0.6 * (state.y - ground), 0.3, 1.2);
    }

    if (state.mode === 'LANDED') {
      // Motors idle on the ground: no translation or yaw.
      state.bvx = state.bvy = state.bvz = 0;
      state.wx = state.wz = 0;
      state.yawRate = 0;
      state.y = ground;
    } else {
      // 3. First-order velocity response (exponential smoothing, dt-independent) ---------
      const a = smoothing(dt, velTau);
      state.bvx += (tvx - state.bvx) * a;
      state.bvy += (tvy - state.bvy) * a;
      state.bvz += (tvz - state.bvz) * a;
      state.yawRate += (tyaw - state.yawRate) * smoothing(dt, yawTau);

      // 4. Integrate: body frame -> world (E, N) using the heading ------------------------
      //    forward = (sin h, cos h), right = (cos h, -sin h)   in (East, North)
      const psi = state.heading * DEG;
      const s = Math.sin(psi), c = Math.cos(psi);
      state.wx = state.bvx * s + state.bvy * c;
      state.wz = state.bvx * c - state.bvy * s;

      state.heading = (state.heading + state.yawRate * dt) % 360;
      if (state.heading < 0) { state.heading += 360; }
      state.x += state.wx * dt;
      state.z += state.wz * dt;
      state.y += state.bvz * dt;

      // 5. Constraints: geofence, ceiling, ground contact --------------------------------
      state.x = clamp(state.x, -CONFIG.geofence, CONFIG.geofence);
      state.z = clamp(state.z, -CONFIG.geofence, CONFIG.geofence);
      if (state.y > CONFIG.maxAlt) { state.y = CONFIG.maxAlt; state.bvz = Math.min(state.bvz, 0); }
      if (state.y <= ground && state.bvz <= 0 && state.mode !== 'TAKEOFF') {
        state.y = ground;
        state.mode = 'LANDED';
        state.bvx = state.bvy = state.bvz = 0;
        state.wx = state.wz = 0;
        state.yawRate = 0;
      }
    }

    // 6. Visual pitch / roll from body velocity (smoothed) -------------------------------
    const flying = state.mode !== 'LANDED';
    const tPitch = flying ? clamp(state.bvx / CONFIG.tiltRefSpeed, -1.3, 1.3) * CONFIG.maxTilt : 0;
    const tRoll = flying ? clamp(state.bvy / CONFIG.tiltRefSpeed, -1.3, 1.3) * CONFIG.maxTilt : 0;
    const ta = smoothing(dt, CONFIG.tiltTau);
    state.pitch += (tPitch - state.pitch) * ta;
    state.roll += (tRoll - state.roll) * ta;

    // 7. Rotor spin ---------------------------------------------------------------------
    const spinTarget = flying ? 40 + 4 * Math.sqrt(state.wx * state.wx + state.wz * state.wz) : 0;
    state.rotorSpeed += (spinTarget - state.rotorSpeed) * smoothing(dt, 0.45);
    for (let i = 0; i < rotors.length; i++) {
      rotors[i].node.rotation.y += rotors[i].dir * state.rotorSpeed * dt;
    }
  }

  /** Push simulation state into the Three.js scene graph. */
  function syncScene() {
    // Sim (E, U, N) -> scene (x, y, -z). Heading is clockwise-from-north => rotation.y = -heading.
    droneRoot.position.set(state.x, state.y, -state.z);
    droneRoot.rotation.y = -state.heading * DEG;
    // Nose-down (forward motion) = negative rotation about X; right-side-down = negative about Z.
    tiltGroup.rotation.x = -state.pitch * DEG;
    tiltGroup.rotation.z = -state.roll * DEG;

    groundMarker.position.set(state.x, 0.03, -state.z);
    const markerScale = 1 + Math.min(state.y, 10) * 0.08;
    groundMarker.scale.set(markerScale, markerScale, markerScale);
    groundMarker.material.opacity = clamp(0.6 - state.y * 0.05, 0.2, 0.6);
    altLine.position.set(state.x, 0, -state.z);
    altLine.scale.y = Math.max(state.y, 0.001);
  }

  // ----------------------------------------------------------- camera modes ---
  let cameraMode = 'ORBIT';              // ORBIT | FOLLOW
  const followPos = new THREE.Vector3();
  const followLook = new THREE.Vector3();
  const followInit = { done: false };
  const tmpDesired = new THREE.Vector3();
  const tmpTarget = new THREE.Vector3();

  function followDesired(outPos, outLook) {
    const h = state.heading * DEG;
    // Behind the drone = opposite of the forward vector (sin h, cos h) in (E, N); scene z = -N.
    outPos.set(
      state.x - Math.sin(h) * CONFIG.followDist,
      state.y + CONFIG.followHeight,
      -(state.z - Math.cos(h) * CONFIG.followDist)
    );
    outLook.set(state.x, state.y + 0.1, -state.z);
    outPos.y = Math.max(outPos.y, 0.4);          // never let the chase camera go underground
  }

  function setCameraMode(mode) {
    cameraMode = mode;
    $('camera-label').textContent = mode;
    if (mode === 'FOLLOW') {
      controls.enabled = false;
      followInit.done = false;
    } else {
      controls.enabled = true;
      controls.target.set(state.x, state.y, -state.z);
      controls.update();
    }
  }

  function updateCamera(dt) {
    if (cameraMode === 'FOLLOW') {
      followDesired(tmpDesired, tmpTarget);
      if (!followInit.done) {
        followPos.copy(camera.position);
        followLook.copy(tmpTarget);
        followInit.done = true;
      }
      followPos.lerp(tmpDesired, smoothing(dt, 0.25));
      followLook.lerp(tmpTarget, smoothing(dt, 0.15));
      camera.position.copy(followPos);
      camera.lookAt(followLook);
    } else {
      controls.update();
    }
  }

  // ------------------------------------------------------------- WebSocket ---
  const WS_URL = 'ws://' + (window.location.hostname || 'localhost') + ':' + CONFIG.wsPort;
  let ws = null;
  let wsConnected = false;
  let reconnectDelay = 500;
  let lastTelemetrySent = 0;

  function setWsStatus(connected) {
    wsConnected = connected;
    $('ws-pill').className = 'pill ' + (connected ? 'on' : 'off');
    $('ws-text').textContent = connected ? 'CONNECTED' : 'DISCONNECTED';
  }

  function connectWs() {
    try {
      ws = new WebSocket(WS_URL);
    } catch (err) {
      setWsStatus(false);
      setTimeout(connectWs, reconnectDelay);
      return;
    }
    ws.onopen = function () {
      reconnectDelay = 500;
      setWsStatus(true);
    };
    ws.onmessage = function (ev) {
      if (typeof ev.data !== 'string') { return; }
      let data;
      try { data = JSON.parse(ev.data); } catch (err) { return; }
      if (data && !Array.isArray(data) && typeof data === 'object') { data = data.cmd || data.vector; }
      handleVector(data);
    };
    ws.onclose = function () {
      setWsStatus(false);
      reconnectDelay = Math.min(reconnectDelay * 1.5, 3000);
      setTimeout(connectWs, reconnectDelay);
    };
    ws.onerror = function () { try { ws.close(); } catch (e) { /* ignore */ } };
  }

  function sendTelemetry(t) {
    if (!wsConnected || !ws || ws.readyState !== WebSocket.OPEN || t - lastTelemetrySent < 0.1) { return; }
    lastTelemetrySent = t;
    ws.send(JSON.stringify({
      telemetry: {
        x: +state.x.toFixed(3), y: +state.y.toFixed(3), z: +state.z.toFixed(3),
        heading: +state.heading.toFixed(2), status: statusLabel()
      }
    }));
  }

  // -------------------------------------------------------- keyboard / UI ---
  const HANDLED_KEYS = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'Space', 'ShiftLeft', 'ShiftRight'];

  function setManual(on) {
    manual.enabled = on;
    manual.keys = Object.create(null);
    $('btn-manual').setAttribute('aria-pressed', on ? 'true' : 'false');
    $('manual-label').textContent = on ? 'ON' : 'OFF';
    if (!on) { cmd.receivedAt = -Infinity; lastRemoteAction = ACTION.HOVER; }
  }

  window.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) { return; }
    if (e.code === 'KeyC' && !e.repeat) { setCameraMode(cameraMode === 'ORBIT' ? 'FOLLOW' : 'ORBIT'); return; }
    if (e.code === 'KeyM' && !e.repeat) { setManual(!manual.enabled); return; }
    if (e.code === 'KeyT' && !e.repeat) { triggerAction(ACTION.TAKEOFF); return; }
    if (e.code === 'KeyL' && !e.repeat) { triggerAction(ACTION.LAND); return; }
    if (e.code === 'KeyR' && !e.repeat) { triggerAction(ACTION.RESET); return; }
    if (manual.enabled && HANDLED_KEYS.indexOf(e.code) !== -1) {
      manual.keys[e.code] = true;
      e.preventDefault();                 // stop Space from scrolling / activating buttons
    }
  });
  window.addEventListener('keyup', function (e) {
    if (HANDLED_KEYS.indexOf(e.code) !== -1) { manual.keys[e.code] = false; }
  });
  window.addEventListener('blur', function () { manual.keys = Object.create(null); });

  function bindButton(id, fn) {
    const el = $(id);
    el.addEventListener('click', function () { fn(); el.blur(); });
  }
  bindButton('btn-camera', function () { setCameraMode(cameraMode === 'ORBIT' ? 'FOLLOW' : 'ORBIT'); });
  bindButton('btn-manual', function () { setManual(!manual.enabled); });
  bindButton('btn-takeoff', function () { triggerAction(ACTION.TAKEOFF); });
  bindButton('btn-land', function () { triggerAction(ACTION.LAND); });
  bindButton('btn-reset', function () { triggerAction(ACTION.RESET); });

  window.addEventListener('resize', function () {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // ------------------------------------------------------------------- HUD ---
  const hud = {
    x: $('t-x'), y: $('t-y'), z: $('t-z'),
    heading: $('t-heading'), cardinal: $('t-cardinal'), rose: $('compass-rose'),
    action: $('t-action'), speed: $('t-speed'), vspeed: $('t-vspeed'), yaw: $('t-yaw'),
    mode: $('t-mode'), vector: $('t-vector'), age: $('t-age')
  };
  const CARDINALS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  let lastHudUpdate = 0;

  function updateHud(t) {
    if (t - lastHudUpdate < 0.05) { return; }       // ~20 Hz is plenty for text
    lastHudUpdate = t;

    hud.x.textContent = state.x.toFixed(2);
    hud.y.textContent = Math.max(0, state.y - CONFIG.groundOffset).toFixed(2);   // altitude above ground
    hud.z.textContent = state.z.toFixed(2);
    hud.heading.textContent = state.heading.toFixed(1).padStart(5, '0');
    hud.cardinal.textContent = CARDINALS[Math.round(state.heading / 45) % 8];
    hud.rose.style.transform = 'rotate(' + (-state.heading).toFixed(1) + 'deg)';

    const label = statusLabel();
    hud.action.textContent = label;
    hud.action.setAttribute('data-state', label);

    hud.speed.textContent = Math.sqrt(state.wx * state.wx + state.wz * state.wz).toFixed(2) + ' m/s';
    hud.vspeed.textContent = state.bvz.toFixed(2) + ' m/s';
    hud.yaw.textContent = state.yawRate.toFixed(1) + ' \u00b0/s';

    hud.mode.textContent = manual.enabled ? 'MANUAL (KEYS)' : 'REMOTE (WS)';
    if (cmd.raw) {
      const age = t - cmd.receivedAt;
      hud.vector.textContent = '[' + cmd.raw.map(function (v, i) { return i === 4 ? v : v.toFixed(1); }).join(', ') + ']';
      hud.age.textContent = age < 60 ? age.toFixed(2) + ' s' + (age > CONFIG.cmdTimeout ? ' (timeout)' : '') : 'stale';
    } else {
      hud.vector.textContent = '\u2014';
      hud.age.textContent = '\u2014';
    }
  }

  // ---------------------------------------------------------------- main loop ---
  const MAX_FRAME = 0.1;                        // clamp dt after tab switches / stalls
  const FIXED_STEP = 1 / 120;                   // sub-step keeps integration stable at low fps
  let lastTime = nowSec();

  function frame() {
    requestAnimationFrame(frame);
    const t = nowSec();
    const frameDt = clamp(t - lastTime, 0.0001, MAX_FRAME);
    lastTime = t;

    let remaining = frameDt;
    while (remaining > 1e-6) {
      const h = Math.min(remaining, FIXED_STEP);
      stepSimulation(h, t);
      remaining -= h;
    }
    syncScene();
    updateCamera(frameDt);
    updateHud(t);
    sendTelemetry(t);
    renderer.render(scene, camera);
  }

  // -------------------------------------------------------------------- start ---
  resetSimulation();
  syncScene();
  setWsStatus(false);
  connectWs();
  $('loading').hidden = true;
  frame();

  // Debug hook for the browser console:  droneSim.send([1, 0, 0, 0, 0])
  window.droneSim = { state: state, send: handleVector, config: CONFIG };

})();
