import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/**
 * SOI Waveguide 3D Mesh — Rib/Strip + Taper + Bend + Tetra/Hex export
 * - Units: micrometers (µm)
 * - Geometry: BOX (SiO2) + Si device layer (strip or rib) + upper cladding (air/oxide)
 * - Features:
 *   • Strip or Rib (partial-etch) waveguide
 *   • Linear taper of widths (start→end)
 *   • Constant-radius in-plane bend by angle (derived R = L/θ)
 *   • Hexa grid export (structured) or tetra (5-tet per hex)
 *   • Element material tagging: core / box / cladding + PML flag
 */
export default function App() {
  // ---- Params (µm) ----
  const [p, setP] = useState({
    // Device & domain
    coreHeight: 0.22,     // Si device layer thickness
    boxThick: 2.0,        // SiO2 BOX thickness (below)
    cladThick: 2.0,       // upper cladding thickness (above)
    sideMargin: 2.0,      // lateral clearance added to domain

    // Waveguide mode
    wgType: "strip",      // "strip" | "rib"
    // For strip (full-etch): widthStart/End used, etchDepth = coreHeight
    widthStart: 0.45,
    widthEnd: 0.45,

    // For rib (partial-etch)
    baseWidthStart: 1.2,  // base slab width start
    baseWidthEnd: 1.2,    // base slab width end
    ribWidthStart: 0.5,   // ridge width start
    ribWidthEnd: 0.5,     // ridge width end
    etchDepth: 0.12,      // etch depth from top (0..coreHeight)

    // Length & bend
    coreLength: 10.0,     // arc length along centerline
    bendAngleDeg: 0.0,    // total bend angle θ (deg); if 0 → straight

    // Mesh/grid resolution (structured grid over bounding box)
    nx: 40,
    ny: 20,
    nz: 120,

    // Sweep visualization resolution
    sweepSegs: 200,

    // PML
    pml: 0.5,

    // Toggles
    showGrid: true,       // structured (volume) grid lines
    showFloor: true,      // floor grid helper (new)
    showAxes: false,      // XYZ axes helper (new)
    showCore: true,
    showSubstrate: true,
    showCladding: true,
    showPML: true,
    gridOpacity: 0.55,

    // Export
    meshType: "hex", // "hex" | "tet"
  });

  const hostRef = useRef(null);
  const threeRef = useRef({ renderer: null, scene: null, camera: null, controls: null, axesHelper: null, gridHelper: null });
  const groupsRef = useRef({ grid: null, solid: null, pml: null });
  const matRef = useRef({});

  // Helpers
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const linspace = (min, max, count) => {
    const arr = new Array(count);
    const step = (max - min) / (count - 1);
    for (let i = 0; i < count; i++) arr[i] = min + i * step;
    return arr;
  };

  // Derived functions for centerline (in-plane bend)
  const bend = useMemo(() => {
    const L = Math.max(1e-6, p.coreLength);
    const theta = (p.bendAngleDeg || 0) * Math.PI / 180; // total bend angle (rad)
    const straight = Math.abs(theta) < 1e-6;
    const R = straight ? Infinity : (L / Math.abs(theta)); // radius from arc length
    const sign = theta >= 0 ? 1 : -1; // +θ bends to +x

    // Centerline position (absolute, starting at s=0 at origin, heading +z)
    function centerline(s) {
      s = clamp(s, 0, L);
      if (straight) return new THREE.Vector3(0, 0, s);
      const phi = (s / R) * sign; // signed local angle
      const Cx = sign * R, Cz = 0;
      const x = Cx - R * Math.cos(phi);
      const z = R * Math.sin(phi);
      return new THREE.Vector3(x, 0, z);
    }

    function tangent(s) {
      if (straight) return new THREE.Vector3(0, 0, 1);
      s = clamp(s, 0, L);
      const phi = (s / R) * sign;
      // derivative wrt s: dx/ds = sin(phi)*sign, dz/ds = cos(phi)
      const dx = Math.sin(phi) * sign;
      const dz = Math.cos(phi);
      const v = new THREE.Vector3(dx, 0, dz);
      v.normalize();
      return v;
    }

    // Extents of the centerline in x and z
    const mid = centerline(L / 2);
    const end = centerline(L);
    const xSpan = straight ? 0 : Math.abs(sign * R - (sign * R - R * Math.cos(theta))) ; // R*(1-cosθ)
    const zSpan = straight ? L : Math.abs(R * Math.sin(theta));

    return { L, theta, R, sign, straight, centerline, tangent, mid, xSpan, zSpan };
  }, [p.coreLength, p.bendAngleDeg]);

  // Width profiles
  const widths = useMemo(() => {
    const widthStrip = (s) => lerp(p.widthStart, p.widthEnd, s / bend.L);
    const ribWidth = (s) => lerp(p.ribWidthStart, p.ribWidthEnd, s / bend.L);
    const baseWidth = (s) => lerp(p.baseWidthStart, p.baseWidthEnd, s / bend.L);
    const slabThick = clamp(p.coreHeight - p.etchDepth, 0, p.coreHeight);
    return { widthStrip, ribWidth, baseWidth, slabThick };
  }, [p.widthStart, p.widthEnd, p.ribWidthStart, p.ribWidthEnd, p.baseWidthStart, p.baseWidthEnd, p.etchDepth, p.coreHeight, bend.L]);

  // Domain extents (bounding box), centered at centerline mid-point to keep view nice
  const domain = useMemo(() => {
    const side = p.sideMargin;
    const maxBase = p.wgType === "strip" ? Math.max(p.widthStart, p.widthEnd) : Math.max(p.baseWidthStart, p.baseWidthEnd);
    const W = maxBase + 2 * side + 2 * bend.xSpan; // include bend lateral reach
    const H = p.boxThick + p.coreHeight + p.cladThick;
    const Lbox = Math.max(bend.L, bend.zSpan + 2 * side);
    // Center shift to place mid-point at origin
    const mid = bend.centerline(bend.L / 2);
    const centerShift = new THREE.Vector3(mid.x, 0, mid.z);
    return { W, H, L: Lbox, centerShift };
  }, [p.sideMargin, p.coreHeight, p.boxThick, p.cladThick, p.widthStart, p.widthEnd, p.baseWidthStart, p.baseWidthEnd, bend.L, bend.xSpan, bend.zSpan, p.wgType]);

  // ---------- Init THREE ----------
  useEffect(() => {
    const host = hostRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1020);

    const camera = new THREE.PerspectiveCamera(50, host.clientWidth / host.clientHeight, 0.01, 4000);
    camera.position.set(6, 5, 12);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const axesHelper = new THREE.AxesHelper(2.0);
    scene.add(axesHelper);
    const gridHelper = new THREE.GridHelper(200, 200, 0x3553a5, 0x1a2b63);
    gridHelper.position.y = -0.0005;
    scene.add(gridHelper);

    scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(8, 12, 6);
    scene.add(dir);

    const world = new THREE.Group();
    const gridGroup = new THREE.Group();
    const solidGroup = new THREE.Group();
    const pmlGroup = new THREE.Group();
    world.add(gridGroup, solidGroup, pmlGroup);
    scene.add(world);

    // Materials
    matRef.current.coreMat = new THREE.MeshPhysicalMaterial({ color: 0x5aa8ff, metalness: 0.0, roughness: 0.7, transmission: 0.75, thickness: 2.0, transparent: true, opacity: 0.9 });
    matRef.current.boxMat  = new THREE.MeshPhysicalMaterial({ color: 0x87ffda, metalness: 0.0, roughness: 0.9, transmission: 0.7, thickness: 2.0, transparent: true, opacity: 0.25 });
    matRef.current.cladMat = new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 0.0, roughness: 0.95, transmission: 0.85, thickness: 2.0, transparent: true, opacity: 0.12 });

    threeRef.current = { renderer, scene, camera, controls, axesHelper, gridHelper };
    groupsRef.current = { grid: gridGroup, solid: solidGroup, pml: pmlGroup };

    const fit = () => {
      const { W, H, L } = domain;
      const radius = Math.max(W, H, L) * 0.75;
      const fov = camera.fov * (Math.PI / 180);
      const dist = radius / Math.sin(fov / 2);
      camera.position.set(dist * 0.6, dist * 0.45, dist * 0.7);
      controls.target.set(0, 0, 0);
      controls.update();
    };
    fit();

    let raf;
    const loop = () => { raf = requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); };
    loop();

    const onResize = () => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(host);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      host.removeChild(renderer.domElement);
    };
  }, []);

  // Toggle helpers visibility (floor grid & axes)
  useEffect(() => {
    const { axesHelper, gridHelper } = threeRef.current;
    if (axesHelper) axesHelper.visible = !!p.showAxes;
    if (gridHelper) gridHelper.visible = !!p.showFloor;
  }, [p.showAxes, p.showFloor]);

  // ---------- Builders ----------
  const buildSolids = () => {
    const { solid } = groupsRef.current;
    if (!solid) return;
    const { W, H, L } = domain;
    solid.clear();

    const { coreMat, boxMat, cladMat } = matRef.current;

    // Substrate (BOX)
    if (p.showSubstrate) {
      const box = new THREE.Mesh(new THREE.BoxGeometry(W, p.boxThick, L), boxMat);
      box.position.set(0, -(p.coreHeight / 2 + p.boxThick / 2), 0);
      solid.add(box);
    }

    // Cladding
    if (p.showCladding) {
      const clad = new THREE.Mesh(new THREE.BoxGeometry(W, p.cladThick, L), cladMat);
      clad.position.set(0, +(p.coreHeight / 2 + p.cladThick / 2), 0);
      solid.add(clad);
    }

    if (!p.showCore) return;

    // Sweep-based core (strip or rib) using instanced boxes
    const segs = clamp(Math.floor(p.sweepSegs), 8, 2000);
    const ds = bend.L / segs;

    // Ridge (full height)
    const geoRidge = new THREE.BoxGeometry(1, 1, 1);
    const ridge = new THREE.InstancedMesh(geoRidge, coreMat, segs);
    ridge.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    solid.add(ridge);

    // Slab (partial-etch)
    const slabTh = widths.slabThick;
    let slab = null;
    if (p.wgType === "rib" && slabTh > 1e-6) {
      const geoSlab = new THREE.BoxGeometry(1, 1, 1);
      slab = new THREE.InstancedMesh(geoSlab, coreMat, segs);
      slab.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      solid.add(slab);
    }

    const dummy = new THREE.Object3D();

    for (let i = 0; i < segs; i++) {
      const s0 = i * ds;
      const s1 = (i + 1) * ds;
      const sm = 0.5 * (s0 + s1);
      const len = s1 - s0;

      const pos = bend.centerline(sm).sub(domain.centerShift);
      const tan = bend.tangent(sm);
      const yaw = Math.atan2(tan.x, tan.z); // rotation around Y to align with tangent

      // Ridge width
      const wR = (p.wgType === "strip") ? widths.widthStrip(sm) : widths.ribWidth(sm);

      // Ridge transform (scale.x=width, scale.y=coreHeight, scale.z=len)
      dummy.position.set(pos.x, 0, pos.z);
      dummy.rotation.set(0, yaw, 0);
      dummy.scale.set(Math.max(1e-4, wR), p.coreHeight, Math.max(1e-4, len));
      dummy.updateMatrix();
      ridge.setMatrixAt(i, dummy.matrix);

      // Slab under ridge for rib
      if (slab) {
        const wB = widths.baseWidth(sm);
        const yCenter = -p.coreHeight / 2 + slabTh / 2;
        dummy.position.set(pos.x, yCenter, pos.z);
        dummy.rotation.set(0, yaw, 0);
        dummy.scale.set(Math.max(1e-4, wB), Math.max(1e-4, slabTh), Math.max(1e-4, len));
        dummy.updateMatrix();
        slab.setMatrixAt(i, dummy.matrix);
      }
    }
    ridge.instanceMatrix.needsUpdate = true;
    if (slab) slab.instanceMatrix.needsUpdate = true;
  };

  const buildGrid = () => {
    const { grid } = groupsRef.current;
    if (!grid) return;
    grid.clear();
    if (!p.showGrid) return;

    const { W, H, L } = domain;
    const nx = clamp(Math.floor(p.nx), 2, 256);
    const ny = clamp(Math.floor(p.ny), 2, 256);
    const nz = clamp(Math.floor(p.nz), 2, 1024);

    const xs = linspace(-W / 2, W / 2, nx + 1);
    const ys = linspace(-(H / 2), +(H / 2), ny + 1);
    const zs = linspace(-L / 2, L / 2, nz + 1);

    const nLinesX = (ny + 1) * (nz + 1);
    const nLinesY = (nx + 1) * (nz + 1);
    const nLinesZ = (nx + 1) * (ny + 1);
    const totalLines = nLinesX + nLinesY + nLinesZ;
    const positions = new Float32Array(totalLines * 2 * 3);
    let ptr = 0;

    for (let j = 0; j <= ny; j++) {
      for (let k = 0; k <= nz; k++) {
        positions[ptr++] = xs[0]; positions[ptr++] = ys[j]; positions[ptr++] = zs[k];
        positions[ptr++] = xs[nx]; positions[ptr++] = ys[j]; positions[ptr++] = zs[k];
      }
    }
    for (let i = 0; i <= nx; i++) {
      for (let k = 0; k <= nz; k++) {
        positions[ptr++] = xs[i]; positions[ptr++] = ys[0]; positions[ptr++] = zs[k];
        positions[ptr++] = xs[i]; positions[ptr++] = ys[ny]; positions[ptr++] = zs[k];
      }
    }
    for (let i = 0; i <= nx; i++) {
      for (let j = 0; j <= ny; j++) {
        positions[ptr++] = xs[i]; positions[ptr++] = ys[j]; positions[ptr++] = zs[0];
        positions[ptr++] = xs[i]; positions[ptr++] = ys[j]; positions[ptr++] = zs[nz];
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x3e6ce3, transparent: true, opacity: p.gridOpacity });
    const lines = new THREE.LineSegments(geom, mat);
    grid.add(lines);
  };

  const buildPML = () => {
    const { pml } = groupsRef.current;
    if (!pml) return;
    pml.clear();
    if (!p.showPML) return;

    const { W, H, L } = domain;
    const t = Math.max(0, p.pml);
    if (t <= 0) return;

    const pmlMat = new THREE.MeshBasicMaterial({ color: 0xff8800, transparent: true, opacity: 0.1, depthWrite: false });
    const pmlEdgeMat = new THREE.LineDashedMaterial({ color: 0xffaa55, dashSize: 0.1, gapSize: 0.06 });

    const shells = [
      { size: [t, H, L], pos: [ +W/2 - t/2, 0, 0] },
      { size: [t, H, L], pos: [ -W/2 + t/2, 0, 0] },
      { size: [W, t, L], pos: [0, +H/2 - t/2, 0] },
      { size: [W, t, L], pos: [0, -H/2 + t/2, 0] },
      { size: [W, H, t], pos: [0, 0, +L/2 - t/2] },
      { size: [W, H, t], pos: [0, 0, -L/2 + t/2] },
    ];
    for (const s of shells) {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(...s.size), pmlMat);
      mesh.position.set(...s.pos);
      pml.add(mesh);
    }

    const boxGeom = new THREE.BoxGeometry(W, H, L);
    const edges = new THREE.EdgesGeometry(boxGeom);
    const outline = new THREE.LineSegments(edges, pmlEdgeMat);
    outline.computeLineDistances();
    pml.add(outline);
  };

  // Initial & reactive builds
  useEffect(() => { buildSolids(); }, [domain, p.showCore, p.showSubstrate, p.showCladding, p.coreHeight, p.boxThick, p.cladThick, p.wgType, p.widthStart, p.widthEnd, p.ribWidthStart, p.ribWidthEnd, p.baseWidthStart, p.baseWidthEnd, p.etchDepth, p.sweepSegs, p.coreLength, p.bendAngleDeg]);
  useEffect(() => { buildGrid(); }, [domain, p.showGrid, p.nx, p.ny, p.nz, p.gridOpacity]);
  useEffect(() => { buildPML(); }, [domain, p.showPML, p.pml]);

  // ---------- Material tagging helper (for export) ----------
  // Map world (x,z) to (s,u): s along centerline [0,L], u~signed radial offset (inward +)
  function worldToSU(x, z) {
    // Convert to absolute coords (undo centering shift)
    const X = x + domain.centerShift.x;
    const Z = z + domain.centerShift.z;

    if (bend.straight) {
      const s = clamp(Z, 0, bend.L);
      const u = -X; // inward (+z direction has u positive to -x to mimic radial inward)
      return { s, u };
    }
    const R = bend.R * bend.sign; // signed radius for convenience
    const Cx = R; const Cz = 0;
    const vx = X - Cx; const vz = Z - Cz;
    const alpha = Math.atan2(vz, vx); // angle from center
    let s = Math.abs(bend.R) * (alpha); // signed arc length; for sign>0, increases with alpha
    if (bend.sign < 0) s = -s;
    s = clamp(s, 0, bend.L);
    const r = Math.hypot(vx, vz);
    const u = (Math.abs(bend.R) - r); // inward (toward centerline) is positive
    return { s, u };
  }

  function widthAt(s) {
    if (p.wgType === "strip") return widths.widthStrip(s);
    return widths.ribWidth(s);
  }
  function baseWidthAt(s) {
    if (p.wgType === "strip") return widths.widthStrip(s);
    return widths.baseWidth(s);
  }

  // ---------- Export mesh JSON ----------
  const generateMeshData = () => {
    const { W, H, L } = domain;
    const nx = Math.floor(p.nx), ny = Math.floor(p.ny), nz = Math.floor(p.nz);
    const xs = linspace(-W/2, W/2, nx+1);
    const ys = linspace(-(H/2), +(H/2), ny+1);
    const zs = linspace(-L/2, L/2, nz+1);

    // Nodes
    const nodes = [];
    for (let k = 0; k <= nz; k++)
      for (let j = 0; j <= ny; j++)
        for (let i = 0; i <= nx; i++)
          nodes.push([xs[i], ys[j], zs[k]]);

    const nodeIndex = (i,j,k) => i + (nx+1)*j + (nx+1)*(ny+1)*k;

    const slabTh = widths.slabThick;

    const elementsHex = [];
    for (let k = 0; k < nz; k++) {
      const zc = 0.5*(zs[k] + zs[k+1]);
      for (let j = 0; j < ny; j++) {
        const yc = 0.5*(ys[j] + ys[j+1]);
        for (let i = 0; i < nx; i++) {
          const xc = 0.5*(xs[i] + xs[i+1]);
          const n000 = nodeIndex(i, j, k);
          const n100 = nodeIndex(i+1, j, k);
          const n110 = nodeIndex(i+1, j+1, k);
          const n010 = nodeIndex(i, j+1, k);
          const n001 = nodeIndex(i, j, k+1);
          const n101 = nodeIndex(i+1, j, k+1);
          const n111 = nodeIndex(i+1, j+1, k+1);
          const n011 = nodeIndex(i, j+1, k+1);

          // Identify material at element center
          let material = 'cladding';
          const yTopCore = +(p.coreHeight/2);
          const yBottomCore = -(p.coreHeight/2);

          if (yc < yBottomCore) {
            material = 'box';
          } else if (yc >= yBottomCore && yc <= yTopCore) {
            // Inside device-layer band: decide silicon presence with rib/strip + taper + bend
            const { s, u } = worldToSU(xc, zc);
            const wR = widthAt(s);
            const wB = baseWidthAt(s);
            const ySlabTop = yBottomCore + slabTh;

            if (p.wgType === 'strip') {
              // full-etch strip: silicon only if |u| <= w/2 across full height
              if (Math.abs(u) <= wR/2) material = 'core';
              else material = 'cladding';
            } else {
              // rib: ridge full height where |u|<=wR/2; slab region limited to y<=ySlabTop for wR/2<|u|<=wB/2
              if (Math.abs(u) <= wR/2) material = 'core';
              else if (Math.abs(u) <= wB/2) {
                material = (yc <= ySlabTop + 1e-9) ? 'core' : 'cladding';
              } else material = 'cladding';
            }
          } else {
            material = 'cladding';
          }

          // PML flag
          const t = p.pml;
          const inPML = (Math.abs(xc) > (W/2 - t)) || (Math.abs(yc) > (H/2 - t)) || (Math.abs(zc) > (L/2 - t));

          elementsHex.push({ type: 'hex8', nodes: [n000, n100, n110, n010, n001, n101, n111, n011], material, pml: !!inPML });
        }
      }
    }

    if (p.meshType === 'hex') {
      return { units: 'micrometer', params: { ...p }, domain, nodes, elements: elementsHex };
    }

    // Convert to tetra (5 tets per hex)
    const elementsTet = [];
    for (const e of elementsHex) {
      const [a,b,c,d,e1,f,g,h] = e.nodes; // naming: a=n000, b=n100, c=n110, d=n010, e1=n001, f=n101, g=n111, h=n011
      // Diagonal from a(000) to g(111)
      elementsTet.push({ type:'tet4', nodes:[a,b,c,g], material:e.material, pml:e.pml });
      elementsTet.push({ type:'tet4', nodes:[a,c,d,g], material:e.material, pml:e.pml });
      elementsTet.push({ type:'tet4', nodes:[a,d,h,g], material:e.material, pml:e.pml });
      elementsTet.push({ type:'tet4', nodes:[a,h,e1,g], material:e.material, pml:e.pml });
      elementsTet.push({ type:'tet4', nodes:[a,e1,f,g], material:e.material, pml:e.pml });
    }

    return { units: 'micrometer', params: { ...p }, domain, nodes, elements: elementsTet };
  };

  // --------- Export & Snapshot ----------
  const download = (filename, dataStr, mime = "application/json") => {
    const blob = new Blob([dataStr], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1200);
  };

  const onExport = () => {
    const data = generateMeshData();
    download("soi_waveguide_mesh.json", JSON.stringify(data));
  };

  const onShot = () => {
    const { renderer, camera, scene } = threeRef.current;
    // Render current view and force a PNG download with timestamped filename
    const ts = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const fname = `soi_waveguide_${ts.getFullYear()}${pad(ts.getMonth()+1)}${pad(ts.getDate())}_${pad(ts.getHours())}${pad(ts.getMinutes())}${pad(ts.getSeconds())}.png`;

    renderer.render(scene, camera);

    if (renderer.domElement.toBlob) {
      renderer.domElement.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = fname; a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1200);
      }, 'image/png');
    } else {
      // Fallback
      const dataURL = renderer.domElement.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = dataURL; a.download = fname; a.click();
    }
  };

  // ---------- UI helpers ----------
  const setNum = (key) => (e) => setP((s) => ({ ...s, [key]: Number(e.target.value) }));
  const setBool = (key) => (e) => setP((s) => ({ ...s, [key]: e.target.checked }));
  const setSel = (key) => (e) => setP((s) => ({ ...s, [key]: e.target.value }));

  // Apply provided physics params to geometry UI (strip wg preset)
  const onApplyParams = () => {
    setP((s) => ({
      ...s,
      wgType: 'strip',
      widthStart: 0.5,
      widthEnd: 0.5,
      coreHeight: 0.22,
      coreLength: 10.0,
    }));
  };

  return (
    <div className="w-full h-screen bg-[#0b1020] text-slate-100 grid grid-rows-[48px_1fr]">
      {/* Header */}
      <div className="flex items-center gap-3 px-3 border-b border-[#1d2a55] bg-[#0f1733]">
        <div className="font-semibold text-sm">SOI Waveguide 3D Mesh (Rib/Strip · Taper · Bend)</div>
        <div className="text-xs opacity-80">units: µm</div>
        <div className="flex-1" />
        <select value={p.meshType} onChange={setSel("meshType")} className="text-xs px-2 py-1 rounded-xl border border-[#2a3b7d] bg-[#142157]">
          <option value="hex">Export: HEX</option>
          <option value="tet">Export: TET</option>
        </select>
        <button onClick={onExport} className="text-xs px-3 py-1.5 rounded-xl border border-[#2a3b7d] bg-[#142157]">Export .json</button>
        <button onClick={onShot} className="text-xs px-3 py-1.5 rounded-xl border border-[#2a3b7d] bg-[#142157]">Snapshot</button>
        <button onClick={onApplyParams} className="text-xs px-3 py-1.5 rounded-xl border border-[#2a3b7d] bg-[#0e3d1a]">Apply Params</button>
      </div>

      {/* Main */}
      <div className="relative w-full h-full">
        {/* Controls Panel */}
        <div className="absolute left-3 top-3 z-10 p-3 rounded-2xl border border-[#223065] bg-black/40 backdrop-blur-sm text-xs space-y-2 w-[340px]">
          <div className="font-semibold text-[11px] mb-1">Device / Domain</div>
          <div className="grid grid-cols-2 gap-2">
            <label className="flex items-center gap-2">Core H
              <input type="number" step="0.005" value={p.coreHeight} onChange={setNum("coreHeight")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            </label>
            <label className="flex items-center gap-2">BOX
              <input type="number" step="0.05" value={p.boxThick} onChange={setNum("boxThick")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            </label>
            <label className="flex items-center gap-2">Clad
              <input type="number" step="0.05" value={p.cladThick} onChange={setNum("cladThick")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            </label>
            <label className="flex items-center gap-2">Margin
              <input type="number" step="0.05" value={p.sideMargin} onChange={setNum("sideMargin")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            </label>
          </div>

          <div className="font-semibold text-[11px] mt-2">Waveguide</div>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2"><input type="radio" name="wg" checked={p.wgType==='strip'} onChange={()=>setP(s=>({...s,wgType:'strip', etchDepth: s.coreHeight}))}/> Strip</label>
            <label className="flex items-center gap-2"><input type="radio" name="wg" checked={p.wgType==='rib'} onChange={()=>setP(s=>({...s,wgType:'rib'}))}/> Rib</label>
          </div>

          {p.wgType === 'strip' ? (
            <div className="grid grid-cols-2 gap-2">
              <label className="flex items-center gap-2">W₀
                <input type="number" step="0.005" value={p.widthStart} onChange={setNum("widthStart")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
              </label>
              <label className="flex items-center gap-2">W₁
                <input type="number" step="0.005" value={p.widthEnd} onChange={setNum("widthEnd")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
              </label>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              <label className="flex items-center gap-2">Base₀
                <input type="number" step="0.01" value={p.baseWidthStart} onChange={setNum("baseWidthStart")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
              </label>
              <label className="flex items-center gap-2">Base₁
                <input type="number" step="0.01" value={p.baseWidthEnd} onChange={setNum("baseWidthEnd")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
              </label>
              <label className="flex items-center gap-2">Rib₀
                <input type="number" step="0.005" value={p.ribWidthStart} onChange={setNum("ribWidthStart")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
              </label>
              <label className="flex items-center gap-2">Rib₁
                <input type="number" step="0.005" value={p.ribWidthEnd} onChange={setNum("ribWidthEnd")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
              </label>
              <label className="flex items-center gap-2">Etch
                <input type="number" step="0.005" value={p.etchDepth} onChange={setNum("etchDepth")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
              </label>
            </div>
          )}

          <div className="font-semibold text-[11px] mt-2">Length & Bend</div>
          <div className="grid grid-cols-2 gap-2">
            <label className="flex items-center gap-2">L
              <input type="number" step="0.1" value={p.coreLength} onChange={setNum("coreLength")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            </label>
            <label className="flex items-center gap-2">θ (deg)
              <input type="number" step="0.5" value={p.bendAngleDeg} onChange={setNum("bendAngleDeg")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            </label>
            <label className="flex items-center gap-2 col-span-2">Sweep segs
              <input type="number" step="1" value={p.sweepSegs} onChange={setNum("sweepSegs")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            </label>
          </div>

          <div className="font-semibold text-[11px] mt-2">Structured Grid</div>
          <div className="grid grid-cols-3 gap-2">
            <label className="flex items-center gap-1">Nx
              <input type="number" step="1" value={p.nx} onChange={setNum("nx")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            </label>
            <label className="flex items-center gap-1">Ny
              <input type="number" step="1" value={p.ny} onChange={setNum("ny")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            </label>
            <label className="flex items-center gap-1">Nz
              <input type="number" step="1" value={p.nz} onChange={setNum("nz")} className="w-full bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            </label>
          </div>

          <div className="font-semibold text-[11px] mt-2">Visibility</div>
          <div className="grid grid-cols-2 gap-2 mt-1">
            <label className="flex items-center gap-2"><input type="checkbox" checked={p.showGrid} onChange={setBool("showGrid")} /> Volume grid</label>
            <label className="flex items-center gap-2"><input type="checkbox" checked={p.showFloor} onChange={setBool("showFloor")} /> Floor grid</label>
            <label className="flex items-center gap-2"><input type="checkbox" checked={p.showAxes} onChange={setBool("showAxes")} /> Axes</label>
            <label className="flex items-center gap-2"><input type="checkbox" checked={p.showCore} onChange={setBool("showCore")} /> Si core</label>
            <label className="flex items-center gap-2"><input type="checkbox" checked={p.showSubstrate} onChange={setBool("showSubstrate")} /> SiO₂ BOX</label>
            <label className="flex items-center gap-2"><input type="checkbox" checked={p.showCladding} onChange={setBool("showCladding")} /> Cladding</label>
            <label className="flex items-center gap-2"><input type="checkbox" checked={p.showPML} onChange={setBool("showPML")} /> PML</label>
          </div>

          <div className="mt-2 flex items-center gap-2">
            <span>PML</span>
            <input type="number" step="0.01" value={p.pml} onChange={setNum("pml")} className="w-24 bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
            <span>Grid α</span>
            <input type="number" step="0.01" value={p.gridOpacity} onChange={setNum("gridOpacity")} className="w-24 bg-transparent border border-[#2a3b7d] rounded px-2 py-1" />
          </div>

          <div className="text-[11px] opacity-80 mt-2">Controls: orbit (drag), pan (right/ctrl), zoom (wheel)</div>
        </div>

        {/* Canvas host */}
        <div ref={hostRef} className="absolute inset-0" />
      </div>
    </div>
  );
}
