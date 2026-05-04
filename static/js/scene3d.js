/**
 * Rich ambient 3D scene behind AnemiaVision (Three.js).
 * SaaS / metaverse–style depth: flowing ribbons, shader sky, instanced helix,
 * orbital line halos. Time drives motion (3D + time as the “fourth” dimension).
 * Respects prefers-reduced-motion and missing WebGL.
 */
(function initAnemiaVisionScene3d() {
    const canvas = document.getElementById("av-scene-canvas");
    if (!canvas || typeof THREE === "undefined") {
        return;
    }

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (reducedMotion.matches) {
        canvas.classList.add("av-scene-canvas--static");
        return;
    }

    function hexToVec3(hex) {
        const c = new THREE.Color(hex);
        return new THREE.Vector3(c.r, c.g, c.b);
    }

    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
        powerPreference: "high-performance"
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 200);
    camera.position.set(0, 0.4, 18.2);

    const mainGroup = new THREE.Group();
    const orbitGroup = new THREE.Group();
    const ribbonGroup = new THREE.Group();
    const helixGroup = new THREE.Group();
    const lineHaloGroup = new THREE.Group();
    const waveGridGroup = new THREE.Group();
    scene.add(mainGroup);
    scene.add(orbitGroup);
    mainGroup.add(ribbonGroup);
    mainGroup.add(helixGroup);
    mainGroup.add(lineHaloGroup);
    mainGroup.add(waveGridGroup);

    function themePalette() {
        const dark = document.documentElement.getAttribute("data-theme") === "dark";
        if (dark) {
            return {
                fog: 0x02060d,
                ambient: 0x152535,
                key: 0x5eead4,
                fill: 0xf472b6,
                wire: 0x38bdf8,
                particle: 0x7dd3fc,
                accent: 0xa78bfa,
                deep: 0x312e81
            };
        }
        return {
            fog: 0xe4f2f5,
            ambient: 0xffffff,
            key: 0x0d9488,
            fill: 0xe11d48,
            wire: 0x0ea5e9,
            particle: 0x2dd4bf,
            accent: 0x6366f1,
            deep: 0x1e3a5f
        };
    }

    let palette = themePalette();
    scene.fog = new THREE.FogExp2(palette.fog, 0.019);

    function darkModeStrength() {
        return document.documentElement.getAttribute("data-theme") === "dark" ? 0.34 : 0.54;
    }

    const ambient = new THREE.AmbientLight(palette.ambient, darkModeStrength());
    scene.add(ambient);

    const hemi = new THREE.HemisphereLight(palette.key, palette.fill, 0.38);
    scene.add(hemi);

    const keyLight = new THREE.DirectionalLight(palette.key, 1.22);
    keyLight.position.set(10, 14, 14);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(palette.fill, 0.68);
    rimLight.position.set(-12, -6, -8);
    scene.add(rimLight);

    const accentLight = new THREE.PointLight(palette.accent, 1.05, 48, 2);
    accentLight.position.set(0, 3, 6);
    scene.add(accentLight);

    const fillPoint = new THREE.PointLight(palette.fill, 0.62, 32, 2);
    fillPoint.position.set(-6, -2, 4);
    scene.add(fillPoint);

    const coolPoint = new THREE.PointLight(palette.wire, 0.45, 36, 2);
    coolPoint.position.set(8, -4, -2);
    scene.add(coolPoint);

    const meshes = [];

    function addShape(geometry, colorHex, emissiveHex, wireframe, opacity = 1, scale = 1) {
        const Mat = wireframe ? THREE.MeshStandardMaterial : THREE.MeshPhysicalMaterial;
        const mat = new Mat({
            color: colorHex,
            emissive: emissiveHex,
            emissiveIntensity: wireframe ? 0.55 : 0.28,
            metalness: 0.42,
            roughness: 0.35,
            wireframe,
            transparent: opacity < 1,
            opacity,
            ...(wireframe
                ? {}
                : {
                      clearcoat: 0.52,
                      clearcoatRoughness: 0.2
                  })
        });
        const mesh = new THREE.Mesh(geometry, mat);
        mesh.scale.setScalar(scale);
        mesh.userData.emissiveKey = null;
        mesh.userData.wire = wireframe;
        if (!wireframe) {
            mesh.userData.emissiveKey = "key";
        }
        mainGroup.add(mesh);
        meshes.push(mesh);
        return mesh;
    }

    const torus = new THREE.TorusGeometry(1.2, 0.4, 28, 56);
    const t1 = addShape(torus, 0xffffff, palette.key, false, 0.9, 1.12);
    t1.position.set(-4.5, 1.5, -1);
    t1.rotation.set(0.55, 0.4, 0);

    const t2 = addShape(new THREE.TorusGeometry(0.9, 0.24, 22, 44), 0xffffff, palette.fill, false, 0.85, 1);
    t2.position.set(4.8, -1.4, -2.2);
    t2.rotation.set(-0.65, 0.2, 0.45);

    const ico = new THREE.IcosahedronGeometry(1.08, 0);
    const i1 = addShape(ico, 0xf8fafc, palette.key, false, 0.72, 1.25);
    i1.position.set(3, 2.8, -4.5);

    const wireIco = new THREE.IcosahedronGeometry(1.55, 1);
    const w1 = addShape(wireIco, palette.wire, palette.wire, true, 0.32, 1);
    w1.position.set(-3.2, -2.4, -3.5);
    w1.userData.emissiveKey = null;

    const oct = new THREE.OctahedronGeometry(1, 0);
    const o1 = addShape(oct, 0xffffff, palette.accent, false, 0.68, 1.05);
    o1.position.set(-5.8, 2.4, -5.5);
    o1.userData.emissiveKey = "accent";

    const knot = new THREE.TorusKnotGeometry(0.78, 0.24, 120, 14);
    const k1 = addShape(knot, 0xffffff, palette.fill, false, 0.8, 0.98);
    k1.position.set(5.5, 2, -4.5);
    k1.rotation.y = 1.15;

    const capsule = new THREE.CapsuleGeometry(0.32, 1.4, 8, 16);
    const c1 = addShape(capsule, 0xffffff, palette.key, false, 0.75, 1.1);
    c1.position.set(0.5, -2.8, -6);
    c1.rotation.z = 0.4;

    const dodec = new THREE.DodecahedronGeometry(0.92, 0);
    const d1 = addShape(dodec, 0xffffff, palette.accent, false, 0.7, 1.08);
    d1.position.set(-2.2, 3.1, -5.8);
    d1.userData.emissiveKey = "accent";

    const skyShell = new THREE.Mesh(
        new THREE.IcosahedronGeometry(8.2, 2),
        new THREE.MeshStandardMaterial({
            color: 0xffffff,
            emissive: palette.wire,
            emissiveIntensity: 0.22,
            wireframe: true,
            transparent: true,
            opacity: 0.11,
            metalness: 0.2,
            roughness: 0.85
        })
    );
    skyShell.position.set(0, 0, -9);
    mainGroup.add(skyShell);

    const megaRing = new THREE.Mesh(
        new THREE.TorusGeometry(6.8, 0.07, 14, 160),
        new THREE.MeshStandardMaterial({
            color: 0xffffff,
            emissive: palette.key,
            emissiveIntensity: 0.48,
            transparent: true,
            opacity: 0.42,
            metalness: 0.88,
            roughness: 0.16,
            side: THREE.DoubleSide
        })
    );
    megaRing.rotation.x = Math.PI / 2.15;
    megaRing.position.set(0, -1.2, -5.5);
    mainGroup.add(megaRing);

    const megaRingB = new THREE.Mesh(
        new THREE.TorusGeometry(8.4, 0.04, 12, 200),
        new THREE.MeshStandardMaterial({
            color: 0xffffff,
            emissive: palette.fill,
            emissiveIntensity: 0.32,
            transparent: true,
            opacity: 0.22,
            metalness: 0.75,
            roughness: 0.2,
            side: THREE.DoubleSide
        })
    );
    megaRingB.rotation.x = Math.PI / 2.4;
    megaRingB.rotation.z = 0.35;
    megaRingB.position.set(0.6, 0.4, -7);
    mainGroup.add(megaRingB);

    const innerOrbit = new THREE.Group();
    mainGroup.add(innerOrbit);
    const miniGeo = new THREE.OctahedronGeometry(0.38, 0);
    for (let j = 0; j < 10; j += 1) {
        const ang = (j / 10) * Math.PI * 2;
        const rad = 3.1;
        const mm = addShape(miniGeo, 0xffffff, j % 2 === 0 ? palette.key : palette.accent, false, 0.62, 0.9);
        mm.position.set(Math.cos(ang) * rad, Math.sin(ang * 2.2) * 0.5, Math.sin(ang) * rad * 0.45 - 1.5);
        mm.userData.emissiveKey = j % 2 === 0 ? "key" : "accent";
        innerOrbit.add(mm);
    }

    const ringGeo = new THREE.TetrahedronGeometry(0.28, 0);
    for (let i = 0; i < 20; i += 1) {
        const a = (i / 20) * Math.PI * 2;
        const r = 7.2 + Math.sin(i * 1.7) * 0.6;
        const m = addShape(
            ringGeo,
            i % 2 === 0 ? 0xffffff : 0xecfeff,
            i % 2 === 0 ? palette.key : palette.fill,
            false,
            0.55 + (i % 5) * 0.06,
            0.85 + (i % 4) * 0.08
        );
        m.position.set(Math.cos(a) * r, Math.sin(a * 2) * 0.9, Math.sin(a) * r * 0.55 - 3);
        m.rotation.set(i * 0.4, i * 0.25, i * 0.15);
        m.userData.emissiveKey = i % 2 === 0 ? "key" : "fill";
        orbitGroup.add(m);
    }

    const skyUniforms = {
        uTime: { value: 0 },
        uColorA: { value: hexToVec3(palette.key) },
        uColorB: { value: hexToVec3(palette.fill) },
        uColorC: { value: hexToVec3(palette.accent) }
    };

    const skyVertex = `
        varying vec3 vPos;
        void main() {
            vPos = position;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `;

    const skyFragment = `
        uniform float uTime;
        uniform vec3 uColorA;
        uniform vec3 uColorB;
        uniform vec3 uColorC;
        varying vec3 vPos;
        void main() {
            vec3 dir = normalize(vPos);
            float lat = dir.y;
            float bands = sin(lat * 9.0 + uTime * 0.35) * 0.5 + 0.5;
            float ribs = sin(dir.x * 14.0 + uTime * 0.28) * sin(dir.z * 11.0 - uTime * 0.22);
            float pulse = sin(uTime * 0.45 + length(dir.xz) * 6.0) * 0.5 + 0.5;
            vec3 base = mix(uColorA, uColorB, bands * 0.65 + ribs * 0.15);
            base = mix(base, uColorC, pulse * 0.18);
            float alpha = 0.04 + bands * 0.035 + ribs * 0.02;
            gl_FragColor = vec4(base, alpha);
        }
    `;

    const skyDome = new THREE.Mesh(
        new THREE.SphereGeometry(52, 48, 32),
        new THREE.ShaderMaterial({
            uniforms: skyUniforms,
            vertexShader: skyVertex,
            fragmentShader: skyFragment,
            side: THREE.BackSide,
            depthWrite: false,
            transparent: true,
            blending: THREE.AdditiveBlending
        })
    );
    skyDome.position.set(0, 0, -6);
    skyDome.renderOrder = -5;
    scene.add(skyDome);

    const ribbonMaterials = [];

    function buildRibbonTube(seed, radius, tubularSeg) {
        const pts = [];
        const steps = 52;
        for (let i = 0; i <= steps; i += 1) {
            const u = i / steps;
            const ang = u * Math.PI * 2 * 2.55 + seed * 1.7;
            const wobble = Math.sin(u * 11 + seed * 2.1) * 0.85;
            pts.push(
                new THREE.Vector3(
                    Math.cos(ang) * (5.4 + wobble * 0.25) + Math.sin(seed) * 0.4,
                    Math.sin(u * Math.PI * 2.4 + seed) * 3.2,
                    Math.sin(ang * 1.08) * (3.8 + u * 1.4) - 4.2 + wobble * 0.15
                )
            );
        }
        const curve = new THREE.CatmullRomCurve3(pts);
        const tubeGeo = new THREE.TubeGeometry(curve, tubularSeg, radius, 10, false);
        const mat = new THREE.MeshPhysicalMaterial({
            color: 0xffffff,
            emissive: new THREE.Color(palette.key),
            emissiveIntensity: 0.42,
            metalness: 0.55,
            roughness: 0.18,
            transparent: true,
            opacity: 0.58,
            clearcoat: 1,
            clearcoatRoughness: 0.12,
            side: THREE.DoubleSide
        });
        mat.userData.emissiveKey = seed % 2 === 0 ? "key" : "fill";
        ribbonMaterials.push(mat);
        const mesh = new THREE.Mesh(tubeGeo, mat);
        mesh.userData.seed = seed;
        mesh.userData.curve = curve;
        mesh.userData.tubularSeg = tubularSeg;
        mesh.userData.radius = radius;
        ribbonGroup.add(mesh);
        return mesh;
    }

    const ribbons = [
        buildRibbonTube(0.2, 0.065, 140),
        buildRibbonTube(2.8, 0.048, 160),
        buildRibbonTube(5.1, 0.055, 120),
        buildRibbonTube(3.4, 0.038, 180)
    ];

    const sparkCount = 200;
    const sparkGeo = new THREE.IcosahedronGeometry(0.13, 0);
    const sparkMat = new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        metalness: 0.35,
        roughness: 0.25,
        transparent: true,
        opacity: 0.88,
        emissive: new THREE.Color(palette.key),
        emissiveIntensity: 0.62
    });
    const sparks = new THREE.InstancedMesh(sparkGeo, sparkMat, sparkCount);
    const dummy = new THREE.Object3D();
    for (let i = 0; i < sparkCount; i += 1) {
        const t = (i / sparkCount) * Math.PI * 2 * 7;
        const r = 4.8 + Math.sin(i * 0.31) * 1.4;
        const y = (i / sparkCount) * 9 - 4.2;
        dummy.position.set(Math.cos(t) * r, y, Math.sin(t) * r * 0.55 - 3.2);
        dummy.rotation.set(i * 0.15, i * 0.22, i * 0.09);
        dummy.scale.setScalar(0.35 + (i % 7) * 0.11);
        dummy.updateMatrix();
        sparks.setMatrixAt(i, dummy.matrix);
    }
    sparks.instanceMatrix.needsUpdate = true;
    helixGroup.add(sparks);

    const lineHalos = [];

    function addLineHalo(rx, ry, z, colorHex, opacity, tiltX, tiltZ) {
        const segs = 96;
        const positions = new Float32Array((segs + 1) * 3);
        for (let i = 0; i <= segs; i += 1) {
            const a = (i / segs) * Math.PI * 2;
            let x = Math.cos(a) * rx;
            let y = Math.sin(a) * ry;
            let zz = z;
            const cosX = Math.cos(tiltX);
            const sinX = Math.sin(tiltX);
            const ny = y * cosX - zz * sinX;
            const nz = y * sinX + zz * cosX;
            y = ny;
            zz = nz;
            const cosZ = Math.cos(tiltZ);
            const sinZ = Math.sin(tiltZ);
            const nx = x * cosZ - y * sinZ;
            const ny2 = x * sinZ + y * cosZ;
            x = nx;
            y = ny2;
            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = zz;
        }
        const lg = new THREE.BufferGeometry();
        lg.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        const line = new THREE.LineLoop(
            lg,
            new THREE.LineBasicMaterial({
                color: colorHex,
                transparent: true,
                opacity,
                blending: THREE.AdditiveBlending,
                depthWrite: false
            })
        );
        lineHaloGroup.add(line);
        lineHalos.push(line);
        return line;
    }

    addLineHalo(9.2, 5.8, -5.5, palette.wire, 0.14, 0.55, 0.2);
    addLineHalo(7.4, 7.4, -4.2, palette.key, 0.11, -0.35, 0.45);
    addLineHalo(11, 4.2, -6.8, palette.accent, 0.09, 0.22, -0.3);
    addLineHalo(5.5, 5.5, -2.8, palette.fill, 0.12, 0.65, 0.55);

    const gridCols = 11;
    const gridRows = 7;
    const tileW = 1.35;
    const tileH = 1.1;
    const tileGeo = new THREE.PlaneGeometry(tileW * 0.92, tileH * 0.92);
    const gridTileMaterials = [];
    for (let gx = 0; gx < gridCols; gx += 1) {
        for (let gy = 0; gy < gridRows; gy += 1) {
            const tileMat = new THREE.MeshPhysicalMaterial({
                color: 0xffffff,
                emissive: new THREE.Color(palette.wire),
                emissiveIntensity: 0.08,
                metalness: 0.72,
                roughness: 0.35,
                transparent: true,
                opacity: 0.12,
                side: THREE.DoubleSide
            });
            gridTileMaterials.push(tileMat);
            const tile = new THREE.Mesh(tileGeo, tileMat);
            const ox = (gx - gridCols / 2) * tileW;
            const oy = (gy - gridRows / 2) * tileH;
            tile.position.set(ox, oy, -14);
            tile.rotation.x = -Math.PI / 2.35;
            tile.userData.phase = gx * 0.37 + gy * 0.51;
            waveGridGroup.add(tile);
        }
    }
    waveGridGroup.position.y = -2.2;

    function particleLayer(count, size, opacity, spread) {
        const positions = new Float32Array(count * 3);
        for (let i = 0; i < count; i += 1) {
            const rad = spread[0] + Math.random() * spread[1];
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            positions[i * 3] = rad * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = rad * Math.sin(phi) * Math.sin(theta) * 0.5;
            positions[i * 3 + 2] = rad * Math.cos(phi) * 0.4 - 4;
        }
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        const mat = new THREE.PointsMaterial({
            color: palette.particle,
            size,
            transparent: true,
            opacity,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            sizeAttenuation: true
        });
        const pts = new THREE.Points(geom, mat);
        mainGroup.add(pts);
        return { points: pts, material: mat };
    }

    const layerA = particleLayer(820, 0.075, 0.5, [10, 26]);
    const layerB = particleLayer(420, 0.046, 0.34, [15, 34]);
    const layerC = particleLayer(280, 0.026, 0.26, [4, 16]);
    const layerD = particleLayer(160, 0.12, 0.22, [2, 9]);

    function applyPalette() {
        palette = themePalette();
        scene.fog.color.setHex(palette.fog);
        ambient.color.setHex(palette.ambient);
        ambient.intensity = darkModeStrength();
        hemi.color.setHex(palette.key);
        hemi.groundColor.setHex(palette.fill);
        keyLight.color.setHex(palette.key);
        rimLight.color.setHex(palette.fill);
        accentLight.color.setHex(palette.accent);
        fillPoint.color.setHex(palette.fill);
        coolPoint.color.setHex(palette.wire);
        skyShell.material.emissive.setHex(palette.wire);
        megaRing.material.emissive.setHex(palette.key);
        megaRingB.material.emissive.setHex(palette.fill);
        skyUniforms.uColorA.value.copy(hexToVec3(palette.key));
        skyUniforms.uColorB.value.copy(hexToVec3(palette.fill));
        skyUniforms.uColorC.value.copy(hexToVec3(palette.accent));
        sparkMat.emissive.setHex(palette.key);
        gridTileMaterials.forEach((m) => {
            m.emissive.setHex(palette.wire);
        });
        ribbonMaterials.forEach((mat, idx) => {
            const k = mat.userData.emissiveKey || (idx % 2 === 0 ? "key" : "fill");
            mat.emissive.setHex(palette[k] || palette.key);
        });
        lineHalos[0].material.color.setHex(palette.wire);
        lineHalos[1].material.color.setHex(palette.key);
        lineHalos[2].material.color.setHex(palette.accent);
        lineHalos[3].material.color.setHex(palette.fill);
        meshes.forEach((m) => {
            if (m.userData.emissiveKey) {
                const key = m.userData.emissiveKey;
                const hex = palette[key] || palette.key;
                m.material.emissive.setHex(hex);
            }
            if (m.userData.wire && m.material.color) {
                m.material.color.setHex(palette.wire);
            }
        });
        orbitGroup.children.forEach((child) => {
            if (child.userData && child.userData.emissiveKey && child.material && child.material.emissive) {
                child.material.emissive.setHex(palette[child.userData.emissiveKey] || palette.key);
            }
        });
        layerA.material.color.setHex(palette.particle);
        layerB.material.color.setHex(palette.wire);
        layerC.material.color.setHex(palette.accent);
        layerD.material.color.setHex(palette.fill);
    }

    orbitGroup.children.forEach((child) => {
        if (!child.userData) child.userData = {};
        if (!child.userData.emissiveKey) {
            child.userData.emissiveKey = "key";
        }
    });

    meshes.forEach((m) => {
        m.userData.baseY = m.position.y;
    });

    const clock = new THREE.Clock();
    let frame = 0;
    let running = true;

    function resize() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }

    function animate() {
        if (!running) {
            frame = 0;
            return;
        }
        frame = requestAnimationFrame(animate);
        const t = clock.getElapsedTime();
        skyUniforms.uTime.value = t;

        mainGroup.rotation.y = t * 0.078;
        mainGroup.rotation.x = Math.sin(t * 0.1) * 0.065;
        orbitGroup.rotation.y = -t * 0.11;
        orbitGroup.rotation.x = Math.cos(t * 0.085) * 0.038;

        ribbonGroup.rotation.y = t * 0.06;
        ribbonGroup.rotation.z = Math.sin(t * 0.07) * 0.04;
        ribbonGroup.position.y = Math.sin(t * 0.31) * 0.15;

        helixGroup.rotation.y = t * 0.19;
        helixGroup.position.x = Math.sin(t * 0.18) * 0.25;

        lineHaloGroup.rotation.y = t * 0.04;
        lineHaloGroup.rotation.x = Math.sin(t * 0.12) * 0.08;
        lineHalos.forEach((line, idx) => {
            line.rotation.z = t * (0.018 + idx * 0.004);
            line.material.opacity = 0.08 + Math.sin(t * 0.5 + idx) * 0.035;
        });

        waveGridGroup.children.forEach((tile) => {
            const ph = tile.userData.phase || 0;
            tile.position.z = -14 + Math.sin(t * 0.55 + ph) * 0.35;
            tile.material.opacity = 0.08 + Math.sin(t * 0.4 + ph * 2) * 0.05;
        });
        waveGridGroup.rotation.z = Math.sin(t * 0.08) * 0.04;

        accentLight.position.x = Math.sin(t * 0.52) * 6;
        accentLight.position.y = 2.6 + Math.cos(t * 0.38) * 1.4;
        accentLight.intensity = 0.85 + Math.sin(t * 1.7) * 0.22;

        fillPoint.position.x = Math.cos(t * 0.33) * 6;
        fillPoint.position.z = 3 + Math.sin(t * 0.26) * 1.4;
        fillPoint.intensity = 0.5 + Math.sin(t * 2) * 0.2;

        coolPoint.position.y = Math.sin(t * 0.41) * 3.5;
        coolPoint.position.z = -2 + Math.cos(t * 0.29) * 2;

        innerOrbit.rotation.y = t * 0.13;
        innerOrbit.rotation.z = Math.sin(t * 0.1) * 0.055;
        megaRing.rotation.z = t * 0.048;
        megaRingB.rotation.z = -t * 0.032;
        skyShell.rotation.y = t * 0.022;
        skyShell.rotation.x = t * 0.014;

        scene.fog.density = 0.019 + Math.sin(t * 0.2) * 0.0045;

        camera.position.x = Math.sin(t * 0.14) * 0.42;
        camera.position.y = 0.4 + Math.cos(t * 0.11) * 0.14;
        camera.position.z = 18.2 + Math.sin(t * 0.09) * 0.22;
        camera.rotation.z = Math.sin(t * 0.07) * 0.018;
        camera.lookAt(0, 0, -2.2);

        meshes.forEach((mesh, idx) => {
            mesh.rotation.x += 0.0022 + idx * 0.00032;
            mesh.rotation.y += 0.003 + idx * 0.00026;
            const baseY = mesh.userData.baseY ?? 0;
            mesh.position.y = baseY + Math.sin(t * 0.62 + idx * 1.15) * 0.4;
        });

        orbitGroup.children.forEach((mesh, idx) => {
            if (!mesh.rotation) return;
            mesh.rotation.x += 0.0038 + idx * 0.0001;
            mesh.rotation.y += 0.0046;
        });

        layerA.points.rotation.y = -t * 0.026;
        layerB.points.rotation.x = t * 0.016;
        layerC.points.rotation.z = t * 0.03;
        layerC.points.rotation.y = t * 0.01;
        layerD.points.rotation.y = t * 0.045;
        layerD.points.rotation.x = Math.sin(t * 0.2) * 0.08;

        sparks.rotation.y = t * 0.08;
        sparkMat.emissiveIntensity = 0.5 + Math.sin(t * 2.4) * 0.18;

        renderer.render(scene, camera);
    }

    const themeObserver = new MutationObserver(() => {
        applyPalette();
    });
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    window.addEventListener("resize", resize);
    resize();
    applyPalette();
    animate();

    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            running = false;
            cancelAnimationFrame(frame);
            frame = 0;
        } else {
            running = true;
            animate();
        }
    });
}());
