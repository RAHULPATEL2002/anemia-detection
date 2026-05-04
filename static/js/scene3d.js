/**
 * Rich ambient 3D scene behind AnemiaVision (Three.js).
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

    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
        powerPreference: "high-performance"
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 160);
    camera.position.set(0, 0.35, 17.5);

    const mainGroup = new THREE.Group();
    const orbitGroup = new THREE.Group();
    scene.add(mainGroup);
    scene.add(orbitGroup);

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
                accent: 0xa78bfa
            };
        }
        return {
            fog: 0xe4f2f5,
            ambient: 0xffffff,
            key: 0x0d9488,
            fill: 0xe11d48,
            wire: 0x0ea5e9,
            particle: 0x2dd4bf,
            accent: 0x6366f1
        };
    }

    let palette = themePalette();
    scene.fog = new THREE.FogExp2(palette.fog, 0.022);

    function darkModeStrength() {
        return document.documentElement.getAttribute("data-theme") === "dark" ? 0.32 : 0.52;
    }

    const ambient = new THREE.AmbientLight(palette.ambient, darkModeStrength());
    scene.add(ambient);

    const hemi = new THREE.HemisphereLight(palette.key, palette.fill, 0.35);
    scene.add(hemi);

    const keyLight = new THREE.DirectionalLight(palette.key, 1.15);
    keyLight.position.set(10, 14, 14);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(palette.fill, 0.62);
    rimLight.position.set(-12, -6, -8);
    scene.add(rimLight);

    const accentLight = new THREE.PointLight(palette.accent, 0.9, 40, 2);
    accentLight.position.set(0, 3, 6);
    scene.add(accentLight);

    const fillPoint = new THREE.PointLight(palette.fill, 0.55, 28, 2);
    fillPoint.position.set(-6, -2, 4);
    scene.add(fillPoint);

    const meshes = [];
    const extras = [];

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
                      clearcoat: 0.48,
                      clearcoatRoughness: 0.22
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

    const skyShell = new THREE.Mesh(
        new THREE.IcosahedronGeometry(8.2, 2),
        new THREE.MeshStandardMaterial({
            color: 0xffffff,
            emissive: palette.wire,
            emissiveIntensity: 0.2,
            wireframe: true,
            transparent: true,
            opacity: 0.12,
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
            emissiveIntensity: 0.45,
            transparent: true,
            opacity: 0.4,
            metalness: 0.88,
            roughness: 0.16,
            side: THREE.DoubleSide
        })
    );
    megaRing.rotation.x = Math.PI / 2.15;
    megaRing.position.set(0, -1.2, -5.5);
    mainGroup.add(megaRing);

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

    const layerA = particleLayer(640, 0.08, 0.52, [9, 22]);
    const layerB = particleLayer(360, 0.048, 0.36, [14, 30]);
    const layerC = particleLayer(220, 0.028, 0.28, [3, 14]);

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
        skyShell.material.emissive.setHex(palette.wire);
        megaRing.material.emissive.setHex(palette.key);
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

        mainGroup.rotation.y = t * 0.085;
        mainGroup.rotation.x = Math.sin(t * 0.11) * 0.07;
        orbitGroup.rotation.y = -t * 0.12;
        orbitGroup.rotation.x = Math.cos(t * 0.09) * 0.04;

        accentLight.position.x = Math.sin(t * 0.5) * 5;
        accentLight.position.y = 2.5 + Math.cos(t * 0.4) * 1.2;
        accentLight.intensity = 0.75 + Math.sin(t * 1.8) * 0.2;

        fillPoint.position.x = Math.cos(t * 0.35) * 5.5;
        fillPoint.position.z = 3 + Math.sin(t * 0.28) * 1.2;
        fillPoint.intensity = 0.48 + Math.sin(t * 2.1) * 0.18;

        innerOrbit.rotation.y = t * 0.14;
        innerOrbit.rotation.z = Math.sin(t * 0.11) * 0.06;
        megaRing.rotation.z = t * 0.052;
        skyShell.rotation.y = t * 0.024;
        skyShell.rotation.x = t * 0.016;

        scene.fog.density = 0.022 + Math.sin(t * 0.22) * 0.005;

        camera.position.x = Math.sin(t * 0.15) * 0.35;
        camera.position.y = 0.35 + Math.cos(t * 0.12) * 0.12;
        camera.lookAt(0, 0, -2);

        meshes.forEach((mesh, idx) => {
            mesh.rotation.x += 0.0024 + idx * 0.00035;
            mesh.rotation.y += 0.0032 + idx * 0.00028;
            const baseY = mesh.userData.baseY ?? 0;
            mesh.position.y = baseY + Math.sin(t * 0.65 + idx * 1.2) * 0.38;
        });

        orbitGroup.children.forEach((mesh, idx) => {
            if (!mesh.rotation) return;
            mesh.rotation.x += 0.004 + idx * 0.0001;
            mesh.rotation.y += 0.005;
        });

        layerA.points.rotation.y = -t * 0.028;
        layerB.points.rotation.x = t * 0.018;
        layerC.points.rotation.z = t * 0.032;
        layerC.points.rotation.y = t * 0.011;

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
