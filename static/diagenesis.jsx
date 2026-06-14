import React, { useRef, useMemo, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows, Html } from '@react-three/drei';
import * as THREE from 'three';

const GRID_SIZE = 10;
const ATOM_COUNT = GRID_SIZE * GRID_SIZE * GRID_SIZE;
const IMPURITY_COUNT = 300;

// Helper: deterministic chaotic noise
const pseudoRandom = (seed) => {
  let x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
};

const DiagenesisSimulation = ({ showOutline, showMath }) => {
  const meshRef = useRef();
  const impuritiesRef = useRef();
  const materialRef = useRef();
  const impurityMatRef = useRef();
  const outlineRef = useRef();

  // Generate coordinate arrays once
  const { positionsA, positionsB, impurityPositions, boxEdges } = useMemo(() => {
    const posA = new Float32Array(ATOM_COUNT * 3);
    const posB = new Float32Array(ATOM_COUNT * 3);
    
    let i = 0;
    const offset = (GRID_SIZE - 1) / 2;
    const spacing = 1.8;

    for (let x = 0; x < GRID_SIZE; x++) {
      for (let y = 0; y < GRID_SIZE; y++) {
        for (let z = 0; z < GRID_SIZE; z++) {
          // Base cubic grid centered at origin
          const bx = (x - offset) * spacing;
          const by = (y - offset) * spacing;
          const bz = (z - offset) * spacing;

          // State A (Chaotic Fossil) - add noise
          const seed = x * 100 + y * 10 + z;
          const nx = (pseudoRandom(seed) - 0.5) * 2.5;
          const ny = (pseudoRandom(seed + 1) - 0.5) * 2.5;
          const nz = (pseudoRandom(seed + 2) - 0.5) * 2.5;
          
          posA[i * 3] = bx + nx;
          posA[i * 3 + 1] = by + ny;
          posA[i * 3 + 2] = bz + nz;

          // State B (Rhombohedral Calcite Lattice) - applying a shear matrix
          const shearFactor = 0.5;
          posB[i * 3] = bx + (bz * shearFactor);
          posB[i * 3 + 1] = by + (bz * shearFactor);
          posB[i * 3 + 2] = bz;

          i++;
        }
      }
    }

    // Generate random impurities scattered throughout the volume
    const impPos = new Float32Array(IMPURITY_COUNT * 3);
    for (let j = 0; j < IMPURITY_COUNT; j++) {
      impPos[j * 3] = (Math.random() - 0.5) * GRID_SIZE * spacing * 1.2;
      impPos[j * 3 + 1] = (Math.random() - 0.5) * GRID_SIZE * spacing * 1.2;
      impPos[j * 3 + 2] = (Math.random() - 0.5) * GRID_SIZE * spacing * 1.2;
    }

    // Geometry for the outer boundary wireframe
    const boxGeo = new THREE.BoxGeometry(17, 17, 17);
    const edges = new THREE.EdgesGeometry(boxGeo);

    return { positionsA: posA, positionsB: posB, impurityPositions: impPos, boxEdges: edges };
  }, []);

  // Initialize instance matrices
  useEffect(() => {
    const dummy = new THREE.Object3D();
    
    if (meshRef.current) {
      for (let i = 0; i < ATOM_COUNT; i++) {
        dummy.position.set(positionsA[i*3], positionsA[i*3+1], positionsA[i*3+2]);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);
      }
      meshRef.current.instanceMatrix.needsUpdate = true;
    }

    if (impuritiesRef.current) {
      for (let i = 0; i < IMPURITY_COUNT; i++) {
        dummy.position.set(impurityPositions[i*3], impurityPositions[i*3+1], impurityPositions[i*3+2]);
        const scale = 0.3 + Math.random() * 0.5;
        dummy.scale.set(scale, scale, scale);
        dummy.updateMatrix();
        impuritiesRef.current.setMatrixAt(i, dummy.matrix);
      }
      impuritiesRef.current.instanceMatrix.needsUpdate = true;
    }
  }, [positionsA, impurityPositions]);

  // Animation Loop (60FPS)
  const dummy = new THREE.Object3D();
  useFrame(() => {
    const slider = document.getElementById('diagenesis-slider');
    const label = document.getElementById('progress-value');
    if (!slider) return;
    
    const t = parseFloat(slider.value) / 100;
    if (label) label.innerText = `${slider.value}%`;

    // 1. Mathematical Interpolation of Coordinates
    if (meshRef.current) {
      for (let i = 0; i < ATOM_COUNT; i++) {
        const x = THREE.MathUtils.lerp(positionsA[i*3], positionsB[i*3], t);
        const y = THREE.MathUtils.lerp(positionsA[i*3+1], positionsB[i*3+1], t);
        const z = THREE.MathUtils.lerp(positionsA[i*3+2], positionsB[i*3+2], t);
        
        dummy.position.set(x, y, z);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);
      }
      meshRef.current.instanceMatrix.needsUpdate = true;
    }

    // 2. Interpolate Materials for Phase Transition
    if (materialRef.current) {
      materialRef.current.roughness = THREE.MathUtils.lerp(1.0, 0.1, t);
      materialRef.current.transmission = THREE.MathUtils.lerp(0.0, 0.95, t);
      materialRef.current.ior = THREE.MathUtils.lerp(1.0, 1.55, t);
      
      const startColor = new THREE.Color('#8a9ba8');
      const endColor = new THREE.Color('#e0ffff');
      materialRef.current.color.lerpColors(startColor, endColor, t);
    }

    if (impurityMatRef.current) {
      impurityMatRef.current.opacity = THREE.MathUtils.lerp(1.0, 0.0, t);
    }

    // 3. Transform the outline wireframe to match the shear
    if (outlineRef.current) {
      const currentShear = 0.5 * t;
      const shearMatrix = new THREE.Matrix4().set(
        1, 0, currentShear, 0,
        0, 1, currentShear, 0,
        0, 0, 1,            0,
        0, 0, 0,            1
      );
      outlineRef.current.matrix.identity();
      outlineRef.current.applyMatrix4(shearMatrix);
      outlineRef.current.matrixAutoUpdate = false;
    }
  });

  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1.5} />
      <Environment preset="city" />
      
      {showMath && (
        <group>
          {/* Thick Mathematical Axes (X=Red, Y=Green, Z=Blue) */}
          <group>
            <mesh position={[8, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
              <cylinderGeometry args={[0.1, 0.1, 16]} />
              <meshBasicMaterial color="#ff4444" />
            </mesh>
            <mesh position={[0, 8, 0]}>
              <cylinderGeometry args={[0.1, 0.1, 16]} />
              <meshBasicMaterial color="#44ff44" />
            </mesh>
            <mesh position={[0, 0, 8]} rotation={[Math.PI / 2, 0, 0]}>
              <cylinderGeometry args={[0.1, 0.1, 16]} />
              <meshBasicMaterial color="#4488ff" />
            </mesh>
          </group>

          {/* 3D Background Grid Planes centered on origin */}
          <gridHelper args={[80, 40, '#8a9ba8', '#4a4e69']} position={[0, 0, 0]} />
          <gridHelper args={[80, 40, '#8a9ba8', '#4a4e69']} position={[0, 0, 0]} rotation={[Math.PI / 2, 0, 0]} />
          <gridHelper args={[80, 40, '#8a9ba8', '#4a4e69']} position={[0, 0, 0]} rotation={[0, 0, Math.PI / 2]} />
        </group>
      )}

      {showOutline && (
        <lineSegments ref={outlineRef} geometry={boxEdges}>
          <lineBasicMaterial color="#ff3366" linewidth={2} />
        </lineSegments>
      )}

      {/* Main Atoms (Calcium Carbonate) */}
      <instancedMesh ref={meshRef} args={[null, null, ATOM_COUNT]}>
        <sphereGeometry args={[0.35, 16, 16]} />
        <meshPhysicalMaterial 
          ref={materialRef} 
          color="#8a9ba8"
          roughness={1.0}
          transmission={0.0}
          thickness={1.5}
          transparent={true}
          opacity={showOutline ? 0.3 : 1.0}
        />
      </instancedMesh>

      {/* Impurities (Dirt/Organics) */}
      <instancedMesh ref={impuritiesRef} args={[null, null, IMPURITY_COUNT]} visible={!showOutline}>
        <dodecahedronGeometry args={[0.2]} />
        <meshStandardMaterial 
          ref={impurityMatRef} 
          color="#ff3366" 
          emissive="#ff3366"
          emissiveIntensity={0.6}
          roughness={0.5} 
          transparent={true} 
          opacity={1.0} 
        />
      </instancedMesh>

      {/* Soft floor shadow */}
      <ContactShadows position={[0, -11.9, 0]} opacity={0.6} scale={40} blur={2} far={15} />
      
      {/* User Interaction */}
      <OrbitControls makeDefault autoRotate autoRotateSpeed={0.5} maxDistance={70} />
    </>
  );
};

const DelayedFallback = () => {
  const [show, setShow] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setShow(true), 3000);
    return () => clearTimeout(timer);
  }, []);
  
  if (!show) return null;
  return (
    <Html center>
      <div style={{ color: 'var(--color-primary)', background: 'var(--color-panel-bg)', padding: '24px 40px', borderRadius: '12px', border: '1px solid var(--color-primary)', fontSize: '1.4rem', fontWeight: 'bold', whiteSpace: 'nowrap', boxShadow: '0 8px 32px rgba(0,0,0,0.5)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
        <span style={{ fontSize: '2.5rem' }}>⏳</span>
        <span>Warming up engine... please wait a second</span>
      </div>
    </Html>
  );
};

const App = () => {
  const [showOutline, setShowOutline] = useState(false);
  const [showMath, setShowMath] = useState(true);

  useEffect(() => {
    const outlineBtn = document.getElementById('toggle-outline-btn');
    if (outlineBtn) {
      const handleOutlineClick = () => {
        setShowOutline(prev => {
          const newState = !prev;
          if (newState) outlineBtn.classList.add('active');
          else outlineBtn.classList.remove('active');
          return newState;
        });
      };
      outlineBtn.addEventListener('click', handleOutlineClick);
      return () => outlineBtn.removeEventListener('click', handleOutlineClick);
    }
  }, []);

  useEffect(() => {
    const mathBtn = document.getElementById('toggle-math-btn');
    if (mathBtn) {
      const handleMathClick = () => {
        setShowMath(prev => {
          const newState = !prev;
          if (newState) mathBtn.classList.add('active');
          else mathBtn.classList.remove('active');
          return newState;
        });
      };
      mathBtn.addEventListener('click', handleMathClick);
      return () => mathBtn.removeEventListener('click', handleMathClick);
    }
  }, []);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {showOutline && (
        <div style={{ position: 'absolute', top: '24px', left: '24px', background: 'rgba(0,0,0,0.85)', padding: '15px', borderRadius: '8px', color: '#fff', textAlign: 'center', border: '1px solid #ff3366', minWidth: '180px', pointerEvents: 'none', zIndex: 10, boxShadow: '0 8px 32px rgba(0,0,0,0.5)' }}>
          <h3 style={{ margin: '0 0 8px 0', fontSize: '1.4rem', color: '#ff3366', letterSpacing: '1px' }}>Trigonal System</h3>
          <p style={{ margin: 0, fontFamily: 'monospace', fontSize: '1.2rem' }}>a = b = c</p>
          <p style={{ margin: 0, fontFamily: 'monospace', fontSize: '1.2rem' }}>α = β = γ ≠ 90°</p>
        </div>
      )}
      <Canvas camera={{ position: [28, 28, 28], fov: 45 }}>
        <React.Suspense fallback={<DelayedFallback />}>
          <DiagenesisSimulation showOutline={showOutline} showMath={showMath} />
        </React.Suspense>
      </Canvas>
    </div>
  );
};

const rootElement = document.getElementById('r3f-root');
if (rootElement) {
  const root = createRoot(rootElement);
  root.render(<App />);
}
