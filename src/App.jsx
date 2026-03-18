import React, { useState, useMemo, useRef, useEffect } from "react";
import 'katex/dist/katex.min.css';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, ReferenceLine,
  ScatterChart, Scatter
} from "recharts";

/* ===============================
   Physics Engine (RK4)
================================*/
function rk4Step(f, fp, fpp, beta, deta) {
  const deriv = (y0, y1, y2) => -y0 * y2 - beta * (1 - y1 * y1);
  const k1_0 = fp, k1_1 = fpp, k1_2 = deriv(f, fp, fpp);
  const k2_0 = fp + 0.5 * deta * k1_1, k2_1 = fpp + 0.5 * deta * k1_2,
        k2_2 = deriv(f + 0.5 * deta * k1_0, fp + 0.5 * deta * k1_1, fpp + 0.5 * deta * k1_2);
  const k3_0 = fp + 0.5 * deta * k2_1, k3_1 = fpp + 0.5 * deta * k2_2,
        k3_2 = deriv(f + 0.5 * deta * k2_0, fp + 0.5 * deta * k2_1, fpp + 0.5 * deta * k2_2);
  const k4_0 = fp + deta * k3_1, k4_1 = fpp + deta * k3_2,
        k4_2 = deriv(f + deta * k3_0, fp + deta * k3_1, fpp + deta * k3_2);

  return [
    f + (deta / 6) * (k1_0 + 2 * k2_0 + 2 * k3_0 + k4_0),
    fp + (deta / 6) * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1),
    fpp + (deta / 6) * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2),
  ];
}

function integrate(beta, fpp0, etaMax = 10, N = 80) {
  const deta = etaMax / N;
  let [f, fp, fpp] = [0, 0, fpp0];
  const result = [];
  for (let i = 0; i <= N; i++) {
    const eta = i * deta;
    result.push({ eta, f, fp, fpp });
    [f, fp, fpp] = rk4Step(f, fp, fpp, beta, deta);
    if (Math.abs(fp) > 2) break;
  }
  return result;
}

function shootingSolve(beta) {
  let [a, b] = [0, 2];
  for (let i = 0; i < 12; i++) {
    const mid = (a + b) / 2;
    const sol = integrate(beta, mid);
    const fpInf = sol[sol.length - 1]?.fp || 0;
    fpInf > 1 ? (b = mid) : (a = mid);
  }
  return integrate(beta, (a + b) / 2);
}

/* ===============================
   Wall Shear vs Beta Curve (computed once)
================================*/
function buildWallShearCurve() {
  const data = [];
  for (let b = -0.18; b <= 1.2; b += 0.05) {
    const sol = shootingSolve(b);
    data.push({
      beta: parseFloat(b.toFixed(2)),
      shear: parseFloat(sol[0].fpp.toFixed(4))
    });
  }
  return data;
}

/* ===============================
   Velocity Field (boundary layer grows downstream)
================================*/
function buildVelocityField(solution) {
  const field = [];
  const xmax = 6, ymax = 4, Nx = 40, Ny = 40;
  for (let i = 0; i <= Nx; i++) {
    const x = i / Nx * xmax;
    for (let j = 0; j <= Ny; j++) {
      const y = j / Ny * ymax;
      const eta = y / Math.sqrt(x + 0.2);
      const p = solution.find(p => p.eta >= eta);
      const u = p ? p.fp : 1;
      field.push({ x, y, u });
    }
  }
  return field;
}

/* ===============================
   Flow regime color
================================*/
function regimeColor(beta) {
  if (beta <= -0.198) return "#7f1d1d";   // separation — dark red
  if (beta < 0)       return "#431407";   // adverse gradient — burnt orange
  if (beta === 0)     return "#0369a1";   // Blasius — blue
  if (beta < 1)       return "#0369a1";   // accelerated — blue
  return "#065f46";                        // stagnation / super-accelerated — dark green
}

/* ===============================
   Main Component
================================*/
export default function App() {
  // beta: live value shown in UI (updates every slider tick)
  const [beta, setBeta] = useState(0);
  // solverBeta: throttled value fed to the solver (at most once per rAF)
  const [solverBeta, setSolverBeta] = useState(0);
  const rafRef = useRef(null);

  const handleSlider = (e) => {
    const val = parseFloat(e.target.value);
    setBeta(val); // instant: updates label, color, wedge angle immediately
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => setSolverBeta(val));
  };

  useEffect(() => () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); }, []);

  // Live display values — always in sync with slider
  const m = beta / (2 - beta + 1e-12);
  const wedgeAngle = (m / (1 + m)) * 180;

  // Heavy computations keyed off solverBeta only
  const mSolver = solverBeta / (2 - solverBeta + 1e-12);
  const solution = useMemo(() => shootingSolve(solverBeta), [solverBeta]);
  const velocityField = useMemo(() => buildVelocityField(solution), [solution]);
  const shearCurve = useMemo(() => buildWallShearCurve(), []);

  const chartData = useMemo(() => {
    return solution
      .filter(p => p.eta <= 8)
      .map(p => ({
        eta: parseFloat(p.eta.toFixed(2)),
        u: parseFloat(p.fp.toFixed(4)),
        tau: parseFloat(p.fpp.toFixed(4)),
        ue: parseFloat(Math.pow(1 + p.eta / 8, mSolver).toFixed(4))
      }));
  }, [solution, mSolver]);

  const metrics = useMemo(() => {
    if (solution.length < 2) return { delta: 0, theta: 0, H: 0, fpp0: 0 };
    let disp = 0, mom = 0;
    const deta = solution[1].eta - solution[0].eta;
    solution.forEach(p => {
      disp += (1 - p.fp) * deta;
      mom += (p.fp * (1 - p.fp)) * deta;
    });
    return {
      delta: disp.toFixed(3),
      theta: mom.toFixed(3),
      H: (disp / mom).toFixed(2),
      fpp0: solution[0].fpp.toFixed(4)
    };
  }, [solution]);

  return (
    <div style={{
      background: "#0f172a",
      color: "#f8fafc",
      minHeight: "100vh",
      width: "100vw",
      margin: 0,
      padding: "20px",
      boxSizing: "border-box",
      fontFamily: '"KaTeX_Main", "KaTeX_Math", serif',
      display: "flex",
      flexDirection: "column"
    }}>
      <style>{`
        body, html { margin: 0; padding: 0; overflow-x: hidden; background: #0f172a; }
      `}</style>

      <header style={{ marginBottom: "20px" }}>
        <h1 style={{ fontSize: "2rem", fontWeight: "800", color: "#38bdf8", margin: 0 }}>
          Falkner–Skan Explorer
        </h1>
        <p style={{ color: "#94a3b8", margin: 0 }}>Similarity Solutions for Boundary Layers</p>
      </header>

      <div style={{ display: "flex", flex: 1, gap: "20px", width: "100%" }}>

        {/* Left Section */}
        <div style={{
          flex: 3,
          background: "#1e293b",
          padding: "20px",
          borderRadius: "16px",
          display: "flex",
          flexDirection: "column",
          boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)"
        }}>

          {/* Velocity Profile Chart */}
          <div style={{ flex: 1, width: "100%", minHeight: "400px" }}>
            <h3 style={{ margin: "0 0 10px 0", color: "#94a3b8", fontSize: "0.85rem", textTransform: "uppercase" }}>
              Boundary Layer Profile
            </h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="eta" stroke="#94a3b8" label={{ value: 'η', position: 'insideRight', fill: '#94a3b8', offset: 10 }} />
                <YAxis stroke="#94a3b8" domain={[0, 1.4]} />
                <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: "8px" }} />
                <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: "10px" }} />
                <ReferenceLine y={1} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'Free Stream', fill: '#ef4444', position: 'right' }} />
                <Line name="Velocity (u/U)" type="monotone" dataKey="u" stroke="#38bdf8" strokeWidth={3} dot={false} isAnimationActive={false} />
                <Line name="Shear (f'')" type="monotone" dataKey="tau" stroke="#fbbf24" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line name="External Velocity (Ue)" type="monotone" dataKey="ue" stroke="#22c55e" strokeWidth={2} dot={false} isAnimationActive={false} strokeDasharray="4 4" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Wall Shear vs Beta Chart */}
          <div style={{ marginTop: "20px", height: "220px" }}>
            <h3 style={{ margin: "0 0 10px 0", color: "#94a3b8", fontSize: "0.85rem", textTransform: "uppercase" }}>
              Wall Shear f''(0) vs Pressure Gradient β
            </h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={shearCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="beta" stroke="#94a3b8" label={{ value: 'β', position: 'insideRight', fill: '#94a3b8', offset: 10 }} />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: "8px" }} />
                <ReferenceLine x={-0.198} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'Separation', fill: '#ef4444', position: 'top' }} />
                <ReferenceLine x={parseFloat(beta.toFixed(2))} stroke="#38bdf8" strokeDasharray="3 3" label={{ value: 'Current β', fill: '#38bdf8', position: 'top' }} />
                <Line name="Wall Shear f''(0)" type="monotone" dataKey="shear" stroke="#f97316" strokeWidth={3} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Velocity Field Scatter */}
          <div style={{ marginTop: "20px", height: "230px" }}>
            <h3 style={{ margin: "0 0 10px 0", color: "#94a3b8", fontSize: "0.85rem", textTransform: "uppercase" }}>
              Velocity Field (boundary layer grows downstream)
            </h3>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid stroke="#334155" />
                <XAxis dataKey="x" type="number" stroke="#94a3b8" name="x" label={{ value: 'x', position: 'insideRight', fill: '#94a3b8' }} />
                <YAxis dataKey="y" type="number" stroke="#94a3b8" name="y" label={{ value: 'y', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Scatter
                  data={velocityField}
                  fill="#38bdf8"
                  shape={(props) => {
                    const { cx, cy, payload } = props;
                    const color = `rgba(56,189,248,${Math.max(0.05, payload.u)})`;
                    return <circle cx={cx} cy={cy} r={3} fill={color} />;
                  }}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Slider */}
          <div style={{ marginTop: "20px", padding: "15px", background: "#0f172a", borderRadius: "12px" }}>
            <label style={{ display: "block", marginBottom: "6px", fontWeight: "600" }}>
              Pressure Gradient (β): <span style={{ color: "#38bdf8" }}>{beta.toFixed(2)}</span>
              {"  "}
              <span style={{ color: "#94a3b8", fontWeight: "400", fontSize: "0.9rem" }}>
                m = {m.toFixed(3)}
              </span>
            </label>
            <div style={{ marginBottom: "8px", color: "#22c55e", fontSize: "0.9rem" }}>
              Wedge angle ≈ {isFinite(wedgeAngle) ? wedgeAngle.toFixed(1) : "90.0"}°
            </div>
            <input
              type="range" min="-0.19" max="1.99" step="0.01" value={beta}
              onChange={handleSlider}
              style={{ width: "100%", cursor: "pointer" }}
            />
            <div style={{ marginTop: "10px", fontSize: "0.85rem", color: "#64748b", lineHeight: "1.6" }}>
              β = 0 → Flat plate (Blasius) &nbsp;|&nbsp; β = 1 → Stagnation point flow &nbsp;|&nbsp; β &lt; 0 → Adverse pressure gradient
            </div>
          </div>
        </div>

        {/* Right Section */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "20px", minWidth: "280px" }}>
          <div style={{ background: "#1e293b", padding: "20px", borderRadius: "16px" }}>
            <h3 style={{ marginTop: 0, fontSize: "0.9rem", color: "#94a3b8", textTransform: "uppercase" }}>Physics Metrics</h3>
            <div style={{ display: "grid", gap: "15px" }}>
              <Metric label="Wall Shear f''(0)" value={metrics.fpp0} color="#fbbf24" />
              <Metric label="Disp. Thickness δ*" value={metrics.delta} color="#38bdf8" />
              <Metric label="Mom. Thickness θ" value={metrics.theta} color="#818cf8" />
              <Metric label="Shape Factor H" value={metrics.H} color="#f472b6" />
              <Metric label="Power-law exponent m" value={isFinite(m) ? m.toFixed(3) : "∞"} color="#22c55e" />
              <Metric label="Wedge Angle" value={`${isFinite(wedgeAngle) ? wedgeAngle.toFixed(1) : "90.0"}°`} color="#f97316" />
            </div>
          </div>

          <div style={{
            background: regimeColor(beta),
            padding: "20px",
            borderRadius: "16px",
            flex: 1,
            transition: "background 0.3s ease"
          }}>
            <h3 style={{ marginTop: 0 }}>Flow Regime</h3>
            <p style={{ fontSize: "0.95rem", lineHeight: "1.6" }}>
              {beta === 0 && "Blasius Flow — zero pressure gradient over a flat plate. Classic similarity solution."}
              {beta > 0 && beta < 1 && "Accelerated Flow — favorable pressure gradient (wedge flow). Thinner, more stable boundary layer."}
              {beta >= 1 && "Stagnation / Super-accelerated — strong favorable gradient. Very thin boundary layer with high wall shear."}
              {beta < 0 && beta > -0.198 && "Decelerated Flow — adverse pressure gradient. Boundary layer thickens; separation risk increases."}
              {beta <= -0.198 && "CRITICAL: Flow Separation! Wall shear approaches zero. Boundary layer breaks away from the surface."}
            </p>
            <div style={{ marginTop: "12px", fontSize: "0.8rem", color: "rgba(255,255,255,0.6)", lineHeight: "1.5" }}>
              <div>Ue(x) ~ x<sup>m</sup></div>
              <div>m = {isFinite(m) ? m.toFixed(3) : "∞"} &nbsp;|&nbsp; β = 2m/(1+m)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, color }) {
  return (
    <div style={{ borderLeft: `4px solid ${color}`, paddingLeft: "12px" }}>
      <div style={{ fontSize: "0.7rem", color: "#94a3b8", fontWeight: "bold" }}>{label}</div>
      <div style={{ fontSize: "1.4rem", fontWeight: "bold" }}>{value}</div>
    </div>
  );
}
