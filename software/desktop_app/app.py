"""
OFNP — Clinician Desktop Application
Open Facial Neuroprosthesis Project

REST API backend for the clinician interface (Sec. 13.1):
  - Patient profile management
  - Stimulation threshold calibration
  - Real-time session monitoring
  - Safety limit configuration
  - Session report export
  - Emergency stop

Frontend: served as HTML5 SPA from /static/
Communication to MCU (Phase 3+): WebSocket bridge to serial/BLE

Routes:
  GET  /                          → clinician dashboard
  GET  /api/status                → system status
  GET  /api/session               → current session info
  POST /api/session/start         → start session
  POST /api/session/stop          → stop session
  POST /api/emergency_stop        → EMERGENCY STOP
  GET  /api/safety/limits         → current safety limits
  POST /api/safety/limits         → update safety limits (clinician only)
  GET  /api/thresholds            → expression detection thresholds
  POST /api/thresholds            → update thresholds (calibration)
  POST /api/scenario/<name>       → inject simulation scenario
  GET  /api/events/stream         → SSE live event stream
  GET  /api/session/export        → download session CSV
  POST /api/notes                 → add clinician note
"""

from __future__ import annotations

import sys
import os
import json
import time
import threading
import queue
import logging
from pathlib import Path
from flask import Flask, jsonify, request, Response, send_from_directory

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from software.signal_processing.pipeline import (
    SignalProcessingPipeline, ExpressionThresholds
)
from software.signal_processing.emg_simulator import EMGSimulator, ScenarioRunner, SimulatedExpression
from software.safety.safety_monitor import SafetyMonitor, SafetyLimits
from software.logging.session_logger import SessionLogger

logger = logging.getLogger("ofnp.desktop_app")


# ─────────────────────────────────────────────────────────────────────────────
# Application State
# ─────────────────────────────────────────────────────────────────────────────

class OFNPApp:
    """Central application state for the clinician interface."""

    def __init__(self):
        self.simulation_only = True  # Phase 0-2: always True
        self.thresholds   = ExpressionThresholds()
        self.safety_limits = SafetyLimits()
        self.safety       = SafetyMonitor(self.safety_limits, simulation_mode=True)
        self.pipeline     = None
        self.session_log  = None
        self.simulator    = EMGSimulator()
        self._session_active = False
        self._pipeline_thread = None
        self._event_queue: queue.Queue = queue.Queue(maxsize=500)
        self._last_status: dict = {}

        self._stats = {
            "total_samples":   0,
            "triggers":        0,
            "safety_blocks":   0,
            "artifacts":       0,
            "mean_latency_ms": 0.0,
        }

    def start_session(self, patient_id: str, clinician_id: str) -> str:
        if self._session_active:
            return "already_active"

        self.session_log = SessionLogger(
            patient_id=patient_id,
            clinician_id=clinician_id,
            phase="simulation",
        )

        def safety_callback(profile, filtered, features):
            ok, reason = self.safety.validate_stimulation(profile, filtered, features)
            if not ok:
                self._stats["safety_blocks"] += 1
                self._push_event("safety_block", {"reason": reason})
            return ok, reason

        def logging_callback(entry_type, data):
            if self.session_log:
                self.session_log.log(entry_type, data)
            self._push_event(entry_type, data)
            if entry_type == "stimulation_command":
                self._stats["triggers"] += 1
            elif entry_type == "expression_estimate" and data.get("expression") != "neutral":
                pass

        self.pipeline = SignalProcessingPipeline(
            thresholds=self.thresholds,
            simulation_only=self.simulation_only,
            safety_callback=safety_callback,
            logging_callback=logging_callback,
        )

        self.safety.start_session()
        self._session_active = True
        self._start_pipeline_loop()
        logger.info("[App] Session started for patient %s", patient_id[:4] + "***")
        return self.session_log.session_id

    def stop_session(self):
        if not self._session_active:
            return
        self._session_active = False
        self.safety.stop_session()
        if self.session_log:
            self.session_log.close()
        logger.info("[App] Session stopped")

    def emergency_stop(self, reason: str = "clinician_triggered"):
        self.safety.emergency_stop(reason)
        self._session_active = False
        self._push_event("emergency_stop", {"reason": reason})
        logger.critical("[App] EMERGENCY STOP: %s", reason)

    def inject_scenario(self, scenario_name: str):
        expressions = {
            "resting":    SimulatedExpression.RESTING,
            "partial_smile": SimulatedExpression.PARTIAL_SMILE,
            "full_smile": SimulatedExpression.FULL_SMILE,
            "cheek":      SimulatedExpression.CHEEK_ACTIVATION,
            "artifact":   SimulatedExpression.MOVEMENT_ARTIFACT,
        }
        if scenario_name in expressions:
            self.simulator.set_expression(expressions[scenario_name])
            return True
        return False

    def _start_pipeline_loop(self):
        def loop():
            while self._session_active:
                self.safety.ping_watchdog()
                samples = self.simulator.next_samples()
                for sample in samples:
                    if self.pipeline:
                        self.pipeline.process_sample(sample)
                        self._stats["total_samples"] += 1
                time.sleep(0.001)  # 1ms → ~1000 Hz

        self._pipeline_thread = threading.Thread(target=loop, daemon=True, name="pipeline")
        self._pipeline_thread.start()

    def _push_event(self, event_type: str, data: dict):
        try:
            self._event_queue.put_nowait({
                "type": event_type,
                "ts":   time.time(),
                **data,
            })
        except queue.Full:
            pass  # Drop oldest if queue full

    def get_status(self) -> dict:
        return {
            "session_active":    self._session_active,
            "simulation_only":   self.simulation_only,
            "safety":            self.safety.status_report(),
            "stats":             {
                **self._stats,
                "mean_latency_ms": round(
                    self.pipeline.mean_latency_ms if self.pipeline else 0, 2
                ),
            },
            "session_id": self.session_log.session_id if self.session_log else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Flask Application
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__, static_folder="static")
    state = OFNPApp()

    # ── Dashboard ─────────────────────────────────────────────────────────────
    @app.get("/")
    def dashboard():
        return _dashboard_html()

    # ── Status ────────────────────────────────────────────────────────────────
    @app.get("/api/status")
    def api_status():
        return jsonify(state.get_status())

    # ── Session management ────────────────────────────────────────────────────
    @app.post("/api/session/start")
    def api_session_start():
        body = request.json or {}
        patient_id   = body.get("patient_id",   "anon")
        clinician_id = body.get("clinician_id", "clinician_001")
        session_id   = state.start_session(patient_id, clinician_id)
        return jsonify({"session_id": session_id, "status": "started"})

    @app.post("/api/session/stop")
    def api_session_stop():
        state.stop_session()
        return jsonify({"status": "stopped"})

    # ── Emergency stop ─────────────────────────────────────────────────────────
    @app.post("/api/emergency_stop")
    def api_emergency_stop():
        reason = (request.json or {}).get("reason", "clinician_triggered")
        state.emergency_stop(reason)
        return jsonify({"status": "emergency_stop_executed", "reason": reason})

    # ── Safety limits ──────────────────────────────────────────────────────────
    @app.get("/api/safety/limits")
    def api_get_limits():
        l = state.safety_limits
        return jsonify({
            "max_current_ma":          l.max_current_ma,
            "max_pulse_width_us":      l.max_pulse_width_us,
            "max_frequency_hz":        l.max_frequency_hz,
            "min_impedance_kohm":      l.min_impedance_kohm,
            "max_impedance_kohm":      l.max_impedance_kohm,
            "max_session_duration_s":  l.max_session_duration_s,
            "max_continuous_stim_s":   l.max_continuous_stim_s,
            "watchdog_timeout_s":      l.watchdog_timeout_s,
            "note": "All limits must be validated by a qualified clinician."
        })

    @app.post("/api/safety/limits")
    def api_set_limits():
        body = request.json or {}
        l = state.safety_limits
        # Only allow reducing limits below defaults or increasing with explicit confirmation
        if body.get("max_current_ma") is not None:
            new_val = float(body["max_current_ma"])
            if new_val > 10.0:
                return jsonify({"error": "current_limit_too_high_consult_clinician"}), 400
            l.max_current_ma = new_val
        if body.get("max_pulse_width_us") is not None:
            l.max_pulse_width_us = min(float(body["max_pulse_width_us"]), 500.0)
        if body.get("max_frequency_hz") is not None:
            l.max_frequency_hz = min(float(body["max_frequency_hz"]), 100.0)
        return jsonify({"status": "updated"})

    # ── Expression thresholds (calibration) ────────────────────────────────────
    @app.get("/api/thresholds")
    def api_get_thresholds():
        t = state.thresholds
        return jsonify({
            "partial_smile_min":    t.partial_smile_min,
            "full_smile_min":       t.full_smile_min,
            "cheek_activation_min": t.cheek_activation_min,
            "min_confidence":       t.min_confidence,
            "debounce_windows":     t.debounce_windows,
            "refractory_period_ms": t.refractory_period_ms,
        })

    @app.post("/api/thresholds")
    def api_set_thresholds():
        body = request.json or {}
        t = state.thresholds
        for key in ["partial_smile_min", "full_smile_min", "cheek_activation_min",
                    "min_confidence", "debounce_windows", "refractory_period_ms"]:
            if key in body:
                setattr(t, key, float(body[key]))
        # Re-create pipeline with new thresholds
        if state.pipeline:
            state.pipeline.detector.thresholds = t
        return jsonify({"status": "thresholds_updated"})

    # ── Scenario injection ─────────────────────────────────────────────────────
    @app.post("/api/scenario/<name>")
    def api_scenario(name: str):
        ok = state.inject_scenario(name)
        if ok:
            return jsonify({"status": "injected", "scenario": name})
        return jsonify({"error": f"unknown_scenario: {name}",
                        "available": ["resting", "partial_smile", "full_smile",
                                      "cheek", "artifact"]}), 400

    # ── Notes ──────────────────────────────────────────────────────────────────
    @app.post("/api/notes")
    def api_add_note():
        note = (request.json or {}).get("note", "")
        if state.session_log:
            state.session_log.add_note(note)
        return jsonify({"status": "noted"})

    # ── Live event stream (SSE) ────────────────────────────────────────────────
    @app.get("/api/events/stream")
    def api_events_stream():
        def generate():
            yield "data: {\"type\": \"connected\"}\n\n"
            while True:
                try:
                    event = state._event_queue.get(timeout=1.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except Exception:
                    yield ": keepalive\n\n"
        return Response(generate(),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    # ── CSV Export ─────────────────────────────────────────────────────────────
    @app.get("/api/session/export")
    def api_export():
        if not state.session_log:
            return jsonify({"error": "no_active_session"}), 404
        csv_path = state.session_log.export_csv()
        return send_from_directory(csv_path.parent, csv_path.name,
                                   as_attachment=True)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard HTML
# ─────────────────────────────────────────────────────────────────────────────

def _dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OFNP Clinician Dashboard</title>
<style>
  :root{--bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;--text:#e2e8f0;--muted:#94a3b8;
        --green:#22c55e;--yellow:#eab308;--red:#ef4444;--blue:#3b82f6}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:ui-monospace,monospace;font-size:13px;padding:16px}
  h1{font-size:18px;font-weight:600;letter-spacing:0.05em;margin-bottom:4px}
  .sub{color:var(--muted);font-size:11px;margin-bottom:20px}
  .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px}
  .card h2{font-size:11px;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px}
  .val{font-size:22px;font-weight:700}
  .pill{display:inline-block;padding:3px 10px;border-radius:99px;font-size:11px;font-weight:600}
  .pill.ok{background:#14532d;color:#86efac}
  .pill.warn{background:#713f12;color:#fde68a}
  .pill.stop{background:#7f1d1d;color:#fca5a5}
  .row{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
  button{background:var(--card);border:1px solid var(--border);color:var(--text);
         padding:7px 16px;border-radius:6px;cursor:pointer;font-size:12px}
  button:hover{background:var(--border)}
  button.danger{background:#7f1d1d;border-color:#ef4444;color:#fca5a5}
  button.primary{background:#1e3a5f;border-color:#3b82f6;color:#93c5fd}
  #log{background:var(--card);border:1px solid var(--border);border-radius:8px;
       padding:12px;height:300px;overflow-y:auto;font-size:11px;line-height:1.6}
  .ev{padding:2px 0;border-bottom:1px solid #1e2130}
  .ev.trigger{color:#86efac}.ev.safety{color:#fca5a5}.ev.note{color:#93c5fd}
  .ev.expression{color:#fde68a}
  label{color:var(--muted);font-size:11px}
  input[type=number]{background:var(--card);border:1px solid var(--border);
    color:var(--text);border-radius:4px;padding:4px 8px;width:80px;font-size:12px}
  .disclaimer{background:#1c1010;border:1px solid #7f1d1d;border-radius:6px;
    padding:10px 14px;margin-bottom:16px;font-size:11px;color:#fca5a5;line-height:1.7}
</style>
</head>
<body>
<h1>OFNP Clinician Dashboard</h1>
<div class="sub">Open Facial Neuroprosthesis Project — MVP 0.1 · SIMULATION MODE</div>

<div class="disclaimer">
  ⚠ EXPERIMENTAL RESEARCH PROTOTYPE — NOT A CERTIFIED MEDICAL DEVICE<br>
  Use only under medical supervision · All stimulation parameters require clinical validation<br>
  Simulation mode active: no electrical output is generated
</div>

<div class="grid">
  <div class="card">
    <h2>System Status</h2>
    <div id="status-pill" class="pill ok">NOMINAL</div>
    <div style="margin-top:8px;color:var(--muted)">Session: <span id="session-id">—</span></div>
  </div>
  <div class="card">
    <h2>Triggers Detected</h2>
    <div class="val" id="trigger-count">0</div>
    <div style="color:var(--muted);font-size:11px;margin-top:4px">expressions this session</div>
  </div>
  <div class="card">
    <h2>Pipeline Latency</h2>
    <div class="val" id="latency">—<span style="font-size:13px;font-weight:400"> ms</span></div>
    <div style="color:var(--muted);font-size:11px;margin-top:4px">target &lt; 50ms</div>
  </div>
</div>

<div class="card" style="margin-bottom:12px">
  <h2 style="margin-bottom:10px">Session Control</h2>
  <div class="row">
    <button class="primary" onclick="startSession()">▶ Start Session</button>
    <button onclick="stopSession()">■ Stop Session</button>
    <button class="danger" onclick="emergencyStop()">⛔ EMERGENCY STOP</button>
  </div>
  <div class="row">
    <b style="color:var(--muted);font-size:11px;margin-top:4px">Inject Expression:</b>
    <button onclick="inject('resting')">Resting</button>
    <button onclick="inject('partial_smile')">Partial Smile</button>
    <button onclick="inject('full_smile')">Full Smile ☺</button>
    <button onclick="inject('cheek')">Cheek</button>
    <button onclick="inject('artifact')">Artifact ⚡</button>
  </div>
</div>

<div class="grid" style="margin-bottom:12px">
  <div class="card">
    <h2>Detection Thresholds</h2>
    <div style="display:flex;flex-direction:column;gap:6px;margin-top:6px">
      <div><label>Partial smile min (×baseline)</label><br>
           <input type="number" id="t-partial" value="2.5" step="0.1" min="1" onchange="updateThresholds()"></div>
      <div><label>Full smile min (×baseline)</label><br>
           <input type="number" id="t-full" value="4.5" step="0.1" min="1" onchange="updateThresholds()"></div>
      <div><label>Debounce windows</label><br>
           <input type="number" id="t-debounce" value="3" step="1" min="1" max="10" onchange="updateThresholds()"></div>
    </div>
  </div>
  <div class="card">
    <h2>Safety Limits (Simulation)</h2>
    <div style="color:var(--muted);font-size:11px;line-height:1.8">
      Max current: <b>5.0 mA</b><br>
      Max pulse width: <b>300 µs</b><br>
      Max frequency: <b>50 Hz</b><br>
      Impedance range: <b>0.5–30 kΩ</b><br>
      Session timeout: <b>60 min</b>
    </div>
  </div>
  <div class="card">
    <h2>Safety Events</h2>
    <div class="val" id="safety-count">0</div>
    <div style="color:var(--muted);font-size:11px;margin-top:4px">blocks this session</div>
    <div id="last-safety" style="color:#fca5a5;font-size:10px;margin-top:6px"></div>
  </div>
</div>

<div class="card">
  <h2 style="margin-bottom:8px">Live Event Log</h2>
  <div id="log"></div>
</div>

<script>
const log = document.getElementById('log');
let triggerCount = 0, safetyCount = 0;

async function startSession() {
  const r = await fetch('/api/session/start', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({patient_id:'demo_patient', clinician_id:'clinician_001'})
  });
  const d = await r.json();
  document.getElementById('session-id').textContent = d.session_id || '—';
  addLog('info', '▶ Session started: ' + (d.session_id || ''));
}

async function stopSession() {
  await fetch('/api/session/stop', {method:'POST'});
  addLog('info', '■ Session stopped');
}

async function emergencyStop() {
  await fetch('/api/emergency_stop', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({reason:'clinician_triggered'})
  });
  document.getElementById('status-pill').textContent = 'EMERGENCY STOP';
  document.getElementById('status-pill').className = 'pill stop';
  addLog('safety', '⛔ EMERGENCY STOP triggered');
}

async function inject(name) {
  await fetch('/api/scenario/' + name, {method:'POST'});
  addLog('note', '→ Scenario: ' + name);
}

async function updateThresholds() {
  const body = {
    partial_smile_min: +document.getElementById('t-partial').value,
    full_smile_min:    +document.getElementById('t-full').value,
    debounce_windows:  +document.getElementById('t-debounce').value,
  };
  await fetch('/api/thresholds', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
}

function addLog(type, msg) {
  const div = document.createElement('div');
  div.className = 'ev ' + type;
  const ts = new Date().toISOString().slice(11,23);
  div.textContent = ts + ' ' + msg;
  log.prepend(div);
  if (log.children.length > 200) log.removeChild(log.lastChild);
}

// SSE live events
const es = new EventSource('/api/events/stream');
es.onmessage = e => {
  const ev = JSON.parse(e.data);
  if (ev.type === 'stimulation_command') {
    triggerCount++;
    document.getElementById('trigger-count').textContent = triggerCount;
    addLog('trigger', '✓ TRIGGER ' + (ev.target_muscle||'') +
           ' intensity=' + (ev.peak_intensity_norm||0).toFixed(2) +
           ' latency=' + (ev.pipeline_latency_ms||0).toFixed(1) + 'ms');
  } else if (ev.type === 'safety_block') {
    safetyCount++;
    document.getElementById('safety-count').textContent = safetyCount;
    document.getElementById('last-safety').textContent = ev.reason || '';
    addLog('safety', '⚠ BLOCKED: ' + (ev.reason||''));
  } else if (ev.type === 'expression_estimate' && ev.expression !== 'neutral') {
    addLog('expression', '◉ ' + ev.expression +
           ' conf=' + (ev.confidence||0).toFixed(2) +
           ' rms=' + (ev.rms_uv||0).toFixed(1) + 'µV');
  }
};

// Poll status every 2s
setInterval(async () => {
  const r = await fetch('/api/status');
  const d = await r.json();
  const lat = d.stats?.mean_latency_ms;
  if (lat) document.getElementById('latency').innerHTML =
    lat.toFixed(1) + '<span style="font-size:13px;font-weight:400"> ms</span>';
  const safety = d.safety?.status;
  const pill = document.getElementById('status-pill');
  pill.textContent = safety?.toUpperCase() || 'NOMINAL';
  pill.className = 'pill ' + (safety === 'shutdown' ? 'stop' : safety === 'warning' ? 'warn' : 'ok');
}, 2000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5003
    app = create_app()
    print(f"\n  OFNP Clinician Dashboard → http://localhost:{port}")
    print("  ⚠  SIMULATION MODE — no electrical output\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
