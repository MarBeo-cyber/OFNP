# 👁 Open Facial Neuroprosthesis Project (OFNP)

> **Non-invasive open-source assistive neuroprosthesis for facial paralysis — synchronized functional electrical stimulation driven by AI expression detection.**

[![CI](https://github.com/YOUR_ORG/open-facial-neuroprosthesis/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/open-facial-neuroprosthesis/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-0--2%20Simulation-yellow)](docs/architecture/)
[![Safety](https://img.shields.io/badge/Safety-First-red)](SAFETY_REQUIREMENTS.md)

---

## ⚠ Medical Disclaimer

**This is an experimental open-source biomedical research prototype.**

- NOT a certified medical device
- NOT for diagnosis, treatment, or therapy
- Use ONLY under medical supervision
- NOT for use in acute neurological conditions
- All stimulation parameters MUST be validated by a qualified clinician
- Current software version: **SIMULATION ONLY** — no electrical output

See [MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md) for full safety notice.

---

## What is OFNP?

OFNP is a research platform for facial neuroprosthetics targeting unilateral facial paralysis (Bell's palsy, congenital, post-traumatic, post-surgical). The system:

1. **Senses** voluntary movements on the healthy hemiface (EMG / IMU / CV)
2. **Detects** facial expressions (smile, cheek activation) via signal processing
3. **Generates** synchronized functional electrical stimulation (FES) patterns for the paretic side
4. **Adapts** stimulation intensity proportionally to the detected movement

Principle: *assistive augmentation, not forced normalization*.

---

## System Architecture

```
Healthy hemiface (sensing)          Paretic hemiface (stimulation)
         │                                      │
    EMG Electrodes                    FES Electrodes
         │                                      │
    ┌────▼────────────────────────────────────▼────┐
    │              Wearable Facial Patch             │
    └────────────────────┬───────────────────────────┘
                         │ ultrathin cable → behind ear
                    ┌────▼────┐
                    │ Belt    │  ← ESP32 + isolated stimulator
                    │  Unit   │     + battery + BLE
                    └────┬────┘
                         │ BLE
              ┌──────────▼──────────┐
              │  Clinician Desktop  │  ← Flask app + dashboard
              └─────────────────────┘
```

### Software Stack

| Module | Path | Function |
|--------|------|---------|
| Signal Pipeline | `software/signal_processing/pipeline.py` | 8-step EMG processing chain |
| EMG Simulator | `software/signal_processing/emg_simulator.py` | Synthetic EMG for Phase 1-2 |
| Safety Monitor | `software/safety/safety_monitor.py` | Mandatory safety gate + watchdog |
| Session Logger | `software/logging/session_logger.py` | Clinical audit trail (JSONL) |
| Clinician App | `software/desktop_app/app.py` | Flask dashboard + REST API |
| Firmware | `firmware/controller/ofnp_controller.ino` | ESP32 (C/Arduino) |

---

## Quick Start (Simulation)

```bash
# Clone
git clone https://github.com/YOUR_ORG/open-facial-neuroprosthesis.git
cd open-facial-neuroprosthesis

# Install
pip install -r requirements.txt

# Run clinician dashboard (simulation mode)
python software/desktop_app/app.py
# → http://localhost:5003

# Run tests
pytest tests/ -v
```

### Dashboard

Open `http://localhost:5003` in your browser. You can:
- Start/stop a simulated session
- Inject expressions (resting, partial smile, full smile, cheek, artifact)
- Monitor real-time trigger detection and safety events
- Adjust detection thresholds
- Trigger emergency stop
- Export session CSV

---

## Signal Processing Pipeline (8 Steps)

```
1. Acquisition    → ADC samples from EMG electrodes (1 kHz)
2. Filtering      → Bandpass 20–450 Hz + 50 Hz notch (Butterworth IIR)
3. Features       → RMS, MAV, ZCR, envelope (200ms window, 50ms hop)
4. Detection      → Threshold-based expression classifier (MVP)
5. Validation     → Debounce (150ms) + refractory period (500ms)
6. Stim profile   → Ramp-up / sustain / ramp-down proportional to intensity
7. Safety check   → MANDATORY last gate — cannot be bypassed
8. Logging        → JSONL audit trail, all events recorded
```

---

## Development Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Done | Documentation and safety framework |
| 1 | ✅ Done | Signal processing + EMG simulation |
| 2 | ✅ Done | Desktop app + full simulation (no output) |
| 3 | 🔲 Next | Bench electrical testing (no human subjects) |
| 4 | 🔲 Future | Single-subject supervised prototype |
| 5 | 🔲 Future | Clinician-guided pilot |

---

## Safety Architecture

All safety checks are mandatory and cannot be bypassed:

```
Hardware watchdog  → ESP32 hardware timer (3s timeout → immediate shutdown)
Software watchdog  → Safety monitor ping (2s timeout → session abort)
Current limiter    → Hard limit 5mA (adjustable down, not up without review)
Impedance monitor  → Block if <0.5 kΩ or >30 kΩ
Continuous timeout → Max 5s continuous stimulation
Emergency stop     → Physical button + BLE command + software trigger
Session timeout    → Max 60 minutes per session
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Priority areas:
- [ ] Real ADS1115 ADC driver (replacing simulator)
- [ ] OpenFace 2 integration for computer vision sensing modality
- [ ] Adaptive threshold calibration (automatic baseline tracking)
- [ ] AR-assisted electrode positioning guide
- [ ] Clinician cloud dashboard (de-identified session analytics)
- [ ] Advanced AI trigger (after Phase 3 safety validation)

**Contributors with clinical experience (neurology, rehabilitation medicine, physiotherapy) are especially welcome.**

---

## Licenses

- Software: [Apache 2.0](LICENSE)
- Hardware: [CERN Open Hardware License v2](hardware/LICENSE)
- Documentation: [CC-BY 4.0](docs/LICENSE)

---

*Experimental biomedical research platform · Safety before performance*
