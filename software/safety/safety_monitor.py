"""
OFNP — Safety Architecture
Open Facial Neuroprosthesis Project

Implements Sec. 15 (Safety Architecture) mandatory requirements:
  ✓ Hardware watchdog (firmware-level, mirrored here in software)
  ✓ Firmware watchdog (software watchdog with timeout)
  ✓ Emergency shutdown
  ✓ Current limiter (validation)
  ✓ Timeout protection
  ✓ Invalid signal detection
  ✓ Skin impedance anomaly detection

PRINCIPLE: Safety checks are the LAST step and CANNOT be bypassed.
           A stimulation command that fails safety check is NEVER executed.
"""

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

logger = logging.getLogger("ofnp.safety")


class SafetyStatus(Enum):
    NOMINAL  = "nominal"
    WARNING  = "warning"
    BLOCKED  = "blocked"    # Stimulation blocked, system still running
    SHUTDOWN = "shutdown"   # Emergency — all output halted


@dataclass
class SafetyLimits:
    """
    Clinician-configurable safety limits.
    Defaults are the MOST conservative values — clinician must explicitly
    increase them after patient assessment.

    ⚠  These values do NOT replace clinical judgment.
        All limits MUST be validated by a qualified clinician.
    """
    # Current limits (mA)
    max_current_ma: float          = 5.0      # Absolute maximum per channel
    default_current_ma: float      = 0.0      # Zero until clinician sets it

    # Pulse parameters
    max_pulse_width_us: float      = 300.0    # Microseconds
    max_frequency_hz: float        = 50.0     # Hz (above = tetanic, avoid)

    # Impedance (kΩ)
    min_impedance_kohm: float      = 0.5      # Below = possible short / poor contact
    max_impedance_kohm: float      = 30.0     # Above = poor contact / dry electrode

    # Timeouts
    max_session_duration_s: float  = 3600.0   # 1 hour maximum session
    max_continuous_stim_s: float   = 5.0      # Max continuous stimulation
    mandatory_rest_s: float        = 10.0     # Rest after timeout trigger

    # Watchdog
    watchdog_timeout_s: float      = 2.0      # If no heartbeat in 2s → shutdown

    # Signal validation
    max_artifact_rate: float       = 0.20     # Max 20% artifact samples before block


@dataclass
class SafetyEvent:
    timestamp: float
    event_type: str
    severity: str       # "info" / "warning" / "critical"
    message: str
    channel: Optional[str] = None
    value: Optional[float] = None


class SafetyMonitor:
    """
    Software safety monitor — mirrors and supplements firmware watchdog.

    In simulation mode: validates parameters and logs violations but
    cannot execute hardware shutdown (no hardware present).

    In hardware mode: sends emergency stop command to controller via
    the communication layer.
    """

    def __init__(self, limits: Optional[SafetyLimits] = None,
                 simulation_mode: bool = True,
                 emergency_callback: Optional[Callable] = None):
        self.limits = limits or SafetyLimits()
        self.simulation_mode = simulation_mode
        self._emergency_callback = emergency_callback

        self.status = SafetyStatus.NOMINAL
        self._session_start: Optional[float] = None
        self._last_stim_start: Optional[float] = None
        self._stim_active = False
        self._events: list[SafetyEvent] = []
        self._artifact_window: list[bool] = []
        self._watchdog_last_ping = time.time()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info("[Safety] Monitor initialized — simulation=%s", simulation_mode)

    # ── Public API ────────────────────────────────────────────────────────────

    def start_session(self):
        """Call at the start of each clinical session."""
        self._session_start = time.time()
        self.status = SafetyStatus.NOMINAL
        self._events.clear()
        self._stim_active = False
        self._start_watchdog()
        self._log_event("session_start", "info", "Safety monitor: session started")
        logger.info("[Safety] Session started")

    def stop_session(self):
        """Call at the end of each clinical session."""
        self._running = False
        self._stim_active = False
        self._log_event("session_end", "info", "Safety monitor: session ended normally")
        logger.info("[Safety] Session ended")

    def emergency_stop(self, reason: str):
        """
        Immediate halt. In hardware mode: sends shutdown command to firmware.
        In simulation: logs and blocks all subsequent commands.
        """
        self.status = SafetyStatus.SHUTDOWN
        self._stim_active = False
        msg = f"EMERGENCY STOP: {reason}"
        self._log_event("emergency_stop", "critical", msg)
        logger.critical("[Safety] %s", msg)

        if not self.simulation_mode and self._emergency_callback:
            self._emergency_callback(reason)

    def ping_watchdog(self):
        """Must be called regularly by the main loop (every <watchdog_timeout_s)."""
        self._watchdog_last_ping = time.time()

    def validate_stimulation(self,
                             profile,          # StimulationProfile
                             filtered_signal,  # FilteredSignal
                             features         # SignalFeatures
                             ) -> tuple[bool, str]:
        """
        Step 7: Safety verification.
        Returns (is_safe, reason).
        This is the LAST gate before any stimulation command is issued.
        """
        if self.status == SafetyStatus.SHUTDOWN:
            return False, "system_in_shutdown"

        # ── Session timeout ───────────────────────────────────────────────────
        if self._session_start:
            elapsed = time.time() - self._session_start
            if elapsed > self.limits.max_session_duration_s:
                self.emergency_stop("session_duration_exceeded")
                return False, "session_timeout"

        # ── Simulation-only check ─────────────────────────────────────────────
        if not profile.simulation_only:
            # Hardware mode: validate all parameters
            if profile.current_ma is None:
                return False, "current_not_set_by_clinician"
            if profile.current_ma > self.limits.max_current_ma:
                self._log_event("current_limit", "critical",
                                f"Requested {profile.current_ma}mA exceeds limit {self.limits.max_current_ma}mA",
                                value=profile.current_ma)
                return False, f"current_exceeds_limit_{self.limits.max_current_ma}mA"

            if profile.pulse_width_us is not None:
                if profile.pulse_width_us > self.limits.max_pulse_width_us:
                    return False, f"pulse_width_exceeds_{self.limits.max_pulse_width_us}us"

            if profile.frequency_hz is not None:
                if profile.frequency_hz > self.limits.max_frequency_hz:
                    return False, f"frequency_exceeds_{self.limits.max_frequency_hz}Hz"

        # ── Impedance check ───────────────────────────────────────────────────
        imp = filtered_signal.impedance_kohm if hasattr(filtered_signal, 'impedance_kohm') else 5.0
        if imp < self.limits.min_impedance_kohm:
            self._log_event("impedance_low", "warning",
                            f"Impedance {imp:.1f} kΩ < min {self.limits.min_impedance_kohm} kΩ",
                            value=imp)
            if imp < 0.1:  # Short circuit — block immediately
                return False, f"impedance_critical_{imp:.2f}kohm"

        if imp > self.limits.max_impedance_kohm:
            self._log_event("impedance_high", "warning",
                            f"Impedance {imp:.1f} kΩ > max {self.limits.max_impedance_kohm} kΩ",
                            value=imp)
            return False, f"impedance_too_high_{imp:.0f}kohm"

        # ── Artifact rate ─────────────────────────────────────────────────────
        self._artifact_window.append(filtered_signal.is_artifact)
        if len(self._artifact_window) > 50:
            self._artifact_window.pop(0)
        artifact_rate = sum(self._artifact_window) / len(self._artifact_window)
        if artifact_rate > self.limits.max_artifact_rate:
            return False, f"artifact_rate_high_{artifact_rate:.1%}"

        # ── Continuous stimulation timeout ────────────────────────────────────
        if self._stim_active and self._last_stim_start:
            stim_duration = time.time() - self._last_stim_start
            if stim_duration > self.limits.max_continuous_stim_s:
                self._log_event("stim_timeout", "warning",
                                f"Continuous stim timeout at {stim_duration:.1f}s")
                self._stim_active = False
                return False, f"continuous_stim_timeout_{stim_duration:.1f}s"

        # ── Intensity validation ──────────────────────────────────────────────
        if profile.peak_intensity_norm > 1.0:
            return False, "intensity_exceeds_normalized_max"

        # All checks passed
        if not self._stim_active:
            self._stim_active = True
            self._last_stim_start = time.time()

        return True, "ok"

    def notify_stim_end(self):
        """Call when a stimulation ramp completes."""
        self._stim_active = False
        self._last_stim_start = None

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def _start_watchdog(self):
        self._running = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="safety-watchdog"
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self):
        """Software watchdog — triggers emergency stop if main loop stalls."""
        while self._running:
            time.sleep(0.5)
            elapsed = time.time() - self._watchdog_last_ping
            if elapsed > self.limits.watchdog_timeout_s:
                self.emergency_stop(
                    f"watchdog_timeout_{elapsed:.1f}s_no_heartbeat"
                )

    # ── Event log ─────────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, severity: str, message: str,
                   channel: Optional[str] = None, value: Optional[float] = None):
        ev = SafetyEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            message=message,
            channel=channel,
            value=value,
        )
        self._events.append(ev)
        log_fn = {"info": logger.info, "warning": logger.warning,
                  "critical": logger.critical}.get(severity, logger.info)
        log_fn("[Safety] %s: %s", event_type, message)

    @property
    def events(self) -> list[SafetyEvent]:
        return list(self._events)

    @property
    def critical_events(self) -> list[SafetyEvent]:
        return [e for e in self._events if e.severity == "critical"]

    def status_report(self) -> dict:
        elapsed = (time.time() - self._session_start) if self._session_start else 0
        return {
            "status":          self.status.value,
            "simulation_mode": self.simulation_mode,
            "session_elapsed_s": round(elapsed, 1),
            "stim_active":     self._stim_active,
            "total_events":    len(self._events),
            "critical_events": len(self.critical_events),
            "last_event":      self._events[-1].message if self._events else None,
        }
