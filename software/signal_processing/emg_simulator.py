"""
OFNP — EMG Sensor Simulator
Open Facial Neuroprosthesis Project

Generates synthetic EMG signals for Phase 1 (passive sensing) and
Phase 2 (simulation without electrical output) development and testing.

Simulates realistic EMG characteristics for facial muscles:
  - Resting baseline: 1–5 µV RMS
  - Zygomaticus major (smile): 20–150 µV RMS during activation
  - Orbicularis oris: 15–100 µV RMS
  - Muscle-specific activation onset, peak, and decay profiles
  - Realistic noise: power line, movement artifact, electrode drift

Reference: surface EMG characteristics from Fridlund & Cacioppo (1986),
           Boxtel (2010) facial EMG normative values.
"""

from __future__ import annotations

import time
import math
import random
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from software.signal_processing.pipeline import RawSample

logger = logging.getLogger("ofnp.sensor_sim")


class SimulatedExpression(Enum):
    RESTING          = "resting"
    PARTIAL_SMILE    = "partial_smile"
    FULL_SMILE       = "full_smile"
    CHEEK_ACTIVATION = "cheek_activation"
    MOVEMENT_ARTIFACT = "movement_artifact"


@dataclass
class MuscleProfile:
    """Normative EMG amplitude profile for a facial muscle."""
    name: str
    baseline_rms_uv: float       # Typical resting RMS
    baseline_noise_uv: float     # Noise floor at rest
    smile_rms_uv: float          # RMS during smile
    cheek_rms_uv: float          # RMS during cheek puff
    peak_uv: float               # Peak amplitude during max effort
    onset_ms: float              # Rise time to activation
    decay_ms: float              # Fall time after activation


MUSCLE_PROFILES = {
    "R_zyg": MuscleProfile(   # Right zygomaticus major (sensing)
        name="Zygomaticus Major (R)",
        baseline_rms_uv=2.5,
        baseline_noise_uv=1.5,
        smile_rms_uv=45.0,
        cheek_rms_uv=20.0,
        peak_uv=180.0,
        onset_ms=80.0,
        decay_ms=150.0,
    ),
    "R_cheek": MuscleProfile(  # Right cheek / risorius (sensing)
        name="Risorius (R)",
        baseline_rms_uv=1.8,
        baseline_noise_uv=1.2,
        smile_rms_uv=28.0,
        cheek_rms_uv=35.0,
        peak_uv=120.0,
        onset_ms=100.0,
        decay_ms=180.0,
    ),
}


class EMGSimulator:
    """
    Synthetic EMG signal generator for one or more facial muscle channels.
    Generates 1 sample at a time, suitable for feeding into the real pipeline.
    """

    SAMPLE_RATE_HZ = 1000
    POWER_LINE_HZ  = 50.0      # EU standard

    def __init__(self, channels: Optional[list[str]] = None,
                 sample_rate_hz: float = 1000.0):
        self.fs = sample_rate_hz
        self.channels = channels or list(MUSCLE_PROFILES.keys())
        self._t = 0.0            # Simulation time (seconds)
        self._dt = 1.0 / self.fs
        self._expression = SimulatedExpression.RESTING
        self._expression_onset = 0.0   # When current expression started

        # Electrode drift (slow random walk)
        self._baseline_drift: dict[str, float] = {ch: 0.0 for ch in self.channels}
        # Impedance (random per channel, stable within session)
        self._impedance: dict[str, float] = {
            ch: random.uniform(2.0, 15.0) for ch in self.channels
        }

        logger.info("[EMGSim] Initialized channels: %s", self.channels)

    def set_expression(self, expression: SimulatedExpression):
        """Inject a simulated facial expression."""
        self._expression = expression
        self._expression_onset = self._t
        logger.debug("[EMGSim] Expression → %s", expression.value)

    def next_samples(self) -> list[RawSample]:
        """Generate one sample per channel at current time step."""
        self._t += self._dt
        samples = []

        for ch in self.channels:
            profile = MUSCLE_PROFILES.get(ch)
            if not profile:
                continue

            voltage_uv = self._synthesise_emg(ch, profile)
            # Electrode drift (slow EMA random walk, ±10µV over minutes)
            self._baseline_drift[ch] += random.gauss(0, 0.01)
            self._baseline_drift[ch] *= 0.999  # Drift decay

            # Impedance slow variation (±0.5 kΩ per minute)
            self._impedance[ch] += random.gauss(0, 0.001)
            self._impedance[ch] = max(0.5, min(45.0, self._impedance[ch]))

            samples.append(RawSample(
                timestamp=time.time(),
                channel_id=ch,
                voltage_uv=voltage_uv + self._baseline_drift[ch],
                impedance_kohm=self._impedance[ch],
            ))

        return samples

    def _synthesise_emg(self, ch: str, profile: MuscleProfile) -> float:
        """Generate one EMG sample for one channel."""
        # Baseline noise (Gaussian)
        noise = random.gauss(0, profile.baseline_noise_uv)

        # Power line interference (50Hz, ±1µV — realistic for facial EMG)
        power_line = 0.8 * math.sin(2 * math.pi * self.POWER_LINE_HZ * self._t)

        # Movement artifact (rare random spikes)
        artifact = 0.0
        if self._expression == SimulatedExpression.MOVEMENT_ARTIFACT:
            artifact = random.gauss(0, 80.0)
        elif random.random() < 0.001:  # 0.1% random artifact
            artifact = random.gauss(0, 30.0)

        # Muscle activation signal
        activation_uv = self._compute_activation(ch, profile)

        # Total signal (bandlimited Gaussian burst for EMG character)
        emg = activation_uv * random.gauss(1.0, 0.3) + noise + power_line + artifact

        return emg

    def _compute_activation(self, ch: str, profile: MuscleProfile) -> float:
        """Compute activation amplitude based on current expression and time."""
        expr = self._expression
        elapsed_ms = (self._t - self._expression_onset) * 1000.0

        if expr == SimulatedExpression.RESTING:
            return 0.0

        # Determine target amplitude
        if expr == SimulatedExpression.FULL_SMILE:
            target_rms = profile.smile_rms_uv
        elif expr == SimulatedExpression.PARTIAL_SMILE:
            target_rms = profile.smile_rms_uv * 0.4
        elif expr == SimulatedExpression.CHEEK_ACTIVATION:
            target_rms = profile.cheek_rms_uv
        else:
            return 0.0

        # Ramp-up envelope
        if elapsed_ms < profile.onset_ms:
            envelope = elapsed_ms / profile.onset_ms
        else:
            envelope = 1.0   # Sustained

        # Convert RMS target to sample amplitude (EMG is approximately Gaussian)
        return target_rms * envelope * random.gauss(0, math.sqrt(2))


class ScenarioRunner:
    """
    Runs a predefined expression scenario for Phase 2 testing.
    Generates a sequence of expressions with timing, feeds into pipeline.
    """

    SCENARIOS = {
        "rest_only": [
            (SimulatedExpression.RESTING, 5.0),
        ],
        "smile_sequence": [
            (SimulatedExpression.RESTING,       3.0),
            (SimulatedExpression.PARTIAL_SMILE, 1.5),
            (SimulatedExpression.RESTING,       2.0),
            (SimulatedExpression.FULL_SMILE,    2.0),
            (SimulatedExpression.RESTING,       2.0),
            (SimulatedExpression.CHEEK_ACTIVATION, 1.5),
            (SimulatedExpression.RESTING,       3.0),
        ],
        "artifact_stress": [
            (SimulatedExpression.RESTING,          2.0),
            (SimulatedExpression.MOVEMENT_ARTIFACT, 0.5),
            (SimulatedExpression.RESTING,          1.0),
            (SimulatedExpression.FULL_SMILE,       1.5),
            (SimulatedExpression.MOVEMENT_ARTIFACT, 0.3),
            (SimulatedExpression.RESTING,          2.0),
        ],
    }

    def __init__(self, scenario_name: str = "smile_sequence"):
        self.simulator = EMGSimulator()
        self.scenario = self.SCENARIOS.get(scenario_name, self.SCENARIOS["smile_sequence"])
        logger.info("[ScenarioRunner] Scenario: %s", scenario_name)

    def run(self, pipeline, on_stim_profile=None):
        """
        Execute the scenario, feeding each sample into the provided pipeline.
        Calls on_stim_profile(profile) when a stimulation command is generated.
        """
        total_samples = 0
        triggered_count = 0

        for expression, duration_s in self.scenario:
            self.simulator.set_expression(expression)
            n_samples = int(duration_s * self.simulator.fs)

            for _ in range(n_samples):
                samples = self.simulator.next_samples()
                for sample in samples:
                    result = pipeline.process_sample(sample)
                    total_samples += 1
                    if result:
                        triggered_count += 1
                        if on_stim_profile:
                            on_stim_profile(result)

                # Real-time pacing (optional — remove for faster simulation)
                # time.sleep(1.0 / self.simulator.fs)

        logger.info("[ScenarioRunner] Done: %d samples, %d triggers",
                    total_samples, triggered_count)
        return total_samples, triggered_count
