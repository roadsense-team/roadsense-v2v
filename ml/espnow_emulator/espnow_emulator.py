"""
ESP-NOW Communication Emulator

Emulates ESP-NOW communication effects (latency, packet loss, jitter)
based on measurements from real ESP32 hardware. Designed for Sim2Real
transfer in reinforcement learning training pipelines.

Author: Amir Khalifa
Date: December 26, 2025
Version: 1.0.0

Physics Models
==============

This emulator implements the following mathematical models based on
empirical measurements from ESP32 hardware:

**Latency Model**::

    L = max(1.0, L_base + (d × F_dist) + N(0, σ_jitter))

Where:
    - L: One-way latency in milliseconds
    - L_base: Base latency (typically 10-20ms for ESP-NOW)
    - d: Distance between sender and receiver in meters
    - F_dist: Distance factor (typically 0.1 ms/m)
    - N(0, σ): Gaussian jitter with std deviation σ
    - Minimum clamped to 1.0ms (physical limit)

**Packet Loss Model (3-Tier Distance-Dependent)**::

    P(loss) = {
        base_rate,                              if d < T₁
        base + (d-T₁)/(T₂-T₁) × (tier₂-base),  if T₁ ≤ d < T₂
        tier₃,                                  if d ≥ T₂
    }

Where:
    - T₁: First distance threshold (typically 50m)
    - T₂: Second distance threshold (typically 100m)
    - base_rate: Loss rate at close range (typically 2%)
    - tier₂: Loss rate at mid-range (typically 15%)
    - tier₃: Loss rate at far range (typically 40%)

**Burst Loss Model**::

    P(burst_continue) = 1 - (1 / mean_burst_length)

During a burst, loss probability is multiplied by burst_multiplier
(default 3x) and capped at max_loss_cap (default 80%).

**Sensor Noise Model**::

    GPS:     pos_noisy = pos + N(0, σ_gps / 111000)  # meters → degrees
    Speed:   v_noisy = max(0, v + N(0, σ_speed))
    Heading: θ_noisy = (θ + N(0, σ_heading)) mod 360
    Accel:   a_noisy = a + N(0, σ_accel)
    Gyro:    ω_noisy = ω + N(0, σ_gyro)

**Domain Randomization**

For robust Sim2Real transfer, the emulator supports randomizing
parameters per episode within specified ranges. This trains agents
to handle variations in communication quality.

Randomized parameters:
    - Latency base (e.g., 10-80ms range)
    - Packet loss base rate (e.g., 0-15% range)
    - Jitter std deviation (e.g., 5-12ms range)
    - GPS noise std (e.g., 2-8m range)

**Causality Enforcement**

Messages are queued and only delivered when their arrival time
(send_time + latency) is reached. This prevents the RL agent from
observing future information, which is critical for Sim2Real validity.

Example Usage
=============

Basic transmission with communication effects::

    from espnow_emulator import ESPNOWEmulator, V2VMessage

    # Initialize emulator
    emulator = ESPNOWEmulator(domain_randomization=True, seed=42)

    # Create message (perfect SUMO data)
    msg = V2VMessage(
        vehicle_id='V002', lat=32.0, lon=34.0, speed=15.0,
        heading=90.0, accel_x=0.5, accel_y=0.0, accel_z=9.81,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.1, timestamp_ms=1000
    )

    # Transmit with ESP-NOW effects
    result = emulator.transmit(
        sender_msg=msg,
        sender_pos=(25.0, 0.0),  # V002 is 25m ahead
        receiver_pos=(0.0, 0.0),  # V001 at origin
        current_time_ms=1000
    )

    if result:
        print(f"Message queued, will arrive at {result.received_at_ms}ms")
    else:
        print("Packet lost!")

    # Get observation (messages delivered based on current time)
    obs = emulator.get_observation(ego_speed=10.0, current_time_ms=1100)
    if obs['v002_valid']:
        print(f"V002 at ({obs['v002_lat']}, {obs['v002_lon']})")
        print(f"Message age: {obs['v002_age_ms']}ms")
"""

import json
import random
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from queue import PriorityQueue


@dataclass(frozen=True)
class V2VMessage:
    """
    Vehicle-to-Vehicle message (matches ESP32 V2VMessage struct).

    This dataclass represents the message content transmitted between vehicles.
    All fields match the C struct on ESP32 for consistency.

    Attributes:
        vehicle_id: Unique vehicle identifier (e.g., "V001", "V002")
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        speed: Speed in m/s
        heading: Heading in degrees (0-360)
        accel_x: Longitudinal acceleration in m/s²
        accel_y: Lateral acceleration in m/s²
        accel_z: Vertical acceleration in m/s² (typically ~9.81)
        gyro_x: Roll rate in rad/s
        gyro_y: Pitch rate in rad/s
        gyro_z: Yaw rate in rad/s
        timestamp_ms: Sender's timestamp in milliseconds
    """
    vehicle_id: str
    lat: float
    lon: float
    speed: float
    heading: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    timestamp_ms: int


@dataclass(frozen=True)
class ReceivedMessage:
    """
    Message as received by ego vehicle (with communication effects applied).

    This represents what the receiving vehicle actually sees after
    ESP-NOW transmission, including latency and age metadata.

    Attributes:
        message: The V2VMessage content
        age_ms: Communication latency in milliseconds
        received_at_ms: Receiver's timestamp when message was received
    """
    message: V2VMessage
    age_ms: int
    received_at_ms: int


class ESPNOWEmulator:
    """
    Emulates ESP-NOW communication effects on perfect SUMO data.

    Uses measured parameters from real ESP32 characterization + domain randomization
    to provide realistic communication effects for RL training.

    The emulator models:
    - Latency: Base latency + distance factor + jitter
    - Packet loss: Distance-dependent probability + optional burst patterns
    - Sensor noise: GPS, speed, acceleration noise (optional)
    - Domain randomization: Randomized parameters per episode for robustness

    Example usage:
        >>> emulator = ESPNOWEmulator(params_file='emulator_params.json')
        >>> msg = V2VMessage(vehicle_id='V002', lat=32.0, lon=34.0, ...)
        >>> received = emulator.transmit(msg, (10, 5), (0, 0), 1000)
        >>> if received:
        ...     print(f"Latency: {received.age_ms}ms")
        ... else:
        ...     print("Packet lost")
    """

    # GPS conversion constant (approximately 111km per degree of latitude)
    # NOTE: Longitude scaling varies by latitude (~111km * cos(lat))
    # This is an approximation for mid-latitudes. For Tel Aviv (~32°N), actual is ~94km/deg.
    METERS_PER_DEG_LAT = 111000.0

    def __init__(self, params_file: str = None, domain_randomization: bool = True, seed: int = None):
        """
        Initialize emulator with measured parameters.

        Args:
            params_file: Path to emulator_params.json from characterization.
                        If None, uses conservative default parameters.
            domain_randomization: If True, randomize parameters per episode
                                 for robust training (train wider than reality).
            seed: Optional random seed for reproducibility. If provided,
                  initializes the emulator's private RNG with this seed.

        Raises:
            FileNotFoundError: If params_file specified but doesn't exist
            json.JSONDecodeError: If params_file contains invalid JSON
            ValueError: If parameters fail validation (invalid ranges, logic)
        """
        self.domain_randomization = domain_randomization

        # FIXED: Isolate RNG to prevent global state pollution
        # Use private random.Random() instance to avoid interference with
        # RL training loop's global random state
        self._rng = random.Random(seed)

        # Step 1: Start with default parameters (complete, valid configuration)
        self.params = self._default_params()

        # Step 2: If params_file provided, load and recursively merge
        if params_file:
            with open(params_file, 'r') as f:
                user_params = json.load(f)
            # Recursively update defaults with user-provided values
            self._recursive_update(self.params, user_params)

        # Step 3: Validate final merged parameters
        self._validate_params()

        # State tracking - FIXED: Use event queue for causality
        self.pending_messages: PriorityQueue = PriorityQueue()  # Messages awaiting delivery
        self.last_received: Dict[str, ReceivedMessage] = {}  # Actually delivered messages
        self.last_loss_state: Dict[str, bool] = {}  # Track burst loss per vehicle
        self.current_time_ms = 0
        self.msg_counter = 0  # Monotonic counter to break ties in PriorityQueue

        # Per-episode randomized parameters (if domain_randomization enabled)
        self._randomize_episode_params()

    def _default_params(self) -> dict:
        """
        Default parameters (conservative estimates).

        These are used when no measured parameters file is provided.
        Values are conservative to avoid unrealistic optimism.

        Returns:
            Dictionary with default emulator parameters
        """
        return {
            'latency': {
                'base_ms': 15,
                'distance_factor': 0.1,  # ms per meter
                'jitter_std_ms': 8,
            },
            'packet_loss': {
                'base_rate': 0.02,
                'distance_threshold_1': 50,
                'distance_threshold_2': 100,
                'rate_tier_1': 0.05,
                'rate_tier_2': 0.15,
                'rate_tier_3': 0.40,
            },
            'burst_loss': {
                'enabled': False,
                'mean_burst_length': 1,
                'loss_multiplier': 3,  # Multiply loss probability during burst
                'max_loss_cap': 0.8,   # Maximum loss probability (80%)
            },
            'sensor_noise': {
                'gps_std_m': 5.0,
                'speed_std_ms': 0.5,
                'accel_std_ms2': 0.2,
                'heading_std_deg': 3.0,
                'gyro_std_rad_s': 0.01,
            },
            'observation': {
                'staleness_threshold_ms': 500,  # Messages older than this are marked invalid
                'monitored_vehicles': ['V002', 'V003'],  # Vehicles to include in observation
            },
            'domain_randomization': {
                'latency_range_ms': [10, 80],
                'loss_rate_range': [0.0, 0.15],
                # FIXED (Phase 8 Review): Add variance parameter ranges
                'jitter_std_range_ms': [5, 12],  # Jitter std can vary 5-12ms
                'gps_noise_range_m': [2.0, 8.0],  # GPS noise std can vary 2-8m
            }
        }

    @staticmethod
    def _recursive_update(base: dict, update: dict) -> dict:
        """
        Recursively update base dictionary with values from update dictionary.

        This implements the "deep merge" strategy: nested dictionaries are
        merged recursively rather than replaced entirely. This allows partial
        configuration files to override only specific parameters while keeping
        others at default values.

        Example:
            base = {"latency": {"base_ms": 15, "jitter_std_ms": 8}}
            update = {"latency": {"base_ms": 999}}
            result = {"latency": {"base_ms": 999, "jitter_std_ms": 8}}

        Args:
            base: Base dictionary (typically defaults)
            update: Update dictionary (typically user-provided overrides)

        Returns:
            Updated base dictionary (modified in-place, also returned)
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                # Recursive case: both are dicts, merge recursively
                ESPNOWEmulator._recursive_update(base[key], value)
            else:
                # Base case: replace value (works for primitives, lists, or new keys)
                base[key] = value
        return base

    def _validate_params(self):
        """
        Validate loaded parameters for correctness.

        Checks:
        1. Type validity: numbers are numbers, lists are lists
        2. Range constraints: probabilities in [0, 1], latencies > 0, etc.
        3. Logical consistency: threshold_1 < threshold_2, etc.

        Raises:
            ValueError: If any validation check fails with descriptive message
        """
        # Validate latency parameters
        if 'latency' in self.params:
            lat = self.params['latency']

            if 'base_ms' in lat and lat['base_ms'] < 0:
                raise ValueError("Latency base_ms must be positive (cannot have negative latency)")

            if 'distance_factor' in lat and lat['distance_factor'] < 0:
                raise ValueError("Latency distance_factor must be non-negative")

            if 'jitter_std_ms' in lat and lat['jitter_std_ms'] < 0:
                raise ValueError("Latency jitter_std_ms must be non-negative")

        # Validate packet loss parameters
        if 'packet_loss' in self.params:
            pl = self.params['packet_loss']

            # Check probability ranges [0, 1]
            for key in ['base_rate', 'rate_tier_1', 'rate_tier_2', 'rate_tier_3']:
                if key in pl:
                    if not (0.0 <= pl[key] <= 1.0):
                        raise ValueError(
                            f"Probability packet_loss.{key} must be between 0 and 1, got {pl[key]}"
                        )

            # Check distance thresholds are positive
            if 'distance_threshold_1' in pl and pl['distance_threshold_1'] < 0:
                raise ValueError("Distance threshold_1 must be non-negative")

            if 'distance_threshold_2' in pl and pl['distance_threshold_2'] < 0:
                raise ValueError("Distance threshold_2 must be non-negative")

            # Check logical constraint: threshold_1 < threshold_2
            if 'distance_threshold_1' in pl and 'distance_threshold_2' in pl:
                if pl['distance_threshold_1'] >= pl['distance_threshold_2']:
                    raise ValueError(
                        f"distance_threshold_1 ({pl['distance_threshold_1']}) must be less than "
                        f"distance_threshold_2 ({pl['distance_threshold_2']})"
                    )

        # Validate sensor noise parameters
        if 'sensor_noise' in self.params:
            sn = self.params['sensor_noise']

            for key in ['gps_std_m', 'speed_std_ms', 'accel_std_ms2', 'heading_std_deg']:
                if key in sn and sn[key] < 0:
                    raise ValueError(f"Sensor noise {key} must be non-negative")

        # Validate domain randomization ranges
        if 'domain_randomization' in self.params:
            dr = self.params['domain_randomization']

            if 'latency_range_ms' in dr:
                lat_range = dr['latency_range_ms']
                if not isinstance(lat_range, list) or len(lat_range) != 2:
                    raise ValueError("domain_randomization.latency_range_ms must be a list of 2 values")
                # FIXED (Phase 8 Review): Prevent "Frozen Simulator" bug
                if lat_range[0] <= 0:
                    raise ValueError(
                        f"domain_randomization.latency_range_ms[0] must be > 0 (got {lat_range[0]}). "
                        "Zero latency would freeze the simulator (time stops advancing)."
                    )
                if lat_range[1] <= 0:
                    raise ValueError("domain_randomization.latency_range_ms[1] must be positive")
                if lat_range[0] > lat_range[1]:
                    raise ValueError("domain_randomization.latency_range_ms[0] must be <= latency_range_ms[1]")

            if 'loss_rate_range' in dr:
                loss_range = dr['loss_rate_range']
                if not isinstance(loss_range, list) or len(loss_range) != 2:
                    raise ValueError("domain_randomization.loss_rate_range must be a list of 2 values")
                if not (0.0 <= loss_range[0] <= 1.0) or not (0.0 <= loss_range[1] <= 1.0):
                    raise ValueError("domain_randomization.loss_rate_range values must be in [0, 1]")
                if loss_range[0] > loss_range[1]:
                    raise ValueError("domain_randomization.loss_rate_range[0] must be <= loss_rate_range[1]")

    def _randomize_episode_params(self):
        """
        Randomize parameters for this episode (domain randomization).

        FIXED (Phase 8 Review): Now randomizes VARIANCE parameters, not just means.
        - Latency: base_ms AND jitter_std_ms
        - Loss: base_rate (already implemented)
        - Sensor noise: gps_std_m (configurable)

        If domain randomization is enabled, this method randomizes latency
        and packet loss parameters within specified ranges. This trains the
        agent to be robust to variations in communication quality.

        If disabled, uses base parameters from measurement.
        """
        if not self.domain_randomization:
            self.episode_latency_base = self.params['latency']['base_ms']
            self.episode_jitter_std = self.params['latency']['jitter_std_ms']
            self.episode_loss_base = self.params['packet_loss']['base_rate']
            self.episode_gps_noise_std = self.params['sensor_noise']['gps_std_m']
            return

        # Randomize within domain randomization ranges
        dr = self.params['domain_randomization']

        # FIXED: Use private RNG (not global random)
        self.episode_latency_base = self._rng.uniform(
            dr['latency_range_ms'][0],
            dr['latency_range_ms'][1]
        )

        self.episode_loss_base = self._rng.uniform(
            dr['loss_rate_range'][0],
            dr['loss_rate_range'][1]
        )

        # FIXED (Phase 8 Review): Randomize variance parameters
        # Jitter std: randomize within configured range
        if 'jitter_std_range_ms' in dr:
            self.episode_jitter_std = self._rng.uniform(
                dr['jitter_std_range_ms'][0],
                dr['jitter_std_range_ms'][1]
            )
        else:
            # Fallback: use default if range not configured
            self.episode_jitter_std = self.params['latency']['jitter_std_ms']

        # GPS noise std: randomize within configured range
        if 'gps_noise_range_m' in dr:
            self.episode_gps_noise_std = self._rng.uniform(
                dr['gps_noise_range_m'][0],
                dr['gps_noise_range_m'][1]
            )
        else:
            # Fallback: use default if range not configured
            self.episode_gps_noise_std = self.params['sensor_noise']['gps_std_m']

    def reset(self, seed: int = None):
        """
        Reset emulator state for new episode.

        Clears all received message history and re-randomizes episode parameters.
        Call this at the start of each new training episode.

        Args:
            seed: Optional random seed for reproducibility. If provided,
                  re-seeds the emulator's private RNG before randomizing
                  episode parameters.
        """
        # FIXED: Allow seeding on reset for reproducible episodes
        if seed is not None:
            self._rng.seed(seed)

        self.pending_messages = PriorityQueue()  # Clear pending messages
        self.last_received = {}
        self.last_loss_state = {}  # Clear burst loss tracking
        self.current_time_ms = 0
        self.msg_counter = 0  # Reset sequence counter
        self._randomize_episode_params()

    def _calculate_distance(self, pos1: Tuple[float, float],
                           pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two positions.

        Args:
            pos1: (x, y) position in meters
            pos2: (x, y) position in meters

        Returns:
            Distance in meters
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _get_loss_probability(self, distance_m: float) -> float:
        """
        Get packet loss probability based on distance.

        Uses tiered model based on measured data:
        - Close range (<threshold_1): base_rate
        - Medium range: linear interpolation
        - Far range (>threshold_2): high loss rate

        Args:
            distance_m: Distance between sender and receiver in meters

        Returns:
            Packet loss probability (0.0 to 1.0)
        """
        pl = self.params['packet_loss']

        base = self.episode_loss_base if self.domain_randomization else pl['base_rate']

        if distance_m < pl['distance_threshold_1']:
            return base
        elif distance_m < pl['distance_threshold_2']:
            # Linear interpolation between thresholds
            t = (distance_m - pl['distance_threshold_1']) / \
                (pl['distance_threshold_2'] - pl['distance_threshold_1'])
            return base + t * (pl['rate_tier_2'] - base)
        else:
            # Beyond threshold 2
            return pl['rate_tier_3']

    def _get_latency(self, distance_m: float) -> float:
        """
        Get one-way latency based on distance.

        Model: latency = base + (distance * factor) + jitter
        Where jitter is Gaussian noise with measured std dev.

        FIXED (Phase 8 Review): Uses episode_jitter_std (can be randomized)
        and private RNG instance.

        Args:
            distance_m: Distance between sender and receiver in meters

        Returns:
            One-way latency in milliseconds (always >= 1ms)
        """
        lat = self.params['latency']

        base = self.episode_latency_base if self.domain_randomization else lat['base_ms']
        distance_component = distance_m * lat['distance_factor']

        # FIXED: Use episode_jitter_std (can be randomized) and private RNG
        jitter_std = self.episode_jitter_std if self.domain_randomization else lat['jitter_std_ms']
        jitter = self._rng.gauss(0, jitter_std)

        return max(1.0, base + distance_component + jitter)

    def _add_sensor_noise(self, msg: V2VMessage) -> V2VMessage:
        """
        Add realistic sensor noise to message (OPTIONAL).

        This is optional for minimal RL observation approach.
        Can be skipped if only using position/speed/accel features.

        FIXED (Phase 8 Review): Uses episode_gps_noise_std (can be randomized)
        and private RNG instance.

        Args:
            msg: Original V2VMessage

        Returns:
            New V2VMessage with noise applied
        """
        noise = self.params['sensor_noise']

        # FIXED: Use episode_gps_noise_std if DR enabled
        gps_noise_std = self.episode_gps_noise_std if self.domain_randomization else noise['gps_std_m']

        # Create copy with noise (using private RNG)
        noisy = V2VMessage(
            vehicle_id=msg.vehicle_id,
            lat=msg.lat + self._rng.gauss(0, gps_noise_std / self.METERS_PER_DEG_LAT),
            lon=msg.lon + self._rng.gauss(0, gps_noise_std / self.METERS_PER_DEG_LAT),
            speed=max(0, msg.speed + self._rng.gauss(0, noise['speed_std_ms'])),
            heading=(msg.heading + self._rng.gauss(0, noise['heading_std_deg'])) % 360,
            accel_x=msg.accel_x + self._rng.gauss(0, noise['accel_std_ms2']),
            accel_y=msg.accel_y + self._rng.gauss(0, noise['accel_std_ms2']),
            accel_z=msg.accel_z + self._rng.gauss(0, noise['accel_std_ms2']),
            gyro_x=msg.gyro_x + self._rng.gauss(0, noise.get('gyro_std_rad_s', 0.01)),
            gyro_y=msg.gyro_y + self._rng.gauss(0, noise.get('gyro_std_rad_s', 0.01)),
            gyro_z=msg.gyro_z + self._rng.gauss(0, noise.get('gyro_std_rad_s', 0.01)),
            timestamp_ms=msg.timestamp_ms,
        )

        return noisy

    def _check_burst_loss(self, vehicle_id: str) -> bool:
        """
        Check if we're in a burst loss period.

        If burst loss is enabled and last packet was lost, there's a higher
        probability this one will be lost too (bursty behavior).

        The continuation probability is derived from mean_burst_length:
            p_continue = 1 - (1 / mean_burst_length)

        Examples:
            mean_burst_length = 1 → p_continue = 0 (no bursts)
            mean_burst_length = 2 → p_continue = 0.5 (50% chance to continue)
            mean_burst_length = 3 → p_continue = 0.66 (66% chance)

        Args:
            vehicle_id: Vehicle ID to check

        Returns:
            True if in burst loss period, False otherwise
        """
        if not self.params['burst_loss']['enabled']:
            return False

        # FIXED: Calculate burst continuation probability from mean_burst_length
        mean_burst_length = self.params['burst_loss']['mean_burst_length']
        if mean_burst_length <= 1:
            # mean_burst_length of 1 means no burst continuation
            return False

        # p_continue = 1 - (1 / mean_burst_length)
        p_continue = 1.0 - (1.0 / mean_burst_length)

        # Check if last packet was lost and if we continue the burst
        # FIXED: Use private RNG
        if self.last_loss_state.get(vehicle_id, False):
            return self._rng.random() < p_continue
        return False

    def transmit(self,
                 sender_msg: V2VMessage,
                 sender_pos: Tuple[float, float],
                 receiver_pos: Tuple[float, float],
                 current_time_ms: int) -> Optional[ReceivedMessage]:
        """
        Simulate transmission of a V2V message through ESP-NOW.

        Applies realistic communication effects:
        - Distance calculation
        - Packet loss (distance-dependent + burst patterns)
        - Latency (base + distance + jitter)
        - Optional sensor noise

        **Important**: Messages are queued for future delivery based on
        latency. They only become visible in `get_observation()` when
        `current_time_ms >= arrival_time`. This enforces causality.

        Args:
            sender_msg: The message being sent (perfect SUMO data)
            sender_pos: Sender position (x, y) in meters
            receiver_pos: Receiver position (x, y) in meters
            current_time_ms: Current simulation time in milliseconds

        Returns:
            ReceivedMessage if transmission succeeded (queued for delivery),
            None if packet was lost.

        Example:
            >>> emulator = ESPNOWEmulator(seed=42)
            >>> msg = V2VMessage(
            ...     vehicle_id='V002', lat=32.0, lon=34.0, speed=15.0,
            ...     heading=90.0, accel_x=0.0, accel_y=0.0, accel_z=9.81,
            ...     gyro_x=0.0, gyro_y=0.0, gyro_z=0.0, timestamp_ms=1000
            ... )
            >>> result = emulator.transmit(msg, (30, 0), (0, 0), 1000)
            >>> if result:
            ...     print(f"Latency: {result.age_ms}ms")
            ...     print(f"Arrives at: {result.received_at_ms}ms")
            ... else:
            ...     print("Packet lost")
        """
        self.current_time_ms = current_time_ms

        # Calculate distance
        distance = self._calculate_distance(sender_pos, receiver_pos)

        # Check for packet loss
        loss_prob = self._get_loss_probability(distance)

        # Burst loss check - use configured multiplier and cap
        if self._check_burst_loss(sender_msg.vehicle_id):
            burst_cfg = self.params['burst_loss']
            loss_prob = min(
                burst_cfg['max_loss_cap'],
                loss_prob * burst_cfg['loss_multiplier']
            )

        # Determine if packet is lost
        # FIXED: Use private RNG
        was_lost = self._rng.random() < loss_prob

        # FIXED: Track loss state for burst loss modeling
        self.last_loss_state[sender_msg.vehicle_id] = was_lost

        if was_lost:
            # Packet lost
            return None

        # Calculate latency
        latency_ms = self._get_latency(distance)

        # Add sensor noise (optional - can skip for minimal RL observation)
        noisy_msg = self._add_sensor_noise(sender_msg)

        # Create received message with future arrival time
        received = ReceivedMessage(
            message=noisy_msg,
            age_ms=int(latency_ms),
            received_at_ms=current_time_ms + int(latency_ms)
        )

        # FIXED: Queue for future delivery (respects causality)
        # Message will only be visible after arrival_time
        # Using msg_counter to break ties when arrival times are equal (prevents TypeError)
        arrival_time = current_time_ms + int(latency_ms)
        self.msg_counter += 1
        self.pending_messages.put((arrival_time, self.msg_counter, sender_msg.vehicle_id, received))

        return received

    def get_observation(self,
                        ego_speed: float,
                        current_time_ms: int,
                        monitored_vehicles: Optional[List[str]] = None) -> Dict:
        """
        Get current observation state for RL agent.

        Processes pending message queue first, delivering messages
        that have arrived (respects causality). This is the primary
        interface for the RL environment to retrieve vehicle states.

        Returns dictionary with latest received messages from each peer,
        including message age (staleness). Messages older than the
        staleness threshold (default 500ms) are marked as invalid.

        Args:
            ego_speed: Current ego vehicle speed in m/s
            current_time_ms: Current simulation time in milliseconds
            monitored_vehicles: Optional list of vehicle IDs to monitor.
                               If None, uses configured 'observation.monitored_vehicles'.
                               This allows dynamic topology (e.g., 6-car platoon).

        Returns:
            Observation dict matching RL environment spec:
            {
                'ego_speed': float,
                'timestamp_ms': int,
                '<vehicle_id>_lat': float,
                '<vehicle_id>_lon': float,
                '<vehicle_id>_speed': float,
                '<vehicle_id>_heading': float,
                '<vehicle_id>_accel_x': float,
                '<vehicle_id>_age_ms': int,
                '<vehicle_id>_valid': bool,
                ... (for each monitored vehicle)
            }

        Example:
            >>> emulator = ESPNOWEmulator(seed=42)
            >>> # Simulate transmission at t=1000ms
            >>> msg = V2VMessage(
            ...     vehicle_id='V002', lat=32.0, lon=34.0, speed=15.0,
            ...     heading=90.0, accel_x=-2.0, accel_y=0.0, accel_z=9.81,
            ...     gyro_x=0.0, gyro_y=0.0, gyro_z=0.0, timestamp_ms=1000
            ... )
            >>> emulator.transmit(msg, (30, 0), (0, 0), 1000)
            >>>
            >>> # At t=1050ms, message may have arrived (depends on latency)
            >>> obs = emulator.get_observation(ego_speed=12.0, current_time_ms=1050)
            >>> print(f"V002 valid: {obs['v002_valid']}")
            >>> print(f"V002 speed: {obs['v002_speed']:.1f} m/s")
            >>> print(f"Message age: {obs['v002_age_ms']}ms")
            >>>
            >>> # Custom vehicle topology (6-car platoon)
            >>> obs = emulator.get_observation(
            ...     ego_speed=12.0,
            ...     current_time_ms=1100,
            ...     monitored_vehicles=['V002', 'V003', 'V004', 'V005', 'V006']
            ... )
        """
        # FIXED: Process pending messages - deliver those that have arrived
        while not self.pending_messages.empty():
            # Peek at next message (without removing)
            # Tuple format: (arrival_time, msg_counter, vehicle_id, received_msg)
            try:
                arrival_time, _seq, vehicle_id, received_msg = self.pending_messages.queue[0]
            except IndexError:
                break

            # Check if message has arrived
            if arrival_time <= current_time_ms:
                # Deliver the message (remove from queue, add to last_received)
                self.pending_messages.get()
                self.last_received[vehicle_id] = received_msg
            else:
                # Future message - stop processing
                break

        obs = {
            'ego_speed': ego_speed,
            'timestamp_ms': current_time_ms,
        }

        # Use provided list or fall back to configured vehicles
        vehicles = monitored_vehicles or self.params['observation']['monitored_vehicles']

        for vehicle_id in vehicles:
            if vehicle_id in self.last_received:
                recv = self.last_received[vehicle_id]
                msg = recv.message

                # Calculate current age (time since received)
                age_ms = current_time_ms - recv.received_at_ms + recv.age_ms

                staleness_threshold = self.params['observation']['staleness_threshold_ms']
                obs[f'{vehicle_id.lower()}_lat'] = msg.lat
                obs[f'{vehicle_id.lower()}_lon'] = msg.lon
                obs[f'{vehicle_id.lower()}_speed'] = msg.speed
                obs[f'{vehicle_id.lower()}_heading'] = msg.heading
                obs[f'{vehicle_id.lower()}_accel_x'] = msg.accel_x
                obs[f'{vehicle_id.lower()}_age_ms'] = age_ms
                obs[f'{vehicle_id.lower()}_valid'] = age_ms < staleness_threshold
            else:
                # No message received yet
                obs[f'{vehicle_id.lower()}_lat'] = 0.0
                obs[f'{vehicle_id.lower()}_lon'] = 0.0
                obs[f'{vehicle_id.lower()}_speed'] = 0.0
                obs[f'{vehicle_id.lower()}_heading'] = 0.0
                obs[f'{vehicle_id.lower()}_accel_x'] = 0.0
                obs[f'{vehicle_id.lower()}_age_ms'] = 9999
                obs[f'{vehicle_id.lower()}_valid'] = False

        return obs


# Convenience function for testing
def test_emulator():
    """
    Test emulator with synthetic data.

    Runs 100 transmissions and reports statistics to verify basic functionality.
    """
    emulator = ESPNOWEmulator(domain_randomization=True)

    # Simulate 100 transmissions
    received_count = 0
    total_latency = 0

    for i in range(100):
        msg = V2VMessage(
            vehicle_id='V002',
            lat=32.0 + i * 0.0001,
            lon=34.0,
            speed=15.0,
            heading=90.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.8,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp_ms=i * 100,
        )

        result = emulator.transmit(
            sender_msg=msg,
            sender_pos=(i * 0.5, 0),  # V002 position
            receiver_pos=(0, 0),      # V001 position (ego)
            current_time_ms=i * 100
        )

        if result:
            received_count += 1
            total_latency += result.age_ms

    print(f"Received: {received_count}/100")
    print(f"Loss rate: {(100 - received_count)}%")
    if received_count > 0:
        print(f"Avg latency: {total_latency / received_count:.1f}ms")


if __name__ == "__main__":
    test_emulator()
