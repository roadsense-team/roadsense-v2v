# Exercise 4: Parameter Variation (Manual Augmentation)

**Date:** December 20, 2025
**SUMO Version:** 1.25.0+ (v1_25_0+0604)
**Status:** ✅ Complete - All 5 scenarios generated and executed

---

## 🎯 Exercise Goal

Learn how to create data augmentation by systematically varying simulation parameters. This exercise demonstrates how to take **1 base scenario** and create **multiple realistic variations** for ML training data diversity.

---

## 📊 Scenarios Created

### BASE (Reference)
- **Speed:** 15 m/s
- **Spacing:** 30m gaps (V003 @60m, V002 @30m, V001 @0m)
- **Deceleration:** 4.5 m/s²
- **Following (tau):** 1.0 seconds
- **Driver imperfection (sigma):** 0.5

### Variation 1: var_speed_fast
- **Change:** Speed +20% (15 → 18 m/s)
- **Effect:** Vehicles travel faster, dynamics evolve quicker
- **Use case:** Simulates highway driving or aggressive drivers

### Variation 2: var_tight_convoy
- **Change:** Tighter spacing (30m → 20m gaps)
- **Effect:** Less safety margin, more stress on car-following
- **Use case:** Simulates congested traffic or tight platooning

### Variation 3: var_weak_brakes
- **Change:** Reduced deceleration (4.5 → 3.5 m/s²)
- **Effect:** Slower braking response, increased stopping distance
- **Use case:** Simulates wet roads, worn brakes, heavy vehicles

### Variation 4: var_aggressive_follower
- **Change:** Aggressive following (tau 1.0 → 0.5 seconds)
- **Effect:** Vehicles maintain closer following distance
- **Use case:** Simulates tailgating, impatient drivers

---

## 📁 Directory Structure

```
exercise4_variations/
├── base/
│   ├── network.net.xml        (OSM road network)
│   ├── vehicles.rou.xml       (SSM enabled, base parameters)
│   ├── scenario.sumocfg       (10Hz sampling, CSV output)
│   ├── fcd_output.csv         (206KB - vehicle trajectories)
│   └── ssm_output.xml         (Safety conflict data)
├── var_speed_fast/
│   ├── vehicles.rou.xml       (departSpeed="18")
│   ├── fcd_output.csv         (206KB)
│   └── ssm_output.xml
├── var_tight_convoy/
│   ├── vehicles.rou.xml       (departPos: 40, 20, 0)
│   ├── fcd_output.csv         (213KB)
│   └── ssm_output.xml
├── var_weak_brakes/
│   ├── vehicles.rou.xml       (decel="3.5")
│   ├── fcd_output.csv         (206KB)
│   └── ssm_output.xml
└── var_aggressive_follower/
    ├── vehicles.rou.xml       (tau="0.5")
    ├── fcd_output.csv         (207KB)
    └── ssm_output.xml
```

---

## ✅ Key Learnings

### 1. **SSM Device Configuration**
All scenarios now include SSM (Surrogate Safety Measures) device:
```xml
<vType id="car" ...>
    <param key="has.ssm.device" value="true"/>
    <param key="device.ssm.measures" value="TTC DRAC PET"/>
    <param key="device.ssm.thresholds" value="4.0 3.4 2.0"/>
    <param key="device.ssm.range" value="100.0"/>
    <param key="device.ssm.file" value="ssm_output.xml"/>
</vType>
```

### 2. **No Conflicts Detected (Expected)**
- All SSM logs are empty (`<SSMLog></SSMLog>`)
- **Why?** Normal convoy driving without emergency events is safe
- **Next step:** Exercise 5 will add TraCI emergency braking to trigger conflicts

### 3. **CSV Outputs Show Different Dynamics**
- All variations produced valid CSV trajectories
- Tight convoy: Slightly larger file (213KB vs 206KB) - more data due to vehicle interactions
- Different speeds/behaviors visible in trajectory data

### 4. **Parameter Variation is Systematic**
Each variation changes **exactly one parameter** while keeping others constant. This allows us to understand:
- **Which parameters matter most** for safety
- **How to create realistic diversity** in training data
- **Foundation for automation** (Python script can generate 100+ variations)

---

## 🚀 Running the Scenarios

### Run All Variations
```bash
cd ~/projects/RoadSense2/sumo_learning/exercise4_variations

for dir in base var_*; do
    echo "Running $dir..."
    cd $dir
    docker run --rm \
      -v $(pwd):/data:Z \
      -w /data \
      ghcr.io/eclipse-sumo/sumo:main \
      sumo -c scenario.sumocfg --no-warnings
    cd ..
done
```

### Run Single Variation with GUI
```bash
cd ~/projects/RoadSense2/sumo_learning/exercise4_variations/var_speed_fast

docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/data:Z \
  -w /data \
  ghcr.io/eclipse-sumo/sumo:main \
  sumo-gui -c scenario.sumocfg
```

---

## 📈 Next Steps (Exercise 5)

To create **actual hazard scenarios** with SSM conflicts, we need:

1. **Add TraCI Python script** to inject emergency events
2. **Program lead vehicle (V003) to brake suddenly** at specific time
3. **Observe TTC/DRAC values** drop below thresholds
4. **Generate hazard labels** automatically from SSM

Example TraCI scenario:
```python
# At t=30s, force V003 to emergency brake
traci.vehicle.setSpeed("V003", 0)
# SSM will detect: TTC < 1.5s, DRAC > 5.0 m/s²
# Label: CRITICAL hazard
```

---

## 🎓 Exercise 4 Success Criteria

✅ Created 5 scenarios (1 base + 4 variations)
✅ Each variation changes exactly 1 parameter
✅ All scenarios run successfully with SUMO 1.25.0+
✅ CSV outputs generated (10Hz, acceleration data)
✅ SSM device enabled for future conflict detection
✅ Learned systematic parameter variation approach

**Exercise 4 Complete! Ready for Exercise 5 (TraCI Emergency Events)** 🎉

---

## 💡 Automation Ideas (Future)

Once Exercise 5 is complete, create Python script to generate variations:

```python
# Auto-generate 20 variations per base scenario
variations = {
    'speed': [0.8, 0.9, 1.0, 1.1, 1.2],
    'spacing': [-10, -5, 0, +5, +10],
    'decel': [3.0, 3.5, 4.0, 4.5, 5.0],
    'tau': [0.5, 0.75, 1.0, 1.25, 1.5]
}

for combo in itertools.product(variations.values()):
    create_scenario(base, combo)
    run_sumo()
    extract_ssm_labels()
```

Result: **1 real scenario → 100+ augmented training scenarios** 🚀
