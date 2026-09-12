# Microduck arm control implementation

Date: 2026-09-12. Status: **simulation-tested control software; no motor, arm, or
physical E-stop has been connected or validated. This is not a safety certification.**

## 1. Implemented boundary

`arm_control/` is a Python standard-library control service with one coordinator
owning each configured arm transport. It does not modify or open the original
Microduck body bus.

Fixed identities:

- Original body servos remain IDs `10-14`, `20-24`, and `30-34`; IMU remains ID
  `200`.
- New left arm is IDs `40-45`; new right arm is IDs `50-55`.
- `md-arm-table-v1` is mode `single_arm`, observation/action `66/6`.
- `md-dualarm-table-v1` is mode `dual_arm`, observation/action `116/12`.
- Each arm action is five normalized joint velocities mapped to at most
  `0.4 rad/s`, then one normalized gripper-width velocity mapped to at most
  `0.01 m/s`.
- The coordinator runs at 50 Hz. It applies per-joint acceleration, position,
  finite-value, sequence, deadline, lease, device, mode, contract, and joint-map
  checks before the transport sees a target.

The six-joint order is:

```text
base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll, gripper_width
```

The RPC server is JSON-RPC 2.0, one JSON object per newline, over a Unix socket.
Unknown envelope and parameter fields are rejected. `manipulation.capabilities`
is generated from local code and validated deployment files; no RPC accepts
caller-provided physical capabilities.

## 2. Safety state and ownership behavior

States are `LOCKED`, `READY`, `EXECUTING`, `PROTECTIVE_STOP`, and `ESTOP`.

- The default daemon transport is `locked`; it cannot acquire or move an arm.
- A lease owns its listed arms exclusively and atomically. A dual-arm lease
  cannot be partially acquired.
- Commands pin `mode`, `contract_id`, dimensions, device set, session epoch,
  monotonic sequence, monotonic deadline, and each joint-map hash.
- Five control periods (100 ms) without an accepted command causes a latched
  protective stop. The lease ceiling is 500 ms.
- A late control tick, transport error, invalid measured state, lost body
  interlock, expired command, or expired lease also causes a protective stop.
- `manipulation.stop` is idempotent in effect and holds the last position; it
  does not automatically open the gripper.
- `manipulation.estop` asks every arm transport to disable torque and latches
  `ESTOP`. Reset requires the exact acknowledgement
  `I inspected the arm and cleared the cause`, a true
  `physical_estop_released` claim, and transport confirmation. The current
  serial transport cannot confirm a physical E-stop circuit, so live ESTOP
  reset remains refused.
- No stop automatically resumes. A stopped arm must be manually reset and a
  new lease acquired.

These are software containment mechanisms. They do not replace a correctly
rated hardware E-stop, power contactor, current protection, guarding, or a
supervised bench procedure.

## 3. Body movement interlock

Arm control never writes body joints. It depends on the `BodyInterlock` adapter
interface. Tests use `SimulatedBodyInterlock`, which provides exclusive
ownership and proves that acquisition/maintenance/release are enforced by the
coordinator.

`RobotdStopAdapter` sends the existing discrete `robot.stop` JSON-RPC request.
It deliberately reports `exclusive = false`: current `robotd` can stop body
velocity, but it cannot grant an exclusive movement lease, so another client
could submit a later body movement intent. Hardware arm acquisition therefore
fails closed with this adapter. A future robotd integration must add a real,
expiring body-hold lease before live arm commands can be accepted.

This is the explicit integration point; there is no direct body joint write or
second body-bus owner in this package.

## 4. Run the locked or simulated daemon

From the repository root:

```bash
# Default: serves capabilities/state but cannot acquire an arm.
python3 -m arm_control.daemon \
  --socket /tmp/microduck-arm-locked.sock

# Explicit simulation: runs both simulated arms at 50 Hz.
python3 -m arm_control.daemon \
  --transport simulation \
  --socket /tmp/microduck-arm.sock
```

In another shell, exercise a single-arm lease and command:

```bash
python3 arm_control/examples/unix_rpc_client.py /tmp/microduck-arm.sock
```

Raw capability request:

```bash
python3 - <<'PY'
import json, socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect("/tmp/microduck-arm.sock")
s.sendall(b'{"jsonrpc":"2.0","id":1,"method":"manipulation.capabilities"}\n')
print(json.loads(s.makefile("rb").readline()))
PY
```

The mutating methods are:

```text
manipulation.acquire
manipulation.command
manipulation.release
manipulation.stop
manipulation.estop
manipulation.reset
```

Read-only methods are `manipulation.capabilities` and `manipulation.state`.
The example client shows the full acquire and command envelopes, including the
server-issued session epoch and joint-map hash.

## 5. Hardware gate and Protocol 2 scope

The hardware path is not a generic `--enable` switch. Before opening a serial
path, `--transport dynamixel` requires both:

1. A root-owned, non-symlink, non-group/world-writable calibrated deployment
   manifest.
2. A separate root-owned, non-symlink, non-group/world-writable, unexpired
   bench authorization matching the deployment ID and naming its purpose.

Example schemas, with placeholder values only:

```json
{
  "schema": "microduck-arm-deployment-v1",
  "deployment_id": "bench-a",
  "calibrated": true,
  "calibration_sha256": "sha256-of-canonical-devices-object",
  "bus_paths": {
    "left-arm": "/dev/microduck-arm-left"
  },
  "devices": {
    "left-arm": {
      "servo_ids": [40, 41, 42, 43, 44, 45],
      "joint_map_hash": "value-returned-by-manipulation.capabilities",
      "calibration": [
        {"name": "base_yaw", "zero_count": 2048, "direction": 1, "counts_per_unit": 651.8986},
        {"name": "shoulder_pitch", "zero_count": 2048, "direction": 1, "counts_per_unit": 651.8986},
        {"name": "elbow_pitch", "zero_count": 2048, "direction": 1, "counts_per_unit": 651.8986},
        {"name": "wrist_pitch", "zero_count": 2048, "direction": 1, "counts_per_unit": 651.8986},
        {"name": "wrist_roll", "zero_count": 2048, "direction": 1, "counts_per_unit": 651.8986},
        {"name": "gripper_width", "zero_count": 1200, "direction": 1, "counts_per_unit": 50000}
      ]
    }
  }
}
```

```json
{
  "schema": "microduck-arm-bench-authorization-v1",
  "deployment_id": "bench-a",
  "authorized_bench": true,
  "expires_unix_s": 1789257600,
  "purpose": "supervised no-load range test"
}
```

`calibration_sha256` is SHA-256 over the compact, key-sorted JSON encoding of
the complete `devices` object. The numerical calibration above is illustrative
and must not be deployed. In particular, the gripper width mapping must come
from measured mechanism calibration.

`arm_control.dynamixel` implements Protocol 2 CRC, byte stuffing, read/write,
sync-read, sync-write, status validation, 1 Mbps POSIX serial configuration,
XL330 goal-position writes, and present current/velocity/position/voltage/
temperature decoding. It has only been packet-tested against simulated byte
data. No U2D2, XL330, UART, USB device, or motor was opened during this work.

Even with valid files, the current `RobotdStopAdapter` is non-exclusive, so the
daemon refuses hardware startup before opening a serial path. This is
intentional until the body authority exposes a real movement-hold lease and a
physical E-stop feedback adapter exists.

## 6. Proposed systemd deployment

- `arm_control/systemd/microduck-arm-control.service` is the installable default:
  dedicated user/group, `0660` socket under a private runtime directory,
  `PrivateDevices=yes`, and locked transport.
- `arm_control/systemd/microduck-arm-control-bench.service` is an intentionally
  non-enabled bench proposal. It grants only named stable arm device paths and
  reads the two authorization files. The sample `DeviceAllow` paths must be
  replaced by udev-backed identities from the actual deployment.

The bench unit should be started only by the supervised bench procedure after
the physical power, E-stop, device identity, and calibration evidence exists.
It is not enabled by `WantedBy`.

## 7. Verification and remaining hardware work

Run:

```bash
python3 -m unittest discover -s arm_control/tests -v
python3 -m compileall -q arm_control
```

Covered in software tests:

- fixed body/arm ID separation and stable joint-map hashes;
- secure deployment/authorization file checks;
- per-arm exclusive leases and dual/single contract shape;
- mode, device, epoch, sequence, deadline, joint-map, finite, and normalized
  action rejection;
- 50 Hz velocity/acceleration integration and position limiting;
- command watchdog, lease stop, protective stop, ESTOP, and manual reset;
- body-interlock acquisition and refusal of the non-exclusive robotd adapter;
- real Unix socket framing/permissions and strict JSON-RPC fields;
- Protocol 2 ping CRC, read/write encoding, sync-write encoding, and status CRC.

Not tested and not claimed:

- serial timing, half-duplex direction behavior, U2D2 compatibility, device
  return ordering, XL330 register/model identity, or EEPROM settings;
- actual joint directions, zero counts, gripper geometry, current units, safe
  torque/gain, thermal/voltage thresholds, collision bounds, or payload;
- physical E-stop feedback, power-cut behavior, dropped-load behavior, or body
  movement exclusion on a real Microduck;
- real-time scheduling guarantees on the Microduck host.

Those items remain G7 supervised bench work. Passing these unit tests does not
authorize hardware motion.
