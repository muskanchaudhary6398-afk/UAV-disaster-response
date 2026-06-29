"""
sitl/mavlink_controller.py

Wraps pymavlink so the rest of the pipeline can issue simple high-level
commands (arm_and_takeoff, hover, send_velocity, goto, land) against an
ArduPilot SITL instance, without sprinkling raw MAVLink message construction
across the perception pipeline.

Tested against:
  - ArduCopter SITL (`sim_vehicle.py -v ArduCopter --console --map`)
  - Connection string typically: 'udp:127.0.0.1:14550' (mavproxy forwarding)
    or 'tcp:127.0.0.1:5760' (direct SITL connection)
"""

import time
from dataclasses import dataclass


@dataclass
class Position:
    lat: float
    lon: float
    alt: float  # relative altitude, meters


class ArduPilotController:
    def __init__(self, connection_string: str = "udp:127.0.0.1:14550", baud: int = 115200):
        from pymavlink import mavutil  # local import: optional dependency for non-SITL usage

        self._mavutil = mavutil
        self.connection_string = connection_string
        print(f"[mavlink] connecting to {connection_string} ...")
        self.master = mavutil.mavlink_connection(connection_string, baud=baud)
        self.master.wait_heartbeat()
        print(f"[mavlink] heartbeat received (sysid={self.master.target_system}, "
              f"compid={self.master.target_component})")

    # ---------------------------------------------------------------- arming

    def set_mode(self, mode: str):
        mode_id = self.master.mode_mapping()[mode]
        self.master.mav.set_mode_send(
            self.master.target_system,
            self._mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )

    def arm(self, timeout: float = 10.0):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            self._mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        self.master.motors_armed_wait()
        print("[mavlink] armed")

    def arm_and_takeoff(self, target_altitude: float):
        self.set_mode("GUIDED")
        self.arm()
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            self._mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, target_altitude,
        )
        print(f"[mavlink] takeoff to {target_altitude} m requested")
        self._wait_for_altitude(target_altitude)

    def _wait_for_altitude(self, target_altitude: float, tolerance: float = 0.5, timeout: float = 30.0):
        start = time.time()
        while time.time() - start < timeout:
            msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
            if msg is None:
                continue
            current_alt = msg.relative_alt / 1000.0
            if current_alt >= target_altitude - tolerance:
                print(f"[mavlink] reached target altitude ({current_alt:.1f} m)")
                return
        print("[mavlink] WARNING: takeoff altitude wait timed out")

    # ------------------------------------------------------------- movement

    def send_ned_velocity(self, vx: float, vy: float, vz: float, duration: float = 1.0):
        """Send a body/NED-frame velocity command (m/s) for `duration` seconds.
        Used for the perception-driven reactive maneuvers, e.g. slow down /
        hold position when a human is freshly detected.
        """
        msg = self.master.mav.set_position_target_local_ned_encode(
            0,
            self.master.target_system, self.master.target_component,
            self._mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,  # only velocity components enabled
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0,
        )
        end_time = time.time() + duration
        while time.time() < end_time:
            self.master.mav.send(msg)
            time.sleep(0.1)

    def hover(self, duration: float = 3.0):
        print(f"[mavlink] holding position for {duration:.1f}s (perception trigger)")
        self.send_ned_velocity(0, 0, 0, duration)

    def goto_relative(self, dx: float, dy: float, dz: float, speed: float = 2.0):
        """Move dx (north) / dy (east) / dz (down) meters at constant speed —
        used for simple obstacle-avoidance reroutes when debris blocks the path."""
        distance = max(abs(dx), abs(dy), abs(dz), 1e-6)
        duration = distance / speed
        vx, vy, vz = (dx / duration, dy / duration, dz / duration)
        self.send_ned_velocity(vx, vy, vz, duration)

    def land(self):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            self._mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        print("[mavlink] land command sent")

    def close(self):
        self.master.close()


class MockController:
    """No-MAVLink-connection stand-in for unit testing decision logic."""

    def __init__(self):
        self.log = []

    def arm_and_takeoff(self, target_altitude: float):
        self.log.append(("arm_and_takeoff", target_altitude))

    def hover(self, duration: float = 3.0):
        self.log.append(("hover", duration))

    def goto_relative(self, dx: float, dy: float, dz: float, speed: float = 2.0):
        self.log.append(("goto_relative", dx, dy, dz, speed))

    def send_ned_velocity(self, vx: float, vy: float, vz: float, duration: float = 1.0):
        self.log.append(("send_ned_velocity", vx, vy, vz, duration))

    def land(self):
        self.log.append(("land",))

    def close(self):
        self.log.append(("close",))
