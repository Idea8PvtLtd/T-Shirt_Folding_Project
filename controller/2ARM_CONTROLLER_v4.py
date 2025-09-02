from pymycobot import ElephantRobot
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import socket
import json
from datetime import datetime

# Robot network and UDP ports
ROBOT1_IP = "192.168.1.159"
ROBOT1_PORT = 5001
ROBOT2_IP = "192.168.1.160"
ROBOT2_PORT = 5002
UDP_PORT_ROBOT1 = 5005
UDP_PORT_ROBOT2 = 5006

# Defaults
DEFAULT_SPEED = 100
GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1

# IO Pins
NUM_DIGITAL_INPUTS = 16
NUM_DIGITAL_OUTPUTS = 16
NUM_TOOL_OUTPUTS = 2

# Plane data class
class Plane:
    def __init__(self, name, limits, initial_coords, rotation_values=None):
        self.name = name
        self.limits = limits  # {'x': (min,max), 'y': (min,max), 'z': (min,max)}
        self.coords = initial_coords  # 6 DOF coords list
        self.rotation_values = rotation_values or [0, 0, 0]

# Enhanced Robot wrapper with connection recovery
class LoggingElephantRobot:
    def __init__(self, robot, robot_name, robot_ip, robot_port):
        self.robot = robot
        self.robot_name = robot_name
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.connected = True
        self.last_successful_command = time.time()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.reconnect_delay = 5  # seconds
    
    def is_connected(self):
        """Check if robot is still connected"""
        try:
            # Quick test command to verify connection
            self.robot.get_angles()
            self.last_successful_command = time.time()
            self.connected = True
            self.reconnect_attempts = 0
            return True
        except Exception:
            self.connected = False
            return False
    
    def reconnect(self):
        """Attempt to reconnect to the robot"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"{self.robot_name}: Max reconnection attempts reached")
            return False
        
        try:
            print(f"{self.robot_name}: Attempting reconnection ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
            
            # Close existing connection
            try:
                self.robot.stop_client()
            except:
                pass
            
            time.sleep(self.reconnect_delay)
            
            # Create new connection
            self.robot = ElephantRobot(self.robot_ip, self.robot_port)
            self.robot.start_client()
            time.sleep(3)
            
            # Test the connection
            if self.is_connected():
                print(f"{self.robot_name}: Reconnection successful")
                return True
            else:
                self.reconnect_attempts += 1
                return False
                
        except Exception as e:
            self.reconnect_attempts += 1
            print(f"{self.robot_name}: Reconnection failed: {e}")
            return False
    
    def safe_execute(self, method_name, *args, **kwargs):
        """Safely execute robot commands with error handling and reconnection"""
        if not self.connected and not self.is_connected():
            if not self.reconnect():
                raise Exception(f"{self.robot_name} is disconnected and reconnection failed")
        
        try:
            method = getattr(self.robot, method_name)
            result = method(*args, **kwargs)
            self.last_successful_command = time.time()
            return result
        except Exception as e:
            self.connected = False
            error_msg = str(e)
            
            # Check if it's a connection error
            if any(err in error_msg for err in ['10054', '10060', 'Connection', 'connection']):
                print(f"{self.robot_name}: Connection lost during {method_name}")
                # Try to reconnect once
                if self.reconnect():
                    try:
                        method = getattr(self.robot, method_name)
                        result = method(*args, **kwargs)
                        return result
                    except Exception as retry_e:
                        raise Exception(f"Command failed after reconnection: {retry_e}")
                else:
                    raise Exception(f"Connection lost and reconnection failed")
            else:
                # Non-connection error, re-raise
                raise e
    
    def __getattr__(self, name):
        attr = getattr(self.robot, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                try:
                    result = self.safe_execute(name, *args, **kwargs)
                    return result
                except Exception as e:
                    print(f"Error executing {name} on {self.robot_name}: {e}")
                    raise
            return wrapper
        return attr

class RobotController:
    def __init__(self, parent_frame, robot, robot_name, robot_id, planes_config, robot_ip, robot_port):
        self.parent_frame = parent_frame
        self.robot = LoggingElephantRobot(robot, robot_name, robot_ip, robot_port)
        self.robot_name = robot_name
        self.robot_id = robot_id
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.planes = planes_config
        
        self.current_plane_name = list(self.planes.keys())[0]
        self.current_plane = self.planes[self.current_plane_name]
        self.current_coords = self.current_plane.coords.copy()

        self.digital_input_states = [False] * NUM_DIGITAL_INPUTS
        self.digital_output_states = [False] * NUM_DIGITAL_OUTPUTS
        self.tool_output_states = [False] * NUM_TOOL_OUTPUTS
        
        self.digital_input_indicators = []
        self.digital_output_buttons = []
        self.tool_output_buttons = []
        
        self.notebook = ttk.Notebook(parent_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.movement_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.movement_tab, text="Movement Control")
        
        self.io_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.io_tab, text="IO Control")
        
        self.create_movement_widgets()
        self.create_io_widgets()
        
        self.io_monitoring = True
        self.io_thread = threading.Thread(target=self.monitor_io_pins, daemon=True)
        self.io_thread.start()

    def create_movement_widgets(self):
        plane_frame = ttk.LabelFrame(self.movement_tab, text="Planes")
        plane_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.plane_label = ttk.Label(plane_frame, text=f"Active Plane: {self.current_plane_name}")
        self.plane_label.grid(row=0, column=0, padx=5, pady=5)

        row_idx = 1
        for plane_name, plane in self.planes.items():
            limits_text = (f"{plane_name} Limits: "
                           f"X{plane.limits['x']} Y{plane.limits['y']} Z{plane.limits['z']}")
            ttk.Label(plane_frame, text=limits_text).grid(row=row_idx, column=0, padx=5, pady=2, sticky="w")
            row_idx += 1
        
        for i, plane_name in enumerate(self.planes.keys()):
            ttk.Button(plane_frame, text=f"Switch to {plane_name}",
                       command=lambda pn=plane_name: self.switch_plane(pn)).grid(row=row_idx + i, column=0, padx=5, pady=2, sticky="w")

        coord_frame = ttk.LabelFrame(self.movement_tab, text="Cartesian Coordinates")
        coord_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(coord_frame, text="X:").grid(row=0, column=0, padx=5, pady=5)
        self.x_entry = ttk.Entry(coord_frame)
        self.x_entry.insert(0, str(self.current_coords[0]))
        self.x_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(coord_frame, text="Y:").grid(row=1, column=0, padx=5, pady=5)
        self.y_entry = ttk.Entry(coord_frame)
        self.y_entry.insert(0, str(self.current_coords[1]))
        self.y_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(coord_frame, text="Z:").grid(row=2, column=0, padx=5, pady=5)
        self.z_entry = ttk.Entry(coord_frame)
        self.z_entry.insert(0, str(self.current_coords[2]))
        self.z_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(coord_frame, text="Speed:").grid(row=3, column=0, padx=5, pady=5)
        self.speed_entry = ttk.Entry(coord_frame)
        self.speed_entry.insert(0, "90000")
        self.speed_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Button(coord_frame, text="Move", command=self.move_robot).grid(row=4, column=0, columnspan=2, pady=5)

        gripper_frame = ttk.LabelFrame(self.movement_tab, text="Gripper Control")
        gripper_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Button(gripper_frame, text="Open Gripper",
                   command=lambda: self.control_gripper(GRIPPER_OPEN)).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(gripper_frame, text="Close Gripper",
                   command=lambda: self.control_gripper(GRIPPER_CLOSE)).grid(row=0, column=1, padx=5, pady=5)

    def create_io_widgets(self):
        io_container = ttk.Frame(self.io_tab)
        io_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        input_frame = ttk.LabelFrame(io_container, text="Digital Inputs")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        output_frame = ttk.LabelFrame(io_container, text="Digital Outputs")
        output_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        tool_frame = ttk.LabelFrame(io_container, text="Tool Outputs")
        tool_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        ttk.Button(input_frame, text="Read All Inputs", command=self.read_all_digital_inputs).grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        for i in range(NUM_DIGITAL_INPUTS):
            row = (i % 8) + 1
            col = i // 8 * 2
            ttk.Label(input_frame, text=f"DI {i}:").grid(row=row, column=col, padx=5, pady=5, sticky="e")
            indicator = ttk.Label(input_frame, text="   ", background="gray")
            indicator.grid(row=row, column=col+1, padx=5, pady=5, sticky="w")
            self.digital_input_indicators.append(indicator)
        
        for i in range(NUM_DIGITAL_OUTPUTS):
            row = (i % 8) + 1
            col = i // 8 * 2
            ttk.Label(output_frame, text=f"DO {i}:").grid(row=row, column=col, padx=5, pady=5, sticky="e")
            button = ttk.Button(output_frame, text="OFF", width=5,
                              command=lambda idx=i: self.toggle_digital_output(idx))
            button.grid(row=row, column=col+1, padx=5, pady=5, sticky="w")
            self.digital_output_buttons.append(button)
        
        for i in range(NUM_TOOL_OUTPUTS):
            ttk.Label(tool_frame, text=f"Tool Out {i}:").grid(row=0, column=i*2, padx=5, pady=5, sticky="e")
            button = ttk.Button(tool_frame, text="OFF", width=5,
                              command=lambda idx=i: self.toggle_tool_output(idx))
            button.grid(row=0, column=i*2+1, padx=5, pady=5, sticky="w")
            self.tool_output_buttons.append(button)
        
        manual_frame = ttk.LabelFrame(io_container, text="Common Operations")
        manual_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        ttk.Button(manual_frame, text="Set All Outputs LOW", command=self.set_all_outputs_low).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(manual_frame, text="Set All Outputs HIGH", command=self.set_all_outputs_high).grid(row=0, column=1, padx=5, pady=5)

    def monitor_io_pins(self):
        """Enhanced IO monitoring that handles connection failures gracefully"""
        while self.io_monitoring:
            try:
                # Only monitor IO if we're on the IO tab and robot is connected
                if (self.notebook.index(self.notebook.select()) == 1 and 
                    hasattr(self.robot, 'is_connected') and 
                    self.robot.is_connected()):
                    self.read_all_digital_inputs()
                    time.sleep(1)
                else:
                    # If not connected or not on IO tab, sleep longer
                    time.sleep(5)
            except Exception as e:
                print(f"IO monitoring error for {self.robot_name}: {str(e)}")
                # Wait longer before retrying if there's an error
                time.sleep(10)

    def read_all_digital_inputs(self):
        """Enhanced digital input reading with connection checking"""
        try:
            # Check if robot is connected before attempting to read
            if hasattr(self.robot, 'is_connected') and not self.robot.is_connected():
                # Set all indicators to gray if disconnected
                for indicator in self.digital_input_indicators:
                    indicator.config(background="gray")
                return
            
            for i in range(NUM_DIGITAL_INPUTS):
                try:
                    pin_value = self.robot.get_digital_in(i)
                    self.digital_input_states[i] = bool(pin_value)
                    color = "green" if self.digital_input_states[i] else "red"
                    self.digital_input_indicators[i].config(background=color)
                except Exception as pin_error:
                    # If individual pin read fails, mark it as unknown
                    self.digital_input_indicators[i].config(background="orange")
                    
        except Exception as e:
            print(f"Error reading digital inputs for {self.robot_name}: {str(e)}")
            # Set all indicators to gray on error
            for indicator in self.digital_input_indicators:
                indicator.config(background="gray")

    def toggle_digital_output(self, pin_index):
        threading.Thread(target=self._toggle_output_thread, args=(pin_index,), daemon=True).start()

    def _toggle_output_thread(self, pin_index):
        try:
            new_state = not self.digital_output_states[pin_index]
            self.robot.set_digital_out(pin_index, int(new_state))
            self.digital_output_states[pin_index] = new_state
            button_text = "ON" if new_state else "OFF"
            self.digital_output_buttons[pin_index].config(text=button_text)
        except Exception as e:
            print(f"Error toggling digital output {pin_index} for {self.robot_name}: {str(e)}")

    def toggle_tool_output(self, pin_index):
        threading.Thread(target=self._toggle_tool_output_thread, args=(pin_index,), daemon=True).start()

    def _toggle_tool_output_thread(self, pin_index):
        try:
            new_state = not self.tool_output_states[pin_index]
            self.robot.set_tool_reference(pin_index, int(new_state))
            self.tool_output_states[pin_index] = new_state
            button_text = "ON" if new_state else "OFF"
            self.tool_output_buttons[pin_index].config(text=button_text)
        except Exception as e:
            print(f"Error toggling tool output {pin_index} for {self.robot_name}: {str(e)}")

    def set_all_outputs_low(self):
        threading.Thread(target=self._set_all_outputs, args=(False,), daemon=True).start()

    def set_all_outputs_high(self):
        threading.Thread(target=self._set_all_outputs, args=(True,), daemon=True).start()

    def _set_all_outputs(self, state):
        try:
            for i in range(NUM_DIGITAL_OUTPUTS):
                self.robot.set_digital_out(i, int(state))
                self.digital_output_states[i] = state
                button_text = "ON" if state else "OFF"
                self.digital_output_buttons[i].config(text=button_text)
                time.sleep(0.1)
        except Exception as e:
            print(f"Error setting all outputs for {self.robot_name}: {str(e)}")

    def process_udp_command(self, message, source_addr):
        try:
            coords = json.loads(message)
            
            has_coordinates = any(key in coords for key in ['x', 'y', 'z'])
            has_io_commands = 'io' in coords
            
            if has_coordinates:
                plane_indicator = coords.get('plane', self.current_plane_name)
                if plane_indicator not in self.planes:
                    print(f"Unknown plane indicator: {plane_indicator} for {self.robot_name}")
                    return
                
                self.switch_plane(plane_indicator)
                
                camera_x = float(coords.get('x', self.current_coords[0]))
                camera_y = float(coords.get('y', self.current_coords[1]))
                camera_z = float(coords.get('z', self.current_coords[2]))
                
                x = self.clamp(camera_x, self.current_plane.limits['x'])
                y = self.clamp(camera_y, self.current_plane.limits['y'])
                z = self.clamp(camera_z, self.current_plane.limits['z'])
                
                self.current_coords[0] = x
                self.current_coords[1] = y
                self.current_coords[2] = z
                
                self.update_entries(x, y, z)
                self.move_robot()
            
            if has_io_commands:
                io_commands = coords['io']
                if 'output' in io_commands:
                    for pin, state in io_commands['output'].items():
                        try:
                            pin_idx = int(pin)
                            if 0 <= pin_idx < NUM_DIGITAL_OUTPUTS:
                                if isinstance(state, str):
                                    state_bool = state.lower() == 'true'
                                else:
                                    state_bool = bool(state)
                                threading.Thread(target=self._set_output_thread, args=(pin_idx, state_bool), daemon=True).start()
                        except ValueError:
                            print(f"Invalid IO pin number: {pin}")
                if 'tool_output' in io_commands:
                    for pin, state in io_commands['tool_output'].items():
                        try:
                            pin_idx = int(pin)
                            if 0 <= pin_idx < NUM_TOOL_OUTPUTS:
                                if isinstance(state, str):
                                    state_bool = state.lower() == 'true'
                                else:
                                    state_bool = bool(state)
                                threading.Thread(target=self._set_tool_output_thread, args=(pin_idx, state_bool), daemon=True).start()
                        except ValueError:
                            print(f"Invalid tool output pin number: {pin}")
                            
            if not has_coordinates and not has_io_commands:
                print(f"Invalid UDP command for {self.robot_name}: No coordinates or IO commands found")
                
        except Exception as e:
            print(f"Invalid UDP data for {self.robot_name}: {str(e)}")

    def _set_output_thread(self, pin_index, state):
        try:
            self.robot.set_digital_out(pin_index, int(state))
            self.digital_output_states[pin_index] = state
            button_text = "ON" if state else "OFF"
            self.digital_output_buttons[pin_index].config(text=button_text)
        except Exception as e:
            print(f"Error setting digital output {pin_index} for {self.robot_name}: {str(e)}")

    def _set_tool_output_thread(self, pin_index, state):
        try:
            self.robot.set_tool_reference(pin_index, int(state))
            self.tool_output_states[pin_index] = state
            button_text = "ON" if state else "OFF"
            self.tool_output_buttons[pin_index].config(text=button_text)
        except Exception as e:
            print(f"Error setting tool output {pin_index} for {self.robot_name}: {str(e)}")

    def clamp(self, val, limits):
        return max(min(val, limits[1]), limits[0])

    def update_entries(self, x, y, z):
        self.x_entry.delete(0, tk.END)
        self.x_entry.insert(0, str(round(x, 3)))
        self.y_entry.delete(0, tk.END)
        self.y_entry.insert(0, str(round(y, 3)))
        self.z_entry.delete(0, tk.END)
        self.z_entry.insert(0, str(round(z, 3)))
        self.current_coords[0] = x
        self.current_coords[1] = y
        self.current_coords[2] = z

    def move_robot(self):
        try:
            x_limits = self.current_plane.limits['x']
            y_limits = self.current_plane.limits['y']
            z_limits = self.current_plane.limits['z']

            x = self.clamp(float(self.x_entry.get()), x_limits)
            y = self.clamp(float(self.y_entry.get()), y_limits)
            z = self.clamp(float(self.z_entry.get()), z_limits)
            speed = int(self.speed_entry.get())

            self.update_entries(x, y, z)
            self.current_coords[0] = x
            self.current_coords[1] = y
            self.current_coords[2] = z

            threading.Thread(target=self._move_thread, args=(self.current_coords, speed), daemon=True).start()
        except ValueError:
            print(f"Invalid input values for {self.robot_name}")

    def _move_thread(self, coords, speed):
        try:
            self.robot.write_coords(coords, speed)
            self.robot.command_wait_done()
            self.current_plane.coords = coords.copy()
        except Exception as e:
            print(f"Movement error for {self.robot_name}: {str(e)}")

    def control_gripper(self, state):
        threading.Thread(target=self._gripper_thread, args=(state,), daemon=True).start()

    def _gripper_thread(self, state):
        try:
            self.robot.set_gripper_state(state, DEFAULT_SPEED)
            time.sleep(2)
        except Exception as e:
            print(f"Gripper error for {self.robot_name}: {str(e)}")

    def switch_plane(self, plane_name):
        if plane_name not in self.planes:
            print(f"Plane {plane_name} not configured for {self.robot_name}")
            return
        self.current_plane.coords = self.current_coords.copy()
        self.current_plane_name = plane_name
        self.current_plane = self.planes[plane_name]
        self.current_coords = self.current_plane.coords.copy()
        self.plane_label.config(text=f"Active Plane: {self.current_plane_name}")
        self.update_entries(self.current_coords[0], self.current_coords[1], self.current_coords[2])

    def stop_monitoring(self):
        self.io_monitoring = False

class DualRobotControlUI(tk.Tk):
    def __init__(self, robot1, robot2, robot1_planes, robot2_planes):
        super().__init__()
        self.robot1 = robot1
        self.robot2 = robot2
        self.title("Dual Robot Arm Control System")
        self.geometry("1400x900")
        
        self.main_notebook = ttk.Notebook(self)
        self.main_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.robot1_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.robot1_frame, text="Robot 1 (159:5001)")
        
        self.robot2_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.robot2_frame, text="Robot 2 (160:5002)")
        
        self.sync_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.sync_frame, text="Synchronized Control")

        self.udp_socket1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket1.bind(("0.0.0.0", UDP_PORT_ROBOT1))
        
        self.udp_socket2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket2.bind(("0.0.0.0", UDP_PORT_ROBOT2))
        
        self.udp_running = True

        self.udp_thread1 = threading.Thread(target=self.udp_listener_robot1, daemon=True)
        self.udp_thread1.start()
        
        self.udp_thread2 = threading.Thread(target=self.udp_listener_robot2, daemon=True)
        self.udp_thread2.start()

        self.robot1_controller = RobotController(self.robot1_frame, robot1, "Robot 1", 1, robot1_planes, ROBOT1_IP, ROBOT1_PORT)
        self.robot2_controller = RobotController(self.robot2_frame, robot2, "Robot 2", 2, robot2_planes, ROBOT2_IP, ROBOT2_PORT)
        
        self.create_sync_controls()
        
        self.status_bar = ttk.Label(self, text=f"Dual Robot System Ready - Robot 1 UDP: {UDP_PORT_ROBOT1}, Robot 2 UDP: {UDP_PORT_ROBOT2}", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_sync_controls(self):
        udp_info_frame = ttk.LabelFrame(self.sync_frame, text="UDP Port Information")
        udp_info_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(udp_info_frame, text=f"Robot 1 UDP Port: {UDP_PORT_ROBOT1}").pack(pady=5)
        ttk.Label(udp_info_frame, text=f"Robot 2 UDP Port: {UDP_PORT_ROBOT2}").pack(pady=5)
        
        emergency_frame = ttk.LabelFrame(self.sync_frame, text="Emergency Controls")
        emergency_frame.pack(fill="x", padx=10, pady=10)
        
        emergency_button = ttk.Button(emergency_frame, text="STOP ALL ROBOTS", command=self.emergency_stop)
        emergency_button.pack(pady=10)
        
        sync_move_frame = ttk.LabelFrame(self.sync_frame, text="Synchronized Movement")
        sync_move_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(sync_move_frame, text="Move Both to Home Position", command=self.move_both_home).pack(pady=5)
        ttk.Button(sync_move_frame, text="Sync Gripper Open", command=lambda: self.sync_gripper(GRIPPER_OPEN)).pack(pady=5)
        ttk.Button(sync_move_frame, text="Sync Gripper Close", command=lambda: self.sync_gripper(GRIPPER_CLOSE)).pack(pady=5)
        
        status_frame = ttk.LabelFrame(self.sync_frame, text="System Status")
        status_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.robot1_status = ttk.Label(status_frame, text="Robot 1: Ready")
        self.robot1_status.pack(pady=5)
        
        self.robot2_status = ttk.Label(status_frame, text="Robot 2: Ready")
        self.robot2_status.pack(pady=5)
        
        # Connection status indicators
        connection_frame = ttk.LabelFrame(status_frame, text="Connection Status")
        connection_frame.pack(fill="x", pady=10)
        
        self.robot1_connection_status = ttk.Label(connection_frame, text="Robot 1: Connected", foreground="green")
        self.robot1_connection_status.pack(pady=2)
        
        self.robot2_connection_status = ttk.Label(connection_frame, text="Robot 2: Connected", foreground="green")
        self.robot2_connection_status.pack(pady=2)
        
        # Start connection monitoring
        self.start_connection_monitoring()

    def start_connection_monitoring(self):
        """Start monitoring robot connections"""
        def monitor_connections():
            while True:
                try:
                    # Check Robot 1
                    if hasattr(self.robot1_controller.robot, 'is_connected'):
                        if self.robot1_controller.robot.is_connected():
                            self.robot1_connection_status.config(text="Robot 1: Connected", foreground="green")
                        else:
                            self.robot1_connection_status.config(text="Robot 1: Disconnected", foreground="red")
                    
                    # Check Robot 2
                    if hasattr(self.robot2_controller.robot, 'is_connected'):
                        if self.robot2_controller.robot.is_connected():
                            self.robot2_connection_status.config(text="Robot 2: Connected", foreground="green")
                        else:
                            self.robot2_connection_status.config(text="Robot 2: Disconnected", foreground="red")
                    
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    print(f"Connection monitoring error: {e}")
                    time.sleep(10)
        
        connection_thread = threading.Thread(target=monitor_connections, daemon=True)
        connection_thread.start()

    def udp_listener_robot1(self):
        while self.udp_running:
            try:
                data, addr = self.udp_socket1.recvfrom(1024)
                message = data.decode()
                self.robot1_controller.process_udp_command(message, addr)
                self.robot1_status.config(text=f"Robot 1: Command received from {addr[0]}")
            except OSError:
                break
            except Exception as e:
                self.robot1_status.config(text=f"Robot 1 UDP error: {str(e)}")

    def udp_listener_robot2(self):
        while self.udp_running:
            try:
                data, addr = self.udp_socket2.recvfrom(1024)
                message = data.decode()
                self.robot2_controller.process_udp_command(message, addr)
                self.robot2_status.config(text=f"Robot 2: Command received from {addr[0]}")
            except OSError:
                break
            except Exception as e:
                self.robot2_status.config(text=f"Robot 2 UDP error: {str(e)}")

    def emergency_stop(self):
        try:
            self.robot1.stop()
            self.robot2.stop()
            self.robot1_status.config(text="Robot 1: EMERGENCY STOPPED")
            self.robot2_status.config(text="Robot 2: EMERGENCY STOPPED")
            self.status_bar.config(text="EMERGENCY STOP ACTIVATED - Both robots stopped")
        except Exception as e:
            self.status_bar.config(text=f"Emergency stop error: {str(e)}")

    def move_both_home(self):
        def move_home_thread():
            try:
                home_coords_1 = self.robot1_controller.current_plane.coords
                home_coords_2 = self.robot2_controller.current_plane.coords
                
                thread1 = threading.Thread(target=self.robot1_controller._move_thread, args=(home_coords_1, 2000), daemon=True)
                thread2 = threading.Thread(target=self.robot2_controller._move_thread, args=(home_coords_2, 2000), daemon=True)
                
                thread1.start()
                thread2.start()
                thread1.join()
                thread2.join()
                
                self.robot1_status.config(text="Robot 1: Moved to home")
                self.robot2_status.config(text="Robot 2: Moved to home")
                self.status_bar.config(text="Both robots moved to home position")
            except Exception as e:
                self.status_bar.config(text=f"Home movement error: {str(e)}")
        
        threading.Thread(target=move_home_thread, daemon=True).start()

    def sync_gripper(self, state):
        def sync_gripper_thread():
            try:
                thread1 = threading.Thread(target=self.robot1_controller._gripper_thread, args=(state,), daemon=True)
                thread2 = threading.Thread(target=self.robot2_controller._gripper_thread, args=(state,), daemon=True)
                thread1.start()
                thread2.start()
                thread1.join()
                thread2.join()
                
                action = "Opened" if state == GRIPPER_OPEN else "Closed"
                self.robot1_status.config(text=f"Robot 1: Gripper {action}")
                self.robot2_status.config(text=f"Robot 2: Gripper {action}")
                self.status_bar.config(text=f"Both grippers {action.lower()}")
            except Exception as e:
                self.status_bar.config(text=f"Sync gripper error: {str(e)}")
        
        threading.Thread(target=sync_gripper_thread, daemon=True).start()

    def on_close(self):
        self.udp_running = False
        self.robot1_controller.stop_monitoring()
        self.robot2_controller.stop_monitoring()
        try:
            self.udp_socket1.close()
            self.udp_socket2.close()
        except Exception:
            pass
        self.destroy()

def connect_to_robot(ip, port, robot_name):
    try:
        print(f"Attempting to connect to {robot_name} at {ip}:{port}...")
        robot = ElephantRobot(ip, port)
        robot.start_client()
        time.sleep(3)
        try:
            angles = robot.get_angles()
            print(f"Successfully connected to {robot_name}. Current angles: {angles}")
            return robot
        except Exception as test_error:
            print(f"Connection test failed for {robot_name}: {test_error}")
            try:
                robot.stop_client()
            except:
                pass
            return None
    except Exception as e:
        print(f"Connection error for {robot_name}: {e}")
        return None

if __name__ == "__main__":
    print("Initializing Dual Robot Control System with Enhanced Connection Recovery...")
    print(f"Robot 1: {ROBOT1_IP}:{ROBOT1_PORT}")
    print(f"Robot 2: {ROBOT2_IP}:{ROBOT2_PORT}")
    print(f"UDP Ports - Robot 1: {UDP_PORT_ROBOT1}, Robot 2: {UDP_PORT_ROBOT2}")
    
    robot1_planes = {
        'p1': Plane('p1', {'x': (213, 394), 'y': (-58, 300), 'z': (-194, 417)},
                    [210, 49.624, 85.891, -86.094, -29.778, -36.692]),
        'p2': Plane('p2', {'x': (235, 358), 'y': (-325, 343), 'z': (-194, 483)},
                    [210, 49.624, 85.891, -86.094, -26.778, -85.337]),
    }

    robot2_planes = {
        'pA': Plane('pA', {'x': (235, 441), 'y': (-250, 49.624), 'z': (-100, 378)},
                    [210, 49.624, 85.891, -88.213, -21.875, -165.173]),
        'pB': Plane('pB', {'x': (-378, -129), 'y': (219, 477), 'z': (187, 504)},
                    [161, 367, 474, 178.953, -0.463, -155.362]),
    }
    
    robot1 = connect_to_robot(ROBOT1_IP, ROBOT1_PORT, "Robot 1")
    robot2 = connect_to_robot(ROBOT2_IP, ROBOT2_PORT, "Robot 2")
    
    if robot1 and robot2:
        try:
            print("Initializing robots...")
            time.sleep(2)
            try:
                robot1.set_speed(DEFAULT_SPEED)
                print("Robot 1 speed set successfully")
            except Exception as e:
                print(f"Warning: Could not set speed for Robot 1: {e}")
            try:
                robot2.set_speed(DEFAULT_SPEED)
                print("Robot 2 speed set successfully")
            except Exception as e:
                print(f"Warning: Could not set speed for Robot 2: {e}")
            
            print("Both robots initialized successfully!")
            print("\nUDP Command Format:")
            print("Robot 1 (port 5005): {'x': 300, 'y': 50, 'z': 100, 'plane': 'p1', 'io': {'output': {'0': 1}, 'tool_output': {'0': 1}}}")
            print("Robot 2 (port 5006): {'x': -200, 'y': 300, 'z': 200, 'plane': 'pA', 'io': {'output': {'1': 0}, 'tool_output': {'1': 0}}}")
            print("\nEnhanced connection recovery is now active!")
            
            app = DualRobotControlUI(robot1, robot2, robot1_planes, robot2_planes)
            app.mainloop()
        except Exception as e:
            print(f"Error during robot initialization: {e}")
        finally:
            try:
                robot1.stop_client()
                print("Robot 1 connection closed.")
            except Exception as e:
                print(f"Error closing Robot 1: {e}")
            try:
                robot2.stop_client()
                print("Robot 2 connection closed.")
            except Exception as e:
                print(f"Error closing Robot 2: {e}")
    elif robot1:
        print("Only Robot 1 connected. Robot 2 connection failed.")
        print("Starting single robot mode...")
        try:
            robot1.set_speed(DEFAULT_SPEED)
            print("Single robot mode not implemented yet. Closing...")
            robot1.stop_client()
        except Exception as e:
            print(f"Error with Robot 1: {e}")
            try:
                robot1.stop_client()
            except Exception:
                pass
    elif robot2:
        print("Only Robot 2 connected. Robot 1 connection failed.")
        print("Starting single robot mode...")
        try:
            robot2.set_speed(DEFAULT_SPEED)
            print("Single robot mode not implemented yet. Closing...")
            robot2.stop_client()
        except Exception as e:
            print(f"Error with Robot 2: {e}")
            try:
                robot2.stop_client()
            except Exception:
                pass
    else:
        print("Failed to connect to both robots. Please check:")
        print("1. IP addresses and ports are correct")
        print("2. Both robots are powered on")
        print("3. Network connectivity to both robots")
        print("4. No firewall blocking the connections")
        print("5. Robots are not already connected to another client")