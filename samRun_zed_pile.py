import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pyzed.sl as sl
import socket
import json
import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


UDP_IP = "255.255.255.255"  # or your PC's IP if remote
UDP_PORT = 5006

#----------------------------------------------------------------------------------#
# pile Camera image points (pixels)
image_points_pile = np.array([
    [592.3, 310],   # camera point 1
    [781.4, 304.6],   # camera point 2
    [792, 486],   # camera point 3
    [600.2, 493.7]    # camera point 4
], dtype=np.float32)

# pile Corresponding real-world points (in mm)
real_world_points_pile = np.array([
    [-119, 484],  # real coordinate 1
    [-119, 221],   # real coordinate 2
    [-364, 221],   # real coordinate 3
    [-364, 484]     # real coordinate 4
], dtype=np.float32)

# Compute the perspective transformation matrix
matrix_pile = cv2.getPerspectiveTransform(image_points_pile, real_world_points_pile)

#========== ZED Camera Setup ==========
def get_zed_frame():
    # Get list of available devices
    device_list = sl.Camera.get_device_list()
    
    ZED_MINI_CAMERA_SN = 14227833  # Serial number of the camera zed 2i

    # Find the cameras by serial number
    zed_mini_camera_device = None
    
    
    for device in device_list:
        if device.serial_number == ZED_MINI_CAMERA_SN:
            zed_mini_camera_device = device
            print(f"Found camera test: ZED {device.camera_model} - SN: {device.serial_number}")    
    
 
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.set_from_serial_number(zed_mini_camera_device.serial_number)

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED Camera not found.")
        exit()

    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    # Grab one frame
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

        rgb_image = image_zed.get_data()

        zed.close()
        return rgb_image, depth_zed
    else:
        print("Failed to grab ZED frame")
        zed.close()
        exit()

# ========== Load RGB + Depth ==========
#rgb_image, depth_zed = get_zed_frame()
#image = Image.fromarray(rgb_image).convert("RGB")

#print(f"rgb_image: {rgb_image}")
#print(f"RGB image shape: {rgb_image.shape}")    # Should be (height, width, 3)
   		
# ========== Select Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to map a camera point to real-world point
def map_to_real_world_pile(camera_x, camera_y):
    """Map a camera point to real-world coordinate"""
    pt = np.array([[camera_x, camera_y]], dtype='float32')
    pt = np.array([pt])
    real_pt = cv2.perspectiveTransform(pt, matrix_pile)
    return real_pt[0][0]  # Returns (x_mm, y_mm) return real_pt[0][0]  # Returns (x_mm, y_mm)

def send_command_pile(x, y, z):
    """Send coordinates via UDP"""
    cal_z = round(float(z), 3)
    
    real_coords = map_to_real_world_pile(x, y)
    real_x_mm = round(real_coords[0], 3)
    real_y_mm = round(real_coords[1], 3)
    
    # Adjusted X based on depth calibration logic
    adjusted_z = round(float(900.00 + 185.00 - cal_z), 3)
    print(f"Mapped real-world coordinates for pile: ({real_x_mm}, {real_y_mm})")
    print(f"Adjusted X (Z-corrected) pile: {adjusted_z}")

    message_pile = json.dumps({
        "x": int(real_x_mm),
        "y": int(real_y_mm),
        "z": adjusted_z-25,  # total distance between arm and cam + cal_z + initial pos of z
        "plane":"pB"
    })
    print(f"message_pile: {message_pile}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(message_pile.encode(), (UDP_IP, UDP_PORT))
    sock.close()
    print("xyz getting process ended")


def process_lowest_depth():
    # Define ROI: 
    x_min = 588
    x_max = 784
    y_min = 314
    y_max = 492

    """Find the lowest depth point in the current view"""
    print("Finding lowest depth point...")
    
    
    rgb_image, depth_zed = get_zed_frame()
    if rgb_image is None:
        print("Failed to capture image")
        return
        
    image = Image.fromarray(rgb_image).convert("RGB")
        
    # Find the lowest depth point in the entire image
    x, y, z = find_lowest_depth_point(depth_zed, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    #x, y, z = find_lowest_depth_point(depth_zed)
        
    # Visualize the result
    visualize_lowest_depth_point(np.array(image), x, y, z)
        
    # Send the coordinates
    send_command_pile(x, y, z)


def find_lowest_depth_point(depth_zed, mask=None, x_min=0, x_max=None, y_min=0, y_max=None):
    """Find the point with the lowest depth value (closest to camera)"""
    # Get depth map dimensions
    width = depth_zed.get_width()
    height = depth_zed.get_height()
    
    # Set default limits if None
    x_max = x_max if x_max is not None else width
    y_max = y_max if y_max is not None else height
    
    lowest_depth = float('inf')
    lowest_x, lowest_y = 0, 0
    
    # Create a grid of points to sample (can be adjusted for performance)
    step = 1  # Check every 5th pixel for better performance
    
    for y in range(y_min, y_max, step):
        for x in range(x_min, x_max, step):
            # If mask is provided, only check points within mask
            if mask is not None:
                if not mask[y, x]:
                    continue
            
            # Get depth value at this point
            err, depth_value = depth_zed.get_value(x, y)
            
            # Check if this is a valid depth and if it's the lowest so far
            if err == sl.ERROR_CODE.SUCCESS and depth_value > 0 and depth_value < lowest_depth:
                lowest_depth = depth_value
                lowest_x, lowest_y = x, y
    
    print(f"Lowest depth point found: X={lowest_x}, Y={lowest_y}, Z={lowest_depth}mm")
    return lowest_x, lowest_y, lowest_depth


def visualize_lowest_depth_point(image, x, y, z):
    """Create visualization for the lowest depth point"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.scatter(x, y, c='red', s=100, edgecolors='black', label=f"X={x}, Y={y}, Z={z}mm")
    plt.legend(loc='upper right')
    plt.title(f"Lowest Depth Point (Closest to Camera)")
    plt.axis('off')
    plt.savefig(f"zed_outputs/lowest_depth_{time.strftime('%Y%m%d-%H%M%S')}.png")
    plt.close()
    
    
process_lowest_depth()
