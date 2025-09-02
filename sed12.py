import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pyzed.sl as sl
import socket
import json
import threading
import time
from rembg import remove, new_session
from transformers import CLIPProcessor, CLIPModel
import math
import sys


# Prompts
category_prompts = ["a t-shirt", "a long-sleeved shirt", "a hoodie", "a sweatshirt", "a pullover", "a tank top"]
color_prompts = ["a red garment", "a blue garment", "a black garment", "a white garment", "a green garment", "a yellow garment", "a gray garment", "a brown garment", "a pink garment", "a purple garment"]
pattern_prompts = ["a plain shirt", "a striped shirt", "a floral shirt", "a checked shirt", "a dotted shirt", "an abstract patterned shirt"]
fit_prompts = ["a slim fit shirt", "an oversized top", "a regular fit shirt", "a cropped shirt", "a shirt with a crew neck", "a shirt with a v-neck", "a shirt with a round neckline"]

# Network configuration
UDP_IP = "255.255.255.255"  # Broadcast address
UDP_PORT1 = 5005 #arm1 
UDP_PORT2 = 5006 #arm2 / pile arm
UDP_PORT3 = 5008
UDP_PORT4 = 5010 #5010
TCP_PORT = 3007


#----------------------------------------------------------------------------------#
# Camera image points (pixels) for arm 1
image_points1 = np.array([
    [889.3, 462.3],   # camera point 1
    [912.4, 1183.6],   # camera point 2
    [1315.2, 1208.3],   # camera point 3
    [1337.1, 484.1]    # camera point 4
], dtype=np.float32)

# Corresponding real-world points (in mm)
real_world_points1 = np.array([
    [-58, 417],  # real coordinate 1
    [-58, -194],   # real coordinate 2
    [300, -194],   # real coordinate 3
    [300, 417]     # real coordinate 4
], dtype=np.float32)

# Compute the perspective transformation matrix
matrix1= cv2.getPerspectiveTransform(image_points1, real_world_points1)


#----------------------------------------------------------------------------------#
# Camera image points (pixels) for arm 2
image_points2 = np.array([
    [1054.7, 546.7],   # camera point 1
    [1060, 1074.4],   # camera point 2
    [1385.6, 1073.8],   # camera point 3
    [1403.5, 545.2]    # camera point 4
], dtype=np.float32)

# Corresponding real-world points (in mm)
real_world_points2 = np.array([
    [-250, 378],  # real coordinate 1
    [-250, -100],   # real coordinate 2
    [49.624, -100],   # real coordinate 3
    [49.624, 378]     # real coordinate 4
], dtype=np.float32)

# Compute the perspective transformation matrix
matrix2= cv2.getPerspectiveTransform(image_points2, real_world_points2)

#----------------------------------------------------------------------------------#
# pile Camera image points (pixels)
image_points_pile = np.array([
    [1355.4, 516.3],   # camera point 1
    [1374.0, 884.2],   # camera point 2
    [983.2, 869.0],   # camera point 3
    [970.1, 527.2]    # camera point 4
], dtype=np.float32)

# pile Corresponding real-world points (in mm)
real_world_points_pile = np.array([
    [-129, 219],  # real coordinate 1
    [-378, 219],   # real coordinate 2
    [-378, 477],   # real coordinate 3
    [-129, 477]     # real coordinate 4
], dtype=np.float32)
# Compute the perspective transformation matrix
matrix_pile = cv2.getPerspectiveTransform(image_points_pile, real_world_points_pile)

#----------------------------------------------------------------------------------#
# Global variables
capture_lock = threading.Lock()
capture_event = threading.Event()

current_point = None
is_camera_initialized1 = False
is_camera_initialized2 = False
zed1 = None
zed2 = None
predictor = None

fin1_temp = False
fin2_temp = False
a1_state = None 
a2_state = None 

def initialize_camera1():
    """Initialize the ZED camera once"""
    global zed1, is_camera_initialized1
    
    if is_camera_initialized1:
        return True
        
    # Get list of available devices
    device_list = sl.Camera.get_device_list()
    
    ZED_MINI_CAMERA_SN = 14227833  # Serial number of the camera zed mimi

    # Find the cameras by serial number
    zed_mini_camera_device = None
    
    for device in device_list:
        if device.serial_number == ZED_MINI_CAMERA_SN:
            zed_mini_camera_device = device
            print(f"Found camera test: ZED {device.camera_model} - SN: {device.serial_number}")  
   
    zed1 = sl.Camera()
    init1 = sl.InitParameters()
    init1.camera_resolution = sl.RESOLUTION.HD2K
    #init1.camera_resolution = sl.RESOLUTION.HD720
    init1.camera_fps = 15  # instead of 60
    init1.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init1.coordinate_units = sl.UNIT.MILLIMETER
    # Set optimal depth range for better accuracy
    init1.depth_minimum_distance = 700 # 70cm minimum
    init1.depth_maximum_distance =  1100 # 110cm maximum
    init1.set_from_serial_number(zed_mini_camera_device.serial_number)
    
    if zed1.open(init1) != sl.ERROR_CODE.SUCCESS:
        print("ZED Camera 1 not found.")
        return False
    
    is_camera_initialized1 = True
    print("Camera 1 initialized successfully")
    return True

def initialize_camera2():
    """Initialize the ZED camera once"""
    global zed2, is_camera_initialized2
    
    if is_camera_initialized2:
        return True
        
    # Get list of available devices
    device_list = sl.Camera.get_device_list()
    
    ZED_2I_CAMERA_SN = 32538075  # Serial number of the camera zed 2i

    # Find the cameras by serial number
    zed_2i_camera_device = None
    
    for device in device_list:
        if device.serial_number == ZED_2I_CAMERA_SN:
            zed_2i_camera_device = device
            print(f"Found LEFT camera: ZED {device.camera_model} - SN: {device.serial_number}")    
    
    zed2 = sl.Camera()
    init2 = sl.InitParameters()
    init2.camera_resolution = sl.RESOLUTION.HD2K
    #init2.camera_resolution = sl.RESOLUTION.HD720
    init2.camera_fps = 15  # instead of 60
    init2.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS  # instead of NEURAL_PLUS on one camera / ULTRA
    init2.coordinate_units = sl.UNIT.MILLIMETER
    # Set optimal depth range for better accuracy
    init2.depth_minimum_distance = 700  # 70cm minimum
    init2.depth_maximum_distance = 1100  # 110cm maximum
    init2.set_from_serial_number(zed_2i_camera_device.serial_number)
    
    if zed2.open(init2) != sl.ERROR_CODE.SUCCESS:
        print("ZED Camera 2 not found.")
        return False
    
    is_camera_initialized2 = True
    print("Camera 2 initialized successfully")
    return True

def initialize_model():
    global processor
    global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prompts
    category_prompts = ["a t-shirt", "a long-sleeved shirt", "a hoodie", "a sweatshirt", "a pullover", "a tank top"]
    color_prompts = ["a red garment", "a blue garment", "a black garment", "a white garment", "a green garment", "a yellow garment", "a gray garment", "a brown garment", "a pink garment", "a purple garment"]
    pattern_prompts = ["a plain shirt", "a striped shirt", "a floral shirt", "a checked shirt", "a dotted shirt", "an abstract patterned shirt"]
    fit_prompts = ["a slim fit shirt", "an oversized top", "a regular fit shirt", "a cropped shirt", "a shirt with a crew neck", "a shirt with a v-neck", "a shirt with a round neckline"]

    # Load FashionCLIP model and processor
    model = CLIPModel.from_pretrained("./models/fashion-clip")
    processor = CLIPProcessor.from_pretrained("./models/fashion-clip")

    print("model initialized")

def get_zed_frame1():
    """Capture a frame from the ZED camera"""
    global zed1
    
    if not is_camera_initialized1:
        if not initialize_camera1():
            return None, None
    
    runtime = sl.RuntimeParameters()
    image_zed1 = sl.Mat()
    depth_zed1 = sl.Mat()
    
    # Grab one frame
    if zed1.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed1.retrieve_image(image_zed1, sl.VIEW.LEFT)
        zed1.retrieve_measure(depth_zed1, sl.MEASURE.DEPTH)
        
        rgb_image1 = image_zed1.get_data()
        return rgb_image1, depth_zed1
    else:
        print("Failed to grab ZED 1 frame")
        return None, None
        
def get_zed_frame2():
    """Capture a frame from the ZED camera"""
    global zed2
    
    if not is_camera_initialized2:
        if not initialize_camera2():
            return None, None
    
    runtime = sl.RuntimeParameters()
    image_zed2 = sl.Mat()
    depth_zed2 = sl.Mat()
    
    # Grab one frame
    if zed2.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed2.retrieve_image(image_zed2, sl.VIEW.LEFT)
        zed2.retrieve_measure(depth_zed2, sl.MEASURE.DEPTH)
        
        rgb_image2 = image_zed2.get_data()
        return rgb_image2, depth_zed2
    else:
        print("Failed to grab ZED 2 frame")
        return None, None
        
           
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
    plt.scatter(x, y, c='red', s=20, edgecolors='black', label=f"X={x}, Y={y}, Z={z}mm")
    plt.legend(loc='upper right')
    plt.title(f"Lowest Depth Point (Closest to Camera)")
    plt.axis('off')
    plt.savefig(f"zed_outputs/lowest_depth_{time.strftime('%Y%m%d-%H%M%S')}.png")
    plt.close()

def map_to_real_world1(camera_x, camera_y):
    """Map a camera point to real-world coordinate"""
    pt = np.array([[camera_x, camera_y]], dtype='float32')
    pt = np.array([pt])
    real_pt = cv2.perspectiveTransform(pt, matrix1)
    return real_pt[0][0]  # Returns (x_mm, y_mm)
    
def map_to_real_world2(camera_x, camera_y):
    """Map a camera point to real-world coordinate"""
    pt = np.array([[camera_x, camera_y]], dtype='float32')
    pt = np.array([pt])
    real_pt = cv2.perspectiveTransform(pt, matrix2)
    return real_pt[0][0]  # Returns (x_mm, y_mm)
    
def map_to_real_world_pile(camera_x, camera_y):
    """Map a camera point to real-world coordinate"""
    pt = np.array([[camera_x, camera_y]], dtype='float32')
    pt = np.array([pt])
    real_pt = cv2.perspectiveTransform(pt, matrix_pile)
    return real_pt[0][0]  # Returns (x_mm, y_mm)
    
def distance_3d(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
def send_command1(x, y, z):
    """Send coordinates via UDP"""
    cal_z = round(float(z), 3)
    
    real_coords = map_to_real_world1(x, y)
    real_x_mm = round(real_coords[0], 3)
    real_y_mm = (round(real_coords[1], 3)) + 10
    
    # Adjusted X based on depth calibration logic
    adjusted_x = round(float(1100.00 + 235.00 -70 - cal_z), 3)
    print(f"Mapped real-world coordinates: ({real_x_mm}, {real_y_mm})")
    print(f"Adjusted X (Z-corrected): {adjusted_x}")

    message1 = json.dumps({
        "x": adjusted_x,  # total distance between arm and cam + cal_z + initial pos of z
        "y": 38.529,
        "z": int(real_y_mm),
        "plane":"p1"
    })
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(message1.encode(), (UDP_IP, UDP_PORT1))
    sock.close()
    print(f"xyz getting process ended - arm1: {message1}")
    
    time.sleep(2)  # Delay for 2 seconds
    
    message1 = json.dumps({
        "x": adjusted_x,  # total distance between arm and cam + cal_z + initial pos of z
        "y": int(real_x_mm),
        "z": int(real_y_mm),
        "plane":"p1"
    })
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(message1.encode(), (UDP_IP, UDP_PORT1))
    sock.close()
    print(f"xyz getting process ended - arm1: {message1}")
    
    time.sleep(2)  # Delay for 2 seconds
    
    message1 = json.dumps({
      "io": {
        "output": {
          "3": "true"
        }
      }
    })
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(message1.encode(), (UDP_IP, UDP_PORT1))
    sock.close()
    print("arm1 process started")
    time.sleep(2)  # Delay for 2 seconds
      
    point1 = (289, 312, 560)
    point2 = (adjusted_x, real_x_mm, real_y_mm)

    dist = distance_3d(point1, point2)
    print("Distance:", dist)
    
    
    message1 = json.dumps({
        "x": 230,  # total distance between arm and cam + cal_z + initial pos of z
        "y": int(312-(dist*0.45)),
        "z": int(560),
        "plane":"p2"
    })
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(message1.encode(), (UDP_IP, UDP_PORT1))
    sock.close()
    print(f"xyz getting process ended - arm1: {message1}")

def send_command2(x, y, z):
    global a1_state
    global a2_state
    print(f"send_command2 funt a1_state: {a1_state} | a2_state: {a2_state}")
    """Send coordinates via UDP"""
    cal_z = round(float(z), 3)
    
    real_coords = map_to_real_world2(x, y)
    real_x_mm = round(real_coords[0], 3)
    real_y_mm = (round(real_coords[1], 3)) + 10
    
    # Adjusted X based on depth calibration logic
    adjusted_x = round(float(1100.00 + 235.00 - (cal_z+50)), 3)
    print(f"Mapped real-world coordinates: ({real_x_mm}, {real_y_mm})")
    print(f"Adjusted X (Z-corrected): {adjusted_x}")
    
    message2 = json.dumps({
        "x": adjusted_x ,  # total distance between arm and cam + cal_z + initial pos of z
        "y": 40.087,
        "z": int(real_y_mm),
        "plane":"pA"
    })
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(message2.encode(), (UDP_IP, UDP_PORT2))
    sock.close()
    print(f"xyz getting process ended - arm2: {message2}")
    
    time.sleep(3)  # Delay for 2 seconds
    
    message2 = json.dumps({
        "x": adjusted_x,  # total distance between arm and cam + cal_z + initial pos of z
        "y": int(real_x_mm),
        "z": int(real_y_mm),
        "plane":"pA"
    })
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(message2.encode(), (UDP_IP, UDP_PORT2))
    sock.close()
    print(f"xyz getting process ended - arm2: {message2}")
    
    time.sleep(2)  # Delay for 10 seconds

    message_grab_assit2 = "stat2"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(message_grab_assit2.encode(), (UDP_IP, UDP_PORT4))
    sock.close()
    print(f"message_grab_assit2: {message_grab_assit2}")
    time.sleep(4) 
    
     
    print(f"message_grab_assit a2_state: {a2_state}")
    if a2_state == 'a2_true': #two arms doesnt grab
    	print("#arm2 doesnt grab")
    	send_command_reset(UDP_PORT2)
    	message0 = json.dumps({                       
      		"io": {
        	"output": {
          	"8": "true"
        	}
     		}
    	})
    	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    	sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    	sock.sendto(message0.encode(), (UDP_IP, UDP_PORT1))
    	sock.close()
    	
    elif a2_state == 'a2_false': #arm2 grab
    	print("#arm2 grab")

    	message0 = json.dumps({                       
      		"io": {
        	"output": {
          	"4": "true"
        	}
     		}
    	})
    	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    	sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    	sock.sendto(message0.encode(), (UDP_IP, UDP_PORT2))
    	sock.close()
    	print("arm2 process started")  

    	
    else: 
    	print("#arm2 grab error")
    	
    	send_command_reset(UDP_PORT2)
    	message0 = json.dumps({                       
      		"io": {
        	"output": {
          	"8": "true"
        	}
     		}
    	})
    	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    	sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    	sock.sendto(message0.encode(), (UDP_IP, UDP_PORT1))
    	sock.close()

def send_command_pile(x, y, z):
    """Send coordinates via UDP"""
    global a1_state
        
    print(f"send_command_pile funt a1_state: {a1_state}")
    cal_z = round(float(z), 3)
    
    real_coords = map_to_real_world_pile(x, y)
    real_x_mm = round(real_coords[0], 3)
    real_y_mm = round(real_coords[1], 3)
    
    # Adjusted X based on depth calibration logic
    adjusted_z = round(float(1000.00 + 187.00 - (cal_z+10)), 3)
    print(f"Mapped real-world coordinates for pile: ({real_x_mm}, {real_y_mm})")
    print(f"Adjusted X (Z-corrected) pile: {adjusted_z}")

    message_pile = json.dumps({
        "x": int(real_x_mm+19),
        "y": int(real_y_mm+25),
        "z": adjusted_z-50,  # total distance between arm and cam + cal_z + initial pos of z
        "plane":"pB"
    })
    print(f"message_pile: {message_pile}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(message_pile.encode(), (UDP_IP, UDP_PORT2))
    sock.close()
    print("xyz getting process ended")
    time.sleep(8)  # Delay for 2 seconds
    
    message_grab_assit1 = "stat"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(message_grab_assit1.encode(), (UDP_IP, UDP_PORT4))
    sock.close()
    print(f"message_grab_assit1: {message_grab_assit1}")
    
    time.sleep(1) 
    
    print(f"message_grab_assit a1_state: {a1_state}")
    if a1_state == 'a1_false': 
    	message0 = json.dumps({                       
      		"io": {
        	"output": {
          	"1": "true"
        	}
      	}
    })
    
    elif a1_state == 'a1_true': 
    	message0 = json.dumps({                       
      		"io": {
        	"output": {
          	"6": "true"
        	}
      	}
    })    

    else: 
    	message0 = json.dumps({                       
      		"io": {
        	"output": {
          	"1": "true"
        	}
      	}
    })
        
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(message0.encode(), (UDP_IP, UDP_PORT2))
    sock.close()
    print("arm2 pile process started")

def send_command_reset(ARM_UDP_PORT):
    global process_state
    
    process_state="cp0"
    print("send_command_reset - io 7:true")
    message_io7 = json.dumps({
        "io": {
            "output": {
                "7": "true"
            }
        }
    })
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(message_io7.encode(), (UDP_IP, ARM_UDP_PORT))
    sock.close()

    #time.sleep(1)
    #print("send_command_reset - io 0:false")

    message_io7 = json.dumps({
        "io": {
            "output": {
                "0": "false"
            }
        }
    })
    #sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    #sock.sendto(message_io7.encode(), (UDP_IP, ARM_UDP_PORT))
    #sock.close()
    
    #time.sleep(1)
    
    #print("send_command_reset - io 7:false")
    message_io7 = json.dumps({
        "io": {
            "output": {
                "7": "false"
            }
        }
    })
    #sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    #sock.sendto(message_io7.encode(), (UDP_IP, ARM_UDP_PORT))
    #sock.close()


    #time.sleep(3)
    #print("send_command_reset - io 0:true")
    
    message_io7 = json.dumps({
        "io": {
            "output": {
                "0": "true"
            }
        }
    })
    #sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    #sock.sendto(message_io7.encode(), (UDP_IP, ARM_UDP_PORT))
    #sock.close()

def process_image1():
    # Define ROI (cropped region)
    x_min = 1026
    x_max = 1334
    y_min = 264
    y_max = 1170

    with capture_lock:
        rgb_image, depth_zed = get_zed_frame2()
        if rgb_image is None:
            print("Failed to capture image 1")
            send_command_reset(UDP_PORT2)
            return
        overlay = Image.fromarray(rgb_image).convert("RGB")
        
        # Crop the original image
        cropped = rgb_image[y_min:y_max, x_min:x_max]
        image = Image.fromarray(cropped).convert("RGB")
        image_np = np.array(image)

        # Background removal (creates binary mask image with transparent BG)
        image_no_bg_np = remove(image_np)  # RGBA output
        segmented_image = Image.fromarray(image_no_bg_np)
        segmented_image.save("arm1_segmented_output.png")
        print("Segmented image saved as segmented_output.png")
        
        print("Original cropped shape:", image_np.shape)
        print("After background removal:", image_no_bg_np.shape)

        # ----------------------------
        # Find lowest mask pixel in cropped image
        # ----------------------------
        # Get only the mask (True = foreground)
        #mask = remove(image_np, only_mask=True)  # shape: (H, W)
        #mask = (mask > 0).astype(np.uint8)
        
        #mask = (remove(image_np, only_mask=True) > 0).astype(np.uint8)
        #mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=10)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        alpha = image_no_bg_np[:, :, 3]
        mask = (alpha > 1).astype(np.uint8) #(alpha > 10).astype(np.uint8) 
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=10)

        # Debug: show or save the mask to verify
        cv2.imwrite("arm1_debug_mask.png", mask * 255)

        # Find foreground coordinates
        coords = np.column_stack(np.where(mask > 0))  # (y, x)
        if coords.size == 0:
            print("No valid mask region found.")
            send_command_reset(UDP_PORT2)
            return
            
        # Convert PIL overlay to NumPy if not already
        overlay_np = np.array(overlay)

        # Create a blank mask with full image size
        full_mask = np.zeros(overlay_np.shape[:2], dtype=np.uint8)

        # Paste cropped mask into its correct location
        full_mask[y_min:y_max, x_min:x_max] = mask
        
        alpha = 0.4  # Transparency level (0 = fully transparent, 1 = solid)
        green_layer = np.zeros_like(overlay_np, dtype=np.uint8)
        green_layer[:, :] = (0, 255, 0)  # solid green
        # Apply green color overlay where mask is present
        overlay_np[full_mask > 0] = cv2.addWeighted(overlay_np[full_mask > 0], 1 - alpha,green_layer[full_mask > 0], alpha, 0)

        # Step 1: Create a valid depth mask (same size as cropped mask)
        valid_depth_mask = np.zeros(mask.shape, dtype=np.uint8)

        for (y, x) in coords:
            full_x = int(x_min + x)
            full_y = int(y_min + y)
            err, depth_val = depth_zed.get_value(full_x, full_y)
            if err == sl.ERROR_CODE.SUCCESS and 0 < depth_val < 2000:
                valid_depth_mask[y, x] = 1

        # Step 2: Combine with original mask
        combined_mask = cv2.bitwise_and(mask, valid_depth_mask)

        # Step 3: Find coords again (only where depth is valid + mask is 1)
        valid_coords = np.column_stack(np.where(combined_mask > 0))

        if valid_coords.size == 0:
            print("No valid depth points found inside mask.")
            send_command_reset(UDP_PORT2)
            return

        # Step 4: Pick the lowest Y point from valid region
        lowest_idx = np.argmax(valid_coords[:, 0])
        crop_y, crop_x = valid_coords[lowest_idx]

        print(f"arm 1 crop value cordinate: {crop_x},{crop_y}")
        
        # Map to full-res
        full_x = int(x_min + crop_x)
        full_y = int(y_min + crop_y-30)
        
        print(f"arm 1 depth value cordinate: {full_x},{full_y}")
        err, z = depth_zed.get_value(full_x, full_y)
        if err != sl.ERROR_CODE.SUCCESS or z <= 0:
            print("Failed to get valid depth value.")
            send_command_reset(UDP_PORT2)
            return

        # Visualize on full image
        visualize_lowest_depth_point(overlay_np, full_x, full_y, z)
        
        if math.isinf(z) or math.isnan(z):
            print(f"[WARNING for arm1] Adjusted X is infinite. Triggering IO output 7. z : {z}")
            send_command_reset(UDP_PORT2)
            return  # Stop further execution if depth is invalid

        # Send command to arm1
        send_command1(full_x, full_y, z)

def process_image2():
    # Define ROI (cropped region)
    x_min = 1000
    x_max = 1300
    y_min = 264
    y_max = 1122

    with capture_lock:
        print("wait capture image 2 till arm 1 move")
        time.sleep(2)  # Delay for 2 seconds
        rgb_image, depth_zed = get_zed_frame2()
        if rgb_image is None:
            print("Failed to capture image 1")
            send_command_reset(UDP_PORT2)
            return
        overlay = Image.fromarray(rgb_image).convert("RGB")
        
        # Crop the original image
        cropped = rgb_image[y_min:y_max, x_min:x_max]
        image = Image.fromarray(cropped).convert("RGB")
        image_np = np.array(image)

        # Background removal (creates binary mask image with transparent BG)
        image_no_bg_np = remove(image_np)  # RGBA output
        segmented_image = Image.fromarray(image_no_bg_np)
        segmented_image.save("arm2_segmented_output.png")
        print("Segmented image saved as segmented_output.png")
        
        print("Original cropped shape:", image_np.shape)
        print("After background removal:", image_no_bg_np.shape)

        # ----------------------------
        # Find lowest mask pixel in cropped image
        # ----------------------------
        # Get only the mask (True = foreground)
        #mask = remove(image_np, only_mask=True)  # shape: (H, W)
        #mask = (mask > 0).astype(np.uint8)
        
        #mask = (remove(image_np, only_mask=True) > 0).astype(np.uint8)
        #mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=10)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        alpha = image_no_bg_np[:, :, 3]
        mask = (alpha > 1).astype(np.uint8) #(alpha > 10).astype(np.uint8) 
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=10)

        # Debug: show or save the mask to verify
        cv2.imwrite("arm2_debug_mask.png", mask * 255)

        # Find foreground coordinates
        coords = np.column_stack(np.where(mask > 0))  # (y, x)
        if coords.size == 0:
            print("No valid mask region found.")
            send_command_reset(UDP_PORT2)
            return
            
        # Convert PIL overlay to NumPy if not already
        overlay_np = np.array(overlay)

        # Create a blank mask with full image size
        full_mask = np.zeros(overlay_np.shape[:2], dtype=np.uint8)

        # Paste cropped mask into its correct location
        full_mask[y_min:y_max, x_min:x_max] = mask
        
        alpha = 0.4  # Transparency level (0 = fully transparent, 1 = solid)
        green_layer = np.zeros_like(overlay_np, dtype=np.uint8)
        green_layer[:, :] = (0, 255, 0)  # solid green
        # Apply green color overlay where mask is present
        overlay_np[full_mask > 0] = cv2.addWeighted(overlay_np[full_mask > 0], 1 - alpha,green_layer[full_mask > 0], alpha, 0)

        
        # Step 1: Create a valid depth mask (same size as cropped mask)
        valid_depth_mask = np.zeros(mask.shape, dtype=np.uint8)

        for (y, x) in coords:
            full_x = int(x_min + x)
            full_y = int(y_min + y)
            err, depth_val = depth_zed.get_value(full_x, full_y)
            if err == sl.ERROR_CODE.SUCCESS and 0 < depth_val < 2000:
                valid_depth_mask[y, x] = 1

        # Step 2: Combine with original mask
        combined_mask = cv2.bitwise_and(mask, valid_depth_mask)

        # Step 3: Find coords again (only where depth is valid + mask is 1)
        valid_coords = np.column_stack(np.where(combined_mask > 0))

        if valid_coords.size == 0:
            print("No valid depth points found inside mask.")
            send_command_reset(UDP_PORT2)
            return

        # Step 4: Pick the lowest Y point from valid region
        lowest_idx = np.argmax(valid_coords[:, 0])
        crop_y, crop_x = valid_coords[lowest_idx]

        print(f"arm 2 crop value cordinate: {crop_x},{crop_y}")
        
        # Map to full-res
        full_x = int(x_min + crop_x)
        full_y = int(y_min + crop_y-30)
        
        print(f"arm 2 depth value cordinate: {full_x},{full_y}")
        err, z = depth_zed.get_value(full_x, full_y)


        if err != sl.ERROR_CODE.SUCCESS or z <= 0:
            print("Failed to get valid depth value.")
            send_command_reset(UDP_PORT2)
            return
            
        # Visualize on full image
        visualize_lowest_depth_point(overlay_np, full_x, full_y, z)
                    
        if math.isinf(z) or math.isnan(z):
            print(f"[WARNING for arm2] Adjusted X is infinite. Triggering IO output 7. z: {z}")
            send_command_reset(UDP_PORT2)
            return  # Stop further execution if depth is invalid


        # Send command to arm1
        send_command2(full_x, full_y, z)

def process_lowest_depth():

    # Define ROI: 
    x_min = 970
    x_max = 1374
    y_min = 516
    y_max = 884

    """Find the lowest depth point in the current view"""
    print("Finding lowest depth point...")
    
    with capture_lock:
        rgb_image, depth_zed = get_zed_frame1()
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

def work_done():
    global fin1_temp, fin2_temp, process_state, a1_state, a2_state
    
    
    message_grab_assit2 = "stat2"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(message_grab_assit2.encode(), (UDP_IP, UDP_PORT4))
    sock.close()
    print(f"message_grab_assit2: {message_grab_assit2}")
    
    time.sleep(0.5)
    message_grab_assit1 = "stat"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(message_grab_assit1.encode(), (UDP_IP, UDP_PORT4))
    sock.close()
    print(f"message_grab_assit1: {message_grab_assit1}")
    time.sleep(0.5)

    
    print(f"message_grab_assit a1_state: {a1_state} | a2_state: {a2_state}")
    if a1_state == 'a1_true' or a2_state == 'a2_true': #two arms doesnt grab or one of arms doesnt grab
        print("#two arms doesnt grab")
        send_command_reset(UDP_PORT2)
        message0 = json.dumps({                       
            "io": {
            "output": {
            "8": "true"
            }
            }
        })
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
        sock.sendto(message0.encode(), (UDP_IP, UDP_PORT1))
        sock.close()
    	    	
   
    elif a1_state == 'a1_false' and a1_state == 'a1_false': #arm1 grab and arm2 grab
        print("#arm1 grab, arm2 grab")

        process_state="cp0"
        message5 = json.dumps({
            "io": {
            "output": {
            "5": "true"
            }
            }
        })
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
        sock.sendto(message5.encode(), (UDP_IP, UDP_PORT1))
        sock.close()
        time.sleep(0.1)

        message5 = json.dumps({
            "io": {
            "output": {
            "5": "true"
            }
            }
        })
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
        sock.sendto(message5.encode(), (UDP_IP, UDP_PORT2))
        sock.close()
        print("full process done\n----------------------\n")
        fin1_temp = False
        fin2_temp = False

        time.sleep(5) 

        messageIN = json.dumps({
            "io": {
            "output": {
            "2": "true"
            }
            }
        })
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
        sock.sendto(messageIN.encode(), (UDP_IP, UDP_PORT1))
        sock.close()
        print("send arm1 process to start...")
        time.sleep(0.1)

        messageIN = json.dumps({
            "io": {
            "output": {
            "0": "true"
            }
            }
        })
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
        sock.sendto(messageIN.encode(), (UDP_IP, UDP_PORT2))
        sock.close()
        print("send arm2 process to start...")


    else: 
        print("#arm1 and arm2 grab error")

        send_command_reset(UDP_PORT2)
        message0 = json.dumps({                       
            "io": {
            "output": {
            "8": "true"
            }
            }
        })
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
        sock.sendto(message0.encode(), (UDP_IP, UDP_PORT1))
        sock.close()
    

# Predict best matching prompt from a group
def predict_best_prompt(image, prompts):
    global processor
    global model
    inputs = processor(text=prompts, images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze().tolist()
    best_idx = torch.tensor(probs).argmax().item()
    return prompts[best_idx], probs[best_idx]

def process_and_send0():
    """Find lowest depth point (closest to camera)"""
    print("Processing with configuration 0 (lowest depth)...")
    process_lowest_depth()

def process_and_send1():
    """Process with configuration 1"""
    print("Processing with configuration 1...")
    process_image1()

def process_and_send2():
    """Process with configuration 2"""
    print("Processing with configuration 2...")
    process_image2()

def process_and_send3():
    """Process with configuration 3"""
    print("Processing with configuration 3...")
    process_image1()

def camera_thread1():
    """Thread for keeping camera alive"""
    while True:
        if not is_camera_initialized1:
            initialize_camera1()
            
        # Just keep the camera warm by grabbing frames occasionally
        if not capture_event.is_set():
            with capture_lock:
                get_zed_frame1()
        
        time.sleep(0.1)  # Small sleep to prevent CPU hogging

def camera_thread2():
    """Thread for keeping camera alive"""
    while True:
        if not is_camera_initialized2:
            initialize_camera2()
            
        # Just keep the camera warm by grabbing frames occasionally
        if not capture_event.is_set():
            with capture_lock:
                get_zed_frame2()
        
        time.sleep(0.1)  # Small sleep to prevent CPU hogging

def udp_listner_thread():
    global a1_state
    global a2_state
   
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)     
    sock.bind(('', UDP_PORT3))
    print(f"listning UDP {UDP_PORT3}")
    
    while True:
    	try:
            data, _ = sock.recvfrom(1024)
            decoded_data = data.decode('utf-8').strip().lower()
            print (f"receved UDP data {UDP_PORT3}: {decoded_data}") 
            
            if decoded_data in ['a1_true', 'a1_false']:
            	a1_state = decoded_data
            	print (f"a1 state  {a1_state}")
            	
            if decoded_data in ['a2_true', 'a2_false']:
            	a2_state = decoded_data
            	print (f"a2 state  {a2_state}")
            	
    	except Exception as e:
                print (f"error in udp: {e}")		          	




def tcp_listener_thread():
    global fin1_temp, fin2_temp, process_state
    """Thread for listening to TCP commands"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
     
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)# Add this line to allow port reuse
    
    server_address = ('', TCP_PORT)
    server_socket.bind(server_address)
    server_socket.listen(1)

    print(f"Listening for TCP messages on port {TCP_PORT}...")
    
    messageIN = json.dumps({
      "io": {
        "output": {
          "2": "true"
        }
      }
    })
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(messageIN.encode(), (UDP_IP, UDP_PORT1))
    sock.close()
    print("send arm1 process to start...")


    messageIN = json.dumps({
      "io": {
        "output": {
          "0": "true"
        }
      }
    })
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    sock.sendto(messageIN.encode(), (UDP_IP, UDP_PORT2))
    sock.close()
    print("send arm2 process to start...")
    
    try:
        while True:
            print("Waiting for a connection...")
            connection, client_address = server_socket.accept()
            try:
                print(f"Connection from {client_address}")
                while True:
                    data = connection.recv(1024)
                    if data:
                        decoded_data = data.decode('utf-8').strip().lower()
                        print(f"Received: {decoded_data}")
                        
                        capture_event.set()  # Signal that we're capturing
                        
                        if decoded_data == 'cp0':
                            process_state="cp0"
                            process_and_send0()
                        elif decoded_data == 'cp1'and process_state == "cp0":
                            process_state="cp1"
                            process_and_send1()
                        elif decoded_data == 'cp2'and process_state == "cp1":
                            process_state="cp2"
                            process_and_send2()
                        elif decoded_data == 'cp3':
                            process_state="cp3"
                            process_and_send3()
                        elif decoded_data == 'fin1':
                            process_state="fin1"
                            fin1_temp = True
                            print("arm1 full process ended- fin1_temp:{fin1_temp}, fin2_temp:{fin2_temp}")
                            work_done()
                            #if (fin1_temp and fin2_temp):
                            	#work_done()
                        #elif decoded_data == 'fin2':
                            #fin2_temp = True
                            #print("arm2 full process ended- fin1_temp:{fin1_temp}, fin2_temp:{fin2_temp}")
                            #if (fin1_temp and fin2_temp):
                            	#work_done()
                        capture_event.clear()  # Done capturing
                    else:
                        print(f"No more data from {client_address}")
                        break
            finally:
                #connection.close()
                print(f"Connection with {client_address} closed")
    except KeyboardInterrupt:
        print("Server shutting down...")
        server_socket.close()
    finally:
        server_socket.close()

if __name__ == "__main__":
    # Initialize SAM2 model
    initialize_model()
    
    # Start camera thread
    camera_thread1 = threading.Thread(target=camera_thread1, daemon=True)
    camera_thread2 = threading.Thread(target=camera_thread2, daemon=True)
    camera_thread1.start()
    camera_thread2.start()
    
    # Start TCP listener thread
    tcp_thread = threading.Thread(target=tcp_listener_thread, daemon=True)
    tcp_thread.start()
    udp_thread = threading.Thread(target=udp_listner_thread, daemon=True)
    udp_thread.start()
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        if is_camera_initialized1 and zed1:
            zed1.close()
            print("Camera 1 closed")
        if is_camera_initialized2 and zed2:
            zed2.close()
            print("Camera 2 closed")
