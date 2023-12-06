import cv2
import numpy as np

# Function to adjust brightness of a frame
def increase_brightness(frame, value=30):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return frame

# Function to detect if a video is taken during nighttime and adjust brightness if needed
def adjust_brightness(video_path, frame_sample_size=300):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    frame_count = 0
    total_brightness = 0
    dark_pixel_ratio = 0
    last_frame_brightness = None
    dark_frame_count = 0
    frame_changes = []

    while frame_count < frame_sample_size:
        success, frame = vid.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_frame)
        total_brightness += avg_brightness

        dark_pixels = np.sum(gray_frame < 30)  # Threshold for dark pixels
        total_pixels = gray_frame.size
        dark_pixel_ratio += dark_pixels / total_pixels

        if last_frame_brightness is not None:
            brightness_change = avg_brightness - last_frame_brightness
            frame_changes.append(brightness_change)

            if brightness_change < -10:
                dark_frame_count += 1

        last_frame_brightness = avg_brightness
        frame_count += 1

    vid.release()

    overall_avg_brightness = total_brightness / frame_count
    overall_dark_pixel_ratio = dark_pixel_ratio / frame_count

    brightness_threshold = 40
    dark_pixel_ratio_threshold = 0.4
    temporal_dark_frame_threshold = 50

    is_night = (
        overall_avg_brightness < brightness_threshold and
        overall_dark_pixel_ratio > dark_pixel_ratio_threshold and
        dark_frame_count > temporal_dark_frame_threshold
    )

    return is_night

# Function to overlay the watermark on a frame
def overlay_watermark(frame, watermark, position=(0, 0)):
    watermark_h, watermark_w, watermark_channels = watermark.shape
    overlay = np.copy(frame[position[1]:position[1]+watermark_h, position[0]:position[0]+watermark_w])
    watermark_mask = watermark[:,:,3] / 255.0
    overlay_mask = 1.0 - watermark_mask

    for c in range(0, 3):
        overlay[:,:,c] = (overlay_mask * overlay[:,:,c] +
                          watermark_mask * watermark[:,:,c])
    frame[position[1]:position[1]+watermark_h, position[0]:position[0]+watermark_w] = overlay
    return frame

# Function to blur faces in a video and overlay another video with a border
def blur_faces_and_overlay(input_video_path, output_video_path, cascade_classifier_path, overlay_video_path, endscreen_path, adjust_brightness=False):
    face_cascade = cv2.CascadeClassifier(cascade_classifier_path)
    cap_main = cv2.VideoCapture(input_video_path)
    cap_overlay = cv2.VideoCapture(overlay_video_path)

    frame_width = int(cap_main.get(3))
    frame_height = int(cap_main.get(4))
    fps = int(cap_main.get(5))

    # Load watermark images and make sure they have an alpha channel
    watermark1 = cv2.imread(watermark_paths[0], cv2.IMREAD_UNCHANGED)
    watermark2 = cv2.imread(watermark_paths[1], cv2.IMREAD_UNCHANGED)

    # Calculate the interval for switching watermarks (every 5 seconds)
    switch_interval = fps * 5  # 5 seconds times frames per second

    # Define the size for the overlay video (for example, 1/4 of the main video size)
    overlay_width = frame_width // 4
    overlay_height = frame_height // 4
    border_thickness = 10  # Thickness of the black border

    # Offset from the corner
    offset_x = 10  # Horizontal offset from the top-left corner
    offset_y = 10  # Vertical offset from the top-left corner

    overlay_width_with_border = overlay_width + 2 * border_thickness
    overlay_height_with_border = overlay_height + 2 * border_thickness

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    last_detected_faces = []

    while True:
        ret_main, frame_main = cap_main.read()
        ret_overlay, frame_overlay = cap_overlay.read()

        if not ret_main:
            break

        if adjust_brightness:
            frame_main = increase_brightness(frame_main)

        if ret_overlay:
            frame_overlay_resized = cv2.resize(frame_overlay, (overlay_width, overlay_height))
            border_background = np.zeros((overlay_height_with_border, overlay_width_with_border, 3), dtype=np.uint8)
            border_background[border_thickness:border_thickness + overlay_height, 
                              border_thickness:border_thickness + overlay_width] = frame_overlay_resized
            
            # Position the overlay near the corner, not exactly at the corner
            start_y = offset_y
            start_x = offset_x
            end_y = start_y + overlay_height_with_border
            end_x = start_x + overlay_width_with_border
            
            frame_main[start_y:end_y, start_x:end_x] = border_background
        else:
            cap_overlay.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Blur only the main video faces
        gray_main = cv2.cvtColor(frame_main, cv2.COLOR_BGR2GRAY)
        detected_faces_main = face_cascade.detectMultiScale(gray_main, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        if len(detected_faces_main) == 0:
            detected_faces_main = last_detected_faces
        else:
            last_detected_faces = detected_faces_main

        for (x, y, w, h) in detected_faces_main:
            # Only blur the face if it's not in the area where the overlay video is positioned
            if y > end_y or x > end_x:
                face = frame_main[y:y+h, x:x+w]
                face = cv2.GaussianBlur(face, (99, 99), 30)
                frame_main[y:y+h, x:x+w] = face

        out.write(frame_main)
        frame_count += 1

    # After processing the main video, add the endscreen video to the end
    cap_endscreen = cv2.VideoCapture(endscreen_path)
    while True:
        ret_endscreen, frame_endscreen = cap_endscreen.read()
        if not ret_endscreen:
            break
        out.write(frame_endscreen)

    cap_main.release()
    cap_overlay.release()
    cap_endscreen.release()
    out.release()
    cv2.destroyAllWindows()

# Main function to process the videos
def process_videos(video_paths, cascade_classifier_path, overlay_video_path, endscreen_path):
    for video_path in video_paths:
        print(f"Processing {video_path}...")

        is_night = adjust_brightness(video_path)
        if is_night:
            print("Shot at night: True")
            brightness_adjust = True
        else:
            print("Shot at night: False")
            brightness_adjust = False

        output_video_path = f"processed_{video_path[:-4]}.avi"
        blur_faces_and_overlay(video_path, output_video_path, cascade_classifier_path, overlay_video_path, endscreen_path, adjust_brightness=brightness_adjust)

# List of video paths and cascade classifier path
video_paths = ["singapore.mp4", "alley.mp4", "traffic.mp4", "office.mp4"]
cascade_classifier_path = "face_detector.xml"
overlay_video_path = "talking.mp4"
endscreen_path = "endscreen.mp4"
watermark_paths = ["2.png", "3.png"]

# Process the videos
process_videos(video_paths, cascade_classifier_path, overlay_video_path, endscreen_path)
