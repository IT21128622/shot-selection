import os
from flask import Flask, request, send_file, jsonify, make_response
from ultralytics import YOLO
import cv2
import numpy as np
import json 
from flask_cors import CORS

app = Flask(__name__)
CORS(app, expose_headers=["Bounce-Label", "Release-Point", "Highest-Point", "Release-Frame", "Highest-Frame", "Segment-Coordinates"])  

MODEL_DIR = './models' 
TEMP_DIR = './temp'  


os.makedirs(TEMP_DIR, exist_ok=True)

def load_model(model_path):
    """Load the YOLO model from the given path."""
    return YOLO(model_path)

def get_video_properties(video_path):
    """Get properties of the video such as FPS, width, and height."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, width, height

def create_video_writer(output_path, fourcc, fps, width, height):
    """Create a VideoWriter object."""
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def process_frame_ball_detection(model, frame):
    """Run inference on the frame to detect the ball and extract bounding boxes."""
    results = model(frame)
    bboxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()
            cls = box.cls.cpu().numpy()
            bboxes.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])
    return bboxes

def get_ball_center(bbox):
    """Calculate the center point of the bounding box."""
    x_center = int((bbox[0] + bbox[2]) / 2)
    y_center = int((bbox[1] + bbox[3]) / 2)
    return (x_center, y_center)

def interpolate_point(p1, p2, t):
    """Interpolate between two points p1 and p2 by factor t."""
    return (int(p1[0] + t * (p2[0] - p1[0])), int(p1[1] + t * (p2[1] - p1[1])))

def track_ball_path(model, video_path):
    """Track the ball path throughout the video and return the list of ball positions."""
    cap, fps, width, height = get_video_properties(video_path)
    if cap is None:
        print("Error: Could not retrieve video properties.")
        return []

    ball_positions = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect ball using YOLO model
        bboxes = process_frame_ball_detection(model, frame)
        balls = [bbox for bbox in bboxes if bbox[5] == 0]  # Assuming class 0 is ball

        if balls:
            x, y = get_ball_center(balls[0])
            ball_positions.append((x, y, frame_idx))  # Append (x, y, frame_idx)
        else:
            ball_positions.append(None)

        frame_idx += 1

    cap.release()
    return ball_positions

def detect_release_and_highest_points(ball_positions):
    """Detect the lowest (release) point and the highest point in the ball's path."""
    release_point = None
    highest_point = None
    frame_release = None
    frame_highest = None
    min_y = float('inf')
    max_y = float('-inf')

    # Find the release point (lowest point)
    for pos in ball_positions:
        if pos is not None:
            x, y, frame_idx = pos
            if y < min_y:  # Lowest point (release point)
                min_y = y
                release_point = pos  # Use the entire tuple to find the index later
                frame_release = frame_idx

    # Find the highest point after the release point
    if release_point is not None:
        release_index = ball_positions.index(release_point)
        for pos in ball_positions[release_index:]:
            if pos is not None:
                x, y, frame_idx = pos
                if y > max_y:  # Highest point after the release point
                    max_y = y
                    highest_point = pos
                    frame_highest = frame_idx

    print(f"Release point detected at: {release_point}")
    print(f"Highest point detected at: {highest_point}")
    return release_point, highest_point, frame_release, frame_highest

def calculate_color_segment_coordinates(bottom_middle1, bottom_middle2):
    """Calculate the start and end coordinates of each length segment with corresponding colors."""
    percentages = [0.1, 0.15, 0.25, 0.4, 0.6]
    colors = ["purple", "blue", "green", "orange", "red"]
    length_labels = ["yorker", "full", "good", "back of length", "short"]

    segment_coords = []

    start_point = bottom_middle2
    for i, percentage in enumerate(percentages):
        end_point = interpolate_point(bottom_middle2, bottom_middle1, percentage)
        segment_coords.append({
            "label": length_labels[i],
            "color": colors[i],
            "start": start_point,
            "end": end_point
        })
        start_point = end_point  # Update the starting point for the next segment

    return segment_coords

def find_bouncing_point_color(segment_coordinates, bounce_point):
    """Find which length segment aligns with the bouncing point based on its y-coordinate."""
    for segment in segment_coordinates:
        start, end = segment["start"], segment["end"]
        # Check if the bouncing point's y-coordinate falls within the segment
        if start[1] <= bounce_point[1] <= end[1] or end[1] <= bounce_point[1] <= start[1]:
            return  segment["label"]
    return None, None  # No color or length label found


def visualize_trajectory(video_path, output_path, ball_positions, release_point, highest_point):
    """Visualize the trajectory on the video and display the release point and highest point."""
    cap, fps, width, height = get_video_properties(video_path)
    if cap is None:
        print("Error: Could not open video for visualization.")
        return

    out = create_video_writer(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, width, height)

    frame_idx = 0
    previous_point = None  # Store the previous point to draw the line
    path_started = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Start drawing the trajectory line only after the release point
        if release_point is not None and frame_idx >= release_point[2]:
            path_started = True

        if path_started and frame_idx < len(ball_positions) and ball_positions[frame_idx] is not None:
            x, y = ball_positions[frame_idx][:2]
            if previous_point:
                # Draw line between previous point and current point
                cv2.line(frame, previous_point, (x, y), (0, 255, 0), 2)
            previous_point = (x, y)  # Update the previous point
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Ball positions in green

        # Draw the release point
        if release_point:
            cv2.circle(frame, (release_point[0], release_point[1]), 10, (255, 0, 0), -1)  # Blue dot for the release point
            cv2.putText(frame, "Release Point", (release_point[0] - 20, release_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw the highest point
        if highest_point and frame_idx >= highest_point[2]:
            cv2.circle(frame, (highest_point[0], highest_point[1]), 10, (0, 0, 255), -1)  # Red dot for the highest point

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_video_ball_detection(model_path, input_video_path, output_video_path):
    """Process video for ball detection."""
    model = load_model(model_path)
    ball_positions = track_ball_path(model, input_video_path)
    
    # Detect the release and highest points
    release_point, highest_point, frame_release, frame_highest = detect_release_and_highest_points(ball_positions)
    
    visualize_trajectory(input_video_path, output_video_path, ball_positions, release_point, highest_point)

    return release_point, highest_point, frame_release, frame_highest

def process_video_stump_detection(model_path, input_video_path, output_video_path, release_point, highest_point, frame_highest):
    """Process video for stump detection and fallback to nearest stump coordinates within ±5 frames."""
    model = load_model(model_path)
    cap, fps, width, height = get_video_properties(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = create_video_writer(output_video_path, fourcc, fps, width, height)

    segment_coordinates = None
    bounce_color = None

    nearest_stumps = None  # Variable to hold the nearest stump coordinates within ±5 frames

    # Track stump coordinates within the range of frame_highest ±5
    stump_coordinates_range = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Perform stump detection on each frame
        bboxes = process_frame_ball_detection(model, frame)
        stumps = [bbox for bbox in bboxes if bbox[5] == 0]  # Assuming class 0 is stump

       
        if frame_highest - 5 <= frame_number <= frame_highest + 5:
            if stumps:
                stump_coordinates_range[frame_number] = stumps

       
        if frame_number == frame_highest and not stumps:
            print(f"No stumps detected at frame {frame_number}. Searching nearby frames for nearest stumps.")
            nearest_stumps = get_nearest_stumps(stump_coordinates_range, frame_highest)

        # Use nearest stumps if no stumps detected at frame_highest
        if frame_number == frame_highest and not stumps and nearest_stumps:
            stumps = nearest_stumps
            print(f"Using nearest stumps detected at frame {frame_number}.")

        # Draw the line and annotations on each frame
        draw_annotations(frame, stumps)

        # If it is the frame with the highest point, calculate color segments
        if highest_point and frame_number == frame_highest:
            if len(stumps) >= 2:
                bottom_middle1 = (int((stumps[0][0] + stumps[0][2]) / 2), int(stumps[0][3]))
                bottom_middle2 = (int((stumps[1][0] + stumps[1][2]) / 2), int(stumps[1][3]))
                segment_coordinates = calculate_color_segment_coordinates(bottom_middle1, bottom_middle2)
                bounce_color = find_bouncing_point_color(segment_coordinates, highest_point)  # Determine the color
                print(f"Segment Coordinates: {segment_coordinates}")
                print(f"Bounce Color: {bounce_color}")

        # Mark the release point and highest point on each frame after detection
        if release_point:
            cv2.circle(frame, (release_point[0], release_point[1]), 10, (255, 0, 0), -1)  # Blue dot for the release point
        if highest_point:
            cv2.circle(frame, (highest_point[0], highest_point[1]), 10, (0, 0, 255), -1)  # Red dot for the highest point

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return segment_coordinates, bounce_color

def get_nearest_stumps(stump_coordinates_range, frame_highest):
    """Get the nearest stump coordinates within ±5 frames of frame_highest."""
    nearest_frame = None
    nearest_stumps = None

    # Sort the frames to find the closest one
    sorted_frames = sorted(stump_coordinates_range.keys())
    for frame in sorted_frames:
        if abs(frame - frame_highest) <= 5:
            nearest_frame = frame
            nearest_stumps = stump_coordinates_range[frame]
            break  # Get the nearest frame

    return nearest_stumps


def interpolate_point(start, end, fraction):
    """Interpolate a point at a certain fraction along a line segment between start and end points."""
    x = int(start[0] + (end[0] - start[0]) * fraction)
    y = int(start[1] + (end[1] - start[1]) * fraction)
    return (x, y)

def draw_annotations(frame, stumps):
    """Draw rectangles and lines between stumps on the frame, indicating specific lengths."""
    stumps = sorted(stumps, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

    if len(stumps) < 2:
        return

    cv2.rectangle(frame, (int(stumps[0][0]), int(stumps[0][1])), (int(stumps[0][2]), int(stumps[0][3])), (0, 255, 0), 4)
    cv2.rectangle(frame, (int(stumps[1][0]), int(stumps[1][1])), (int(stumps[1][2]), int(stumps[1][3])), (0, 0, 255), 2)

    # Calculate the bottom middle point of each stump
    bottom_middle1 = (int((stumps[0][0] + stumps[0][2]) / 2), int(stumps[0][3]))  # Top stump
    bottom_middle2 = (int((stumps[1][0] + stumps[1][2]) / 2), int(stumps[1][3]))  # Bottom stump

    # Define segment percentages based on cricket lengths
    percentages = [0.1, 0.15, 0.25, 0.4, 0.6]

    # Color segments: Purple, Blue, Green, Orange, Red
    colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 165, 255), (0, 0, 255)]
    length_labels = ["Yorker", "Full", "Good", "Back of Length", "Short"]

    # Draw the segments from bottom stump to top stump
    start_point = bottom_middle2  # Start from the bottom stump
    for i, percentage in enumerate(percentages):
        # Interpolate points from bottom to top using percentages
        end_point = interpolate_point(bottom_middle2, bottom_middle1, percentage)
        cv2.line(frame, start_point, end_point, colors[i], 2)

        # Annotate each segment with the length label
        label_point = (end_point[0] - 40, end_point[1] - 10)
        cv2.putText(frame, length_labels[i], label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1, cv2.LINE_AA)

        start_point = end_point  # Update the starting point for the next segment

def convert_to_h264_opencv(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Convert to H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


@app.route('/process-video', methods=['POST'])
def process_video_request():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    input_video_path = os.path.join(TEMP_DIR, 'input_video.mp4')
    ball_output_path = os.path.join(TEMP_DIR, 'output_video_ball.mp4')
    stump_output_path = os.path.join(TEMP_DIR, 'output_video_stump.mp4')
    final_output_path = os.path.join(TEMP_DIR, 'final_output_h264.mp4')

    video_file.save(input_video_path)

    # Step 1: Ball detection
    ball_model_path = os.path.join(MODEL_DIR, 'ball.pt')
    release_point, highest_point, frame_release, frame_highest = process_video_ball_detection(
        ball_model_path, input_video_path, ball_output_path
    )

    # Step 2: Stump detection on the output of ball detection
    stump_model_path = os.path.join(MODEL_DIR, 'stump.pt')
    segment_coordinates, bounce_label = process_video_stump_detection(
        stump_model_path, ball_output_path, stump_output_path, release_point, highest_point, frame_highest
    )

    # Step 3: Convert stump_output_path video to H.264 format
    convert_to_h264_opencv(stump_output_path, final_output_path)

    # Prepare the response with the final H.264 video
    response = make_response(
        send_file(final_output_path, mimetype='video/mp4', as_attachment=True, download_name='processed_video_h264.mp4')
    )

    # header metadata
    response.headers['Release-Point'] = f'{release_point[0]},{release_point[1]}'
    response.headers['Highest-Point'] = f'{highest_point[0]},{highest_point[1]}'
    response.headers['Release-Frame'] = str(frame_release)
    response.headers['Highest-Frame'] = str(frame_highest)

    # Serialize the segment coordinates as a JSON string in the headers
    if segment_coordinates:
        response.headers['Segment-Coordinates'] = json.dumps(segment_coordinates)

    # add lables to the header
    if bounce_label:
        response.headers['Bounce-Label'] = bounce_label  
        print(f'Bounce label: {bounce_label}')

    return response


if __name__ == '__main__':
    app.run(debug=True)
