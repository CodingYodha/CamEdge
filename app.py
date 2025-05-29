import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# --- All your helper functions from the script above ---
# (is_centroid_in_zone, draw_text_with_background, etc.)
# --- And the main processing function: detect_and_custom_track_video_streamlit ---
# This function will need to be adapted to work with Streamlit's state and UI.
# For example, instead of cv2.imshow, it might yield frames or save the video.

# --- CONFIGURATION (can be moved to session state or UI controls) ---
MODEL_PATH = 'best_bag_box.pt' # Ensure this path is accessible by Streamlit app

# Global for model to load only once
@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    # Remap class name
    original_roboflow_name = "counting  industry - v1 2024-06-18 6-04am"
    new_name_for_roboflow_class = "bag"
    remapped = False
    for idx, name_in_model in model.model.names.items():
        if name_in_model == original_roboflow_name:
            model.model.names[idx] = new_name_for_roboflow_class
            remapped = True
            break
    if remapped:
        st.success(f"Model class remapped: '{original_roboflow_name}' -> '{new_name_for_roboflow_class}'")
    else:
        st.warning(f"Class '{original_roboflow_name}' not found in model for remapping.")
    return model

model = load_yolo_model(MODEL_PATH)
AVAILABLE_CLASSES = list(model.model.names.values())


# --- ROI Selection ---
# For Streamlit, direct OpenCV mouse callback is not possible for ROI.
# Options:
# 1. Manual input: st.number_input for X, Y, W, H. Show preview on a static frame.
# 2. streamlit-drawable-canvas: More interactive but adds a dependency.

def get_first_frame(uploaded_file):
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        os.unlink(video_path) # Clean up temp file
        if ret:
            return frame
    return None

# --- Main Processing function adapted for Streamlit ---
# (This is a simplified adaptation, full video processing and display needs careful handling)
def process_video_streamlit(video_bytes, selected_class, transfer_zone_rect, conf_threshold, output_placeholder):
    # Similar to detect_and_custom_track_video but writes frames to Streamlit or saves output
    # For simplicity, let's assume it processes and returns final counts and maybe a path to processed video
    
    # Save uploaded video to a temporary file to pass to OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_bytes)
    video_path = tfile.name

    # Reset states (could use session_state for these if function is re-entrant)
    item_id_map_st = {}
    next_display_item_id_st = 1
    item_zone_tracking_info_st = {}
    loaded_count_st = 0
    unloaded_count_st = 0

    # --- Re-implement core logic of detect_and_custom_track_video here ---
    # Using item_id_map_st, loaded_count_st etc.
    # Instead of cv2.imshow, you would update st.image or st.empty()
    # For full video output, save it and then use st.video
    
    # Placeholder for the actual processing loop from your script:
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define a temporary path for the output video
    # This needs to be accessible by st.video if you want to display it.
    # For Streamlit Cloud, all paths are relative to the app's root.
    output_video_filename = "processed_output.mp4"
    
    # If running locally and want to save to a specific path, ensure it's writable.
    # For cloud deployment, use relative paths or temp files.
    # For this example, let's assume we're creating it in the current directory.
    temp_output_dir = tempfile.mkdtemp()
    output_video_path = os.path.join(temp_output_dir, output_video_filename)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    image_placeholder = st.empty() # To show live frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: total_frames = 1 # Avoid division by zero for short/invalid videos

    tx, ty, tw, th = transfer_zone_rect
    TRANSFER_ZONE_COORDS_x1y1x2y2 = (tx, ty, tx + tw, ty + th)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        annotated_frame = frame.copy()

        # ... (Copy drawing of ROI, processing logic from your script) ...
        # Make sure to use item_id_map_st, loaded_count_st, etc.
        # Draw the semi-transparent Transfer Zone
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, 
                      (TRANSFER_ZONE_COORDS_x1y1x2y2[0], TRANSFER_ZONE_COORDS_x1y1x2y2[1]),
                      (TRANSFER_ZONE_COORDS_x1y1x2y2[2], TRANSFER_ZONE_COORDS_x1y1x2y2[3]),
                      (255, 0, 255), -1)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
        cv2.rectangle(annotated_frame,
                      (TRANSFER_ZONE_COORDS_x1y1x2y2[0], TRANSFER_ZONE_COORDS_x1y1x2y2[1]),
                      (TRANSFER_ZONE_COORDS_x1y1x2y2[2], TRANSFER_ZONE_COORDS_x1y1x2y2[3]),
                      (255, 0, 255), 2) 
        cv2.putText(annotated_frame, "Transfer Zone",
                    (TRANSFER_ZONE_COORDS_x1y1x2y2[0], TRANSFER_ZONE_COORDS_x1y1x2y2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255, 255), 2)

        results_st = model.track(frame, conf=conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[AVAILABLE_CLASSES.index(selected_class)])
        
        current_item_count_frame = 0

        if results_st[0].boxes is not None:
            boxes_data = results_st[0].boxes.xyxy.cpu().numpy()
            confs_data = results_st[0].boxes.conf.cpu().numpy()
            # class_ids_data = results_st[0].boxes.cls.cpu().numpy().astype(int) # Not needed if filtering by class in model.track
            
            original_tracker_ids_data = None
            if results_st[0].boxes.id is not None:
                original_tracker_ids_data = results_st[0].boxes.id.cpu().numpy().astype(int)

            for i in range(len(boxes_data)):
                x1_item, y1_item, x2_item, y2_item = map(int, boxes_data[i])
                # class_name = model.names[class_ids_data[i]] # No, use selected_class
                # if class_name != selected_class: continue # Already filtered by model.track

                current_item_count_frame +=1
                label_st = f"{selected_class} {confs_data[i]:.2f}"
                
                cx = (x1_item + x2_item) // 2
                cy = (y1_item + y2_item) // 2

                if original_tracker_ids_data is not None:
                    original_tracker_id = original_tracker_ids_data[i]
                    if original_tracker_id not in item_id_map_st:
                        item_id_map_st[original_tracker_id] = next_display_item_id_st
                        next_display_item_id_st += 1
                    display_id_for_this_item = item_id_map_st[original_tracker_id]
                    label_st = f"ID:{display_id_for_this_item} {selected_class} {confs_data[i]:.2f}"

                    currently_in_zone = is_centroid_in_zone(cx, cy, TRANSFER_ZONE_COORDS_x1y1x2y2)
                    if display_id_for_this_item not in item_zone_tracking_info_st:
                        item_zone_tracking_info_st[display_id_for_this_item] = {"was_in_zone": currently_in_zone}
                    else:
                        was_in_zone = item_zone_tracking_info_st[display_id_for_this_item]["was_in_zone"]
                        if not was_in_zone and currently_in_zone: loaded_count_st += 1
                        elif was_in_zone and not currently_in_zone: unloaded_count_st += 1
                        item_zone_tracking_info_st[display_id_for_this_item]["was_in_zone"] = currently_in_zone
                
                item_color = (0, 255, 0) if selected_class == 'box' else (255, 165, 0)
                cv2.rectangle(annotated_frame, (x1_item, y1_item), (x2_item, y2_item), item_color, 2)
                cv2.putText(annotated_frame, label_st, (x1_item, y1_item - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, item_color, 2)
                cv2.circle(annotated_frame, (cx, cy), 5, item_color, -1)

        # Display counts on frame (using your draw_text_with_background)
        y_offset_st = 30
        font_face_st = cv2.FONT_HERSHEY_SIMPLEX; font_scale_st = 0.8; text_thickness_st = 2; bg_color_st = (50,50,50); padding_st = 5
        draw_text_with_background(annotated_frame, f"Loaded: {loaded_count_st}", (10, y_offset_st), font_face_st, font_scale_st, (0,255,0), bg_color_st, text_thickness_st, padding_st)
        y_offset_st += 40
        draw_text_with_background(annotated_frame, f"Unloaded: {unloaded_count_st}", (10, y_offset_st), font_face_st, font_scale_st, (0,0,255), bg_color_st, text_thickness_st, padding_st)
        y_offset_st += 40
        draw_text_with_background(annotated_frame, f"{selected_class} in frame: {current_item_count_frame}", (10, y_offset_st), font_face_st, 0.7, (0,255,255), bg_color_st, text_thickness_st, padding_st)


        if writer:
            writer.write(annotated_frame)
        
        # Update Streamlit UI
        image_placeholder.image(annotated_frame, channels="BGR")
        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")


    cap.release()
    if writer:
        writer.release()
    
    status_text.text("Video processing complete!")
    progress_bar.empty() # Clear progress bar

    total_unique_items = next_display_item_id_st - 1 if next_display_item_id_st > 1 else 0
    
    # Clean up temp input file
    os.unlink(video_path)

    return total_unique_items, loaded_count_st, unloaded_count_st, output_video_path


# --- Streamlit UI ---
st.set_page_config(layout="wide")

# Logo - place logo.png in the same directory as app.py
try:
    st.image("logo.png", width=150)
except Exception as e:
    st.warning(f"Could not load logo.png: {e}")

st.title(f"📦/🛍️ Item Tracking & Counting App")

# Initialize session state variables
if 'roi_coords_manual' not in st.session_state:
    st.session_state.roi_coords_manual = {"x": 100, "y": 100, "w": 300, "h": 200}
if 'first_frame_roi' not in st.session_state:
    st.session_state.first_frame_roi = None
if 'selected_class' not in st.session_state:
    st.session_state.selected_class = AVAILABLE_CLASSES[0] if AVAILABLE_CLASSES else 'box'


uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Settings")
    st.session_state.selected_class = st.selectbox(
        "Select item type to track:",
        options=AVAILABLE_CLASSES,
        index=AVAILABLE_CLASSES.index(st.session_state.selected_class) if st.session_state.selected_class in AVAILABLE_CLASSES else 0
    )
    conf_threshold_st = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.55, 0.05)

    st.markdown("---")
    st.subheader("🎯 Define Transfer Zone (ROI)")
    st.info("Adjust the ROI values below. The ROI will be shown on the first frame if a video is uploaded.")

    if uploaded_file is not None and st.session_state.first_frame_roi is None:
        # Get and store the first frame only once per upload
        st.session_state.first_frame_roi = get_first_frame(uploaded_file)

    roi_x = st.number_input("ROI X (top-left)", value=st.session_state.roi_coords_manual["x"], min_value=0)
    roi_y = st.number_input("ROI Y (top-left)", value=st.session_state.roi_coords_manual["y"], min_value=0)
    roi_w = st.number_input("ROI Width", value=st.session_state.roi_coords_manual["w"], min_value=10)
    roi_h = st.number_input("ROI Height", value=st.session_state.roi_coords_manual["h"], min_value=10)
    
    # Update session state for persistence if values change
    st.session_state.roi_coords_manual = {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h}
    
    if st.session_state.first_frame_roi is not None:
        frame_with_roi_preview = st.session_state.first_frame_roi.copy()
        tx, ty, tw, th = roi_x, roi_y, roi_w, roi_h
        # Draw semi-transparent ROI
        overlay_st = frame_with_roi_preview.copy()
        cv2.rectangle(overlay_st, (tx,ty), (tx+tw, ty+th), (255,0,255, 128), -1)
        alpha_st = 0.3
        cv2.addWeighted(overlay_st, alpha_st, frame_with_roi_preview, 1-alpha_st, 0, frame_with_roi_preview)
        cv2.rectangle(frame_with_roi_preview, (tx,ty), (tx+tw, ty+th), (255,0,255), 2) # Border
        st.image(frame_with_roi_preview, caption="ROI Preview on First Frame", channels="BGR", use_column_width=True)
    else:
        st.caption("Upload a video to see ROI preview on its first frame.")


with col2:
    st.subheader("📊 Results")
    output_video_placeholder = st.empty()
    
    final_counts_col1, final_counts_col2, final_counts_col3 = st.columns(3)
    unique_items_placeholder = final_counts_col1.empty()
    loaded_placeholder = final_counts_col2.empty()
    unloaded_placeholder = final_counts_col3.empty()


if uploaded_file is not None:
    if st.button(f"Process Video for {st.session_state.selected_class}s", use_container_width=True):
        output_video_placeholder.empty() # Clear previous video
        unique_items_placeholder.empty()
        loaded_placeholder.empty()
        unloaded_placeholder.empty()

        st.info(f"Processing video for '{st.session_state.selected_class}'... This may take a while.")
        
        video_bytes = uploaded_file.getvalue()
        transfer_zone_from_ui = (
            st.session_state.roi_coords_manual["x"],
            st.session_state.roi_coords_manual["y"],
            st.session_state.roi_coords_manual["w"],
            st.session_state.roi_coords_manual["h"]
        )

        total_unique, total_loaded, total_unloaded, processed_video_path = process_video_streamlit(
            video_bytes,
            st.session_state.selected_class,
            transfer_zone_from_ui,
            conf_threshold_st,
            output_video_placeholder # Pass placeholder for live updates
        )

        if processed_video_path and os.path.exists(processed_video_path):
            st.success("Video processed!")
            output_video_placeholder.video(processed_video_path)
            # Clean up the temp processed video file after displaying
            # os.unlink(processed_video_path) 
            # Be careful with unlinking if st.video needs the file to persist for a bit
        elif total_unique is not None : # Processing happened but no video path (e.g. error during save)
             st.success("Processing logic finished, but output video might not be available.")
        else:
            st.error("Video processing failed.")

        unique_items_placeholder.metric(f"Total Unique '{st.session_state.selected_class}s'", total_unique if total_unique is not None else "N/A")
        loaded_placeholder.metric("Total Loaded", total_loaded if total_loaded is not None else "N/A")
        unloaded_placeholder.metric("Total Unloaded", total_unloaded if total_unloaded is not None else "N/A")

else:
    st.info("Awaiting video upload...")

st.markdown("---")
st.caption("App by ElevateTrust.AI Solutions")