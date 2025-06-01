import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
from PIL import Image
import time
# import threading # Not used for now to keep complexity down
# import queue # Not used for now

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Item Tracking App")

# --- Helper Functions ---
def is_centroid_in_zone(cx, cy, zone_coords):
    """Check if centroid is inside the zone"""
    x1, y1, x2, y2 = zone_coords
    return x1 <= cx <= x2 and y1 <= cy <= y2

def draw_text_with_background(img, text, position, font, font_scale, text_color, bg_color, thickness, padding):
    """Draw text with background rectangle"""
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def cv2_to_pil(cv2_image):
    """Convert OpenCV image (BGR) to PIL Image (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def display_opencv_image(cv2_image, caption="", use_container_width=True): # MODIFIED
    """Display OpenCV image in Streamlit"""
    pil_image = cv2_to_pil(cv2_image)
    st.image(pil_image, caption=caption, use_container_width=use_container_width) # MODIFIED

def draw_roi_preview(frame, roi_coords, show_transparent=False):
    """Draw ROI on frame - transparent when processing, semi-transparent when adjusting"""
    frame_with_roi = frame.copy()
    tx, ty, tw, th = roi_coords
    
    if not show_transparent:
        # Semi-transparent for ROI adjustment
        overlay = frame_with_roi.copy()
        cv2.rectangle(overlay, (tx, ty), (tx+tw, ty+th), (255, 0, 255), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame_with_roi, 1-alpha, 0, frame_with_roi)
        cv2.rectangle(frame_with_roi, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 2)
        cv2.putText(frame_with_roi, "Transfer Zone", (tx, ty-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    # When show_transparent=True, we draw a subtle border for processed video/live view
    elif show_transparent and tx >= 0 and ty >= 0 and tw > 0 and th > 0 : # only draw if valid coords
         cv2.rectangle(frame_with_roi, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 1) # Thin border
    
    return frame_with_roi

# --- Configuration ---
MODEL_PATH = 'best_bag_box.pt' # Ensure this model is available

# Global for model to load only once
@st.cache_resource
def load_yolo_model(model_path):
    """Load YOLO model with error handling"""
    try:
        model = YOLO(model_path)
        
        original_roboflow_name = "counting  industry - v1 2024-06-18 6-04am"
        new_name_for_roboflow_class = "bag" # Example, adjust if your model has different names
        remapped = False
        
        # Check if model.model.names exists and is a dictionary
        if hasattr(model, 'model') and hasattr(model.model, 'names') and isinstance(model.model.names, dict):
            for idx, name_in_model in model.model.names.items():
                if name_in_model == original_roboflow_name:
                    model.model.names[idx] = new_name_for_roboflow_class
                    remapped = True
                    break
            
            if remapped:
                st.success(f"Model class remapped: '{original_roboflow_name}' -> '{new_name_for_roboflow_class}'")
            # else: # This can be noisy if no remapping is needed every time
            #     st.info(f"No class remapping needed. Available classes: {list(model.model.names.values())}")
        else:
            st.warning("Model names attribute not found or not in expected format. Skipping remapping.")
            # st.info(f"Available classes (if any found): {list(model.model.names.values()) if hasattr(model, 'model') and hasattr(model.model, 'names') else 'Unknown'}")


        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        st.error(f"Please ensure '{os.path.basename(model_path)}' is in your app directory or provide the correct path.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Try to load model
model = load_yolo_model(MODEL_PATH)
if model is not None and hasattr(model, 'model') and hasattr(model.model, 'names'):
    AVAILABLE_CLASSES = list(model.model.names.values())
else:
    AVAILABLE_CLASSES = ['bag', 'box'] # Fallback
    if model is None:
      st.warning("Using fallback classes as model failed to load. Please check model file and path.")
    else: # Model loaded but names attribute missing
      st.warning("Model loaded, but class names could not be determined. Using fallback classes.")


# --- ROI Selection ---
def get_first_frame(uploaded_file_or_path):
    """Extract first frame from uploaded video or path"""
    cap = None
    temp_path = None
    try:
        if hasattr(uploaded_file_or_path, 'read'): # Uploaded file object
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_or_path.name)[1]) as tfile:
                tfile.write(uploaded_file_or_path.read())
                temp_path = tfile.name
            video_source = temp_path
        else: # Path string
            video_source = uploaded_file_or_path

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error(f"Could not open video source: {video_source}")
            return None
            
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        else:
            st.error("Could not read first frame from video")
            return None
            
    except Exception as e:
        st.error(f"Error extracting first frame: {str(e)}")
        return None
    finally:
        if cap:
            cap.release()
        if temp_path and os.path.exists(temp_path): # Ensure temp_path exists before unlinking
            try:
                os.unlink(temp_path)
            except Exception as e:
                st.warning(f"Could not delete temp file {temp_path}: {e}")


# --- Video Processing Functions ---
def process_single_frame(frame, model_instance, selected_class, transfer_zone_coords, conf_threshold, 
                        item_id_map, next_display_item_id, item_zone_tracking_info, 
                        loaded_count, unloaded_count, show_zone_border=True): # MODIFIED show_zone to show_zone_border
    """Process a single frame and return annotated frame with updated counts"""
    annotated_frame = frame.copy()
    
    tx, ty, tw, th = transfer_zone_coords
    TRANSFER_ZONE_COORDS_x1y1x2y2 = (tx, ty, tx + tw, ty + th)

    # Draw transfer zone border if requested (subtle)
    if show_zone_border and tx >= 0 and ty >= 0 and tw > 0 and th > 0 : # Only draw if valid coords
        cv2.rectangle(annotated_frame,
                      (TRANSFER_ZONE_COORDS_x1y1x2y2[0], TRANSFER_ZONE_COORDS_x1y1x2y2[1]),
                      (TRANSFER_ZONE_COORDS_x1y1x2y2[2], TRANSFER_ZONE_COORDS_x1y1x2y2[3]),
                      (255, 0, 255), 1)  # Thin border only
    
    # Get class index for filtering
    class_index = None # Default to no specific class filter
    if hasattr(model_instance, 'model') and hasattr(model_instance.model, 'names') and model_instance.model.names:
        class_names_list = list(model_instance.model.names.values())
        if selected_class in class_names_list:
            class_index = class_names_list.index(selected_class)
        else:
            st.warning(f"Selected class '{selected_class}' not found in model classes: {class_names_list}. Tracking all classes or first if available.")
            # Fallback strategy: track all if selected not found, or first if list is not empty
            # To track all when selected_class is not found, class_index remains None
            # If you'd rather default to the first class: class_index = 0 
    else:
        st.error("Model class names not accessible for filtering. Tracking all classes.")

    current_item_count_frame = 0
    
    try:
        track_args = {'conf': conf_threshold, 'persist': True, 'tracker': "bytetrack.yaml", 'verbose': False}
        if class_index is not None: # Only add 'classes' arg if a valid index was found
            track_args['classes'] = [class_index]
        
        results = model_instance.track(frame, **track_args)

    except Exception as e:
        st.error(f"Error during YOLO tracking: {e}")
        return annotated_frame, loaded_count[0], unloaded_count[0], next_display_item_id[0], current_item_count_frame
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes_data = results[0].boxes.xyxy.cpu().numpy()
        confs_data = results[0].boxes.conf.cpu().numpy()
        
        original_tracker_ids_data = None
        if results[0].boxes.id is not None:
            original_tracker_ids_data = results[0].boxes.id.cpu().numpy().astype(int)
        
        for i in range(len(boxes_data)):
            x1_item, y1_item, x2_item, y2_item = map(int, boxes_data[i])
            current_item_count_frame += 1
            
            item_cls_id_tensor = results[0].boxes.cls[i] if results[0].boxes.cls is not None else None
            item_cls_id = int(item_cls_id_tensor.cpu()) if item_cls_id_tensor is not None else None
            
            item_class_name = selected_class # Default to user-selected class
            if item_cls_id is not None and hasattr(model_instance, 'model') and hasattr(model_instance.model, 'names') and item_cls_id in model_instance.model.names:
                 item_class_name = model_instance.model.names[item_cls_id]
            
            label_base = f"{item_class_name} {confs_data[i]:.2f}"
            
            cx = (x1_item + x2_item) // 2
            cy = (y1_item + y2_item) // 2
            
            display_id_for_this_item = None 

            if original_tracker_ids_data is not None and i < len(original_tracker_ids_data): # Ensure index is valid
                original_tracker_id = original_tracker_ids_data[i]
                if original_tracker_id not in item_id_map:
                    item_id_map[original_tracker_id] = next_display_item_id[0]
                    next_display_item_id[0] += 1
                display_id_for_this_item = item_id_map[original_tracker_id]
                label = f"ID:{display_id_for_this_item} {label_base}"
                
                currently_in_zone = is_centroid_in_zone(cx, cy, TRANSFER_ZONE_COORDS_x1y1x2y2)
                if display_id_for_this_item not in item_zone_tracking_info:
                    item_zone_tracking_info[display_id_for_this_item] = {
                        "was_in_zone": currently_in_zone, 
                        "has_been_counted_load": False, 
                        "has_been_counted_unload": False
                    }
                
                info = item_zone_tracking_info[display_id_for_this_item]
                was_in_zone = info["was_in_zone"]
                
                if not was_in_zone and currently_in_zone and not info["has_been_counted_load"]:
                    loaded_count[0] += 1
                    info["has_been_counted_load"] = True
                    info["has_been_counted_unload"] = False 
                elif was_in_zone and not currently_in_zone and not info["has_been_counted_unload"]:
                    unloaded_count[0] += 1
                    info["has_been_counted_unload"] = True
                    info["has_been_counted_load"] = False 
                
                info["was_in_zone"] = currently_in_zone
            else: 
                label = label_base

            item_color = (0, 255, 0) if item_class_name.lower() == 'box' else (255, 165, 0) 
            cv2.rectangle(annotated_frame, (x1_item, y1_item), (x2_item, y2_item), item_color, 2)
            cv2.putText(annotated_frame, label, (x1_item, y1_item - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, item_color, 2)
            if display_id_for_this_item is not None : 
                cv2.circle(annotated_frame, (cx, cy), 5, item_color, -1) 
    
    y_offset = 30
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_thickness = 2
    bg_color = (50, 50, 50)
    padding = 5
    
    draw_text_with_background(annotated_frame, f"Loaded: {loaded_count[0]}", 
                            (10, y_offset), font_face, font_scale, 
                            (0, 255, 0), bg_color, text_thickness, padding)
    y_offset += 40
    draw_text_with_background(annotated_frame, f"Unloaded: {unloaded_count[0]}", 
                            (10, y_offset), font_face, font_scale, 
                            (0, 0, 255), bg_color, text_thickness, padding)
    y_offset += 40
    draw_text_with_background(annotated_frame, f"Items in frame: {current_item_count_frame}", 
                            (10, y_offset), font_face, 0.7, 
                            (0, 255, 255), bg_color, text_thickness, padding)
    
    return annotated_frame, loaded_count[0], unloaded_count[0], next_display_item_id[0], current_item_count_frame

def process_video_streamlit(video_bytes, selected_class, transfer_zone_rect, conf_threshold): 
    """Process video, yields progress, saves video to temp path for download."""
    if model is None:
        st.error("Model not loaded. Cannot process video.")
        yield None, None, None, None, None, None 
        return
    
    st.session_state.processed_video_path = None 

    temp_input_path = None
    output_video_path = None
    writer = None
    cap = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(video_bytes)
            temp_input_path = tfile.name
        
        item_id_map = {}
        next_display_item_id = [1] 
        item_zone_tracking_info = {}
        loaded_count = [0]
        unloaded_count = [0]
        
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            st.error(f"Could not open video file for processing: {temp_input_path}")
            yield None, None, None, None, None, None
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = max(fps, 1) # Ensure FPS is at least 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_file_obj:
            output_video_path = output_file_obj.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            st.error(f"Failed to open video writer for {output_video_path}")
            yield None, None, None, None, None, output_video_path 
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_count += 1
            
            annotated_frame, current_loaded, current_unloaded, current_next_id, items_in_frame = process_single_frame(
                frame, model, selected_class, transfer_zone_rect, conf_threshold,
                item_id_map, next_display_item_id, item_zone_tracking_info,
                loaded_count, unloaded_count, show_zone_border=True 
            )
            
            if writer is not None:
                writer.write(annotated_frame)
            
            total_unique_items = current_next_id - 1 if current_next_id > 1 else 0
            progress = frame_count / total_frames if total_frames > 0 else 0
            yield annotated_frame, current_loaded, current_unloaded, total_unique_items, progress, None 

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        current_loaded_val = loaded_count[0] if 'loaded_count' in locals() else None
        current_unloaded_val = unloaded_count[0] if 'unloaded_count' in locals() else None
        final_unique_val = (next_display_item_id[0] - 1) if 'next_display_item_id' in locals() and next_display_item_id[0] > 1 else 0
        yield None, current_loaded_val, current_unloaded_val, final_unique_val, None, output_video_path 
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
            except Exception as e_unlink:
                st.warning(f"Could not delete temp input file {temp_input_path}: {e_unlink}")
        
        if output_video_path and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            st.session_state.processed_video_path = output_video_path
        else: 
            st.session_state.processed_video_path = None
            if output_video_path and os.path.exists(output_video_path): 
                try:
                    os.unlink(output_video_path)
                except Exception as e_unlink_out:
                    st.warning(f"Could not delete temp output file {output_video_path}: {e_unlink_out}")

    final_loaded = loaded_count[0]
    final_unloaded = unloaded_count[0]
    final_unique = next_display_item_id[0] - 1 if next_display_item_id[0] > 1 else 0
    yield None, final_loaded, final_unloaded, final_unique, 1.0, st.session_state.processed_video_path


def live_stream_processing_loop(selected_class, transfer_zone_rect, conf_threshold, camera_index):
    """Processes live camera stream frames. Call within a Streamlit managed loop."""
    if model is None:
        st.error("Model not loaded. Cannot process live stream.")
        st.session_state.live_stream_active = False 
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"Could not open camera {camera_index}. Please check connection and index.")
        st.session_state.live_stream_active = False 
        return

    item_id_map = {}
    next_display_item_id = [1]
    item_zone_tracking_info = {}
    loaded_count = [0]
    unloaded_count = [0]

    stream_placeholder = st.empty()
    live_metrics_col1, live_metrics_col2, live_metrics_col3 = st.columns(3)
    unique_metric_live_ph = live_metrics_col1.empty() # Swapped for clarity
    loaded_metric_live_ph = live_metrics_col2.empty()
    unloaded_metric_live_ph = live_metrics_col3.empty()
    
    try:
        while st.session_state.get('live_stream_active', False): 
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("Failed to read frame from camera. Stream may have ended or camera disconnected.")
                st.session_state.live_stream_active = False 
                break
            
            annotated_frame, current_loaded, current_unloaded, current_next_id, _ = process_single_frame(
                frame, model, selected_class, transfer_zone_rect, conf_threshold,
                item_id_map, next_display_item_id, item_zone_tracking_info,
                loaded_count, unloaded_count, show_zone_border=True 
            )
            
            stream_placeholder.image(cv2_to_pil(annotated_frame), channels="RGB", use_container_width=True) 
            
            total_unique = next_display_item_id[0] - 1 if next_display_item_id[0] > 1 else 0
            unique_metric_live_ph.metric("Unique Items", total_unique)
            loaded_metric_live_ph.metric("Loaded", current_loaded)
            unloaded_metric_live_ph.metric("Unloaded", current_unloaded)
            
    except Exception as e:
        st.error(f"Error during live stream processing: {str(e)}")
    finally:
        if cap.isOpened(): # Check if cap was successfully opened before trying to release
            cap.release()
        stream_placeholder.empty() 
        unique_metric_live_ph.empty()
        loaded_metric_live_ph.empty()
        unloaded_metric_live_ph.empty()
        if st.session_state.get('live_stream_active', False): 
             st.info("Live stream stopped.")
        st.session_state.live_stream_active = False 

# --- Streamlit UI ---

try:
    st.image("logo.png", width=150) 
except Exception: 
    st.markdown("### 📦 Enhanced Item Tracking App") 

st.title("📦/🛍️ Enhanced Item Tracking & Counting App")

if 'roi_coords_manual' not in st.session_state:
    st.session_state.roi_coords_manual = {"x": 100, "y": 100, "w": 300, "h": 200}
if 'first_frame_roi' not in st.session_state:
    st.session_state.first_frame_roi = None
if 'selected_class' not in st.session_state:
    st.session_state.selected_class = AVAILABLE_CLASSES[0] if AVAILABLE_CLASSES else 'bag'
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "Upload Video"
if 'processed_video_path' not in st.session_state: 
    st.session_state.processed_video_path = None

if 'live_stream_state' not in st.session_state:
    st.session_state.live_stream_state = "initial"  
if 'live_stream_captured_frame' not in st.session_state:
    st.session_state.live_stream_captured_frame = None
if 'live_stream_roi_coords' not in st.session_state:
    st.session_state.live_stream_roi_coords = (100, 100, 200, 150) 
if 'live_stream_active' not in st.session_state: 
    st.session_state.live_stream_active = False


st.sidebar.header("📹 Processing Mode")
processing_mode_options = ["Upload Video", "Live Stream/Webcam"]
current_mode_index = processing_mode_options.index(st.session_state.processing_mode) if st.session_state.processing_mode in processing_mode_options else 0
new_processing_mode = st.sidebar.radio(
    "Choose processing mode:",
    processing_mode_options,
    index=current_mode_index,
    key="processing_mode_radio"
)

if st.session_state.processing_mode != new_processing_mode: 
    st.session_state.processing_mode = new_processing_mode
    if new_processing_mode == "Upload Video":
        st.session_state.live_stream_state = "initial"
        st.session_state.live_stream_active = False
        st.session_state.live_stream_captured_frame = None
    else: 
        st.session_state.first_frame_roi = None
        st.session_state.processed_video_path = None
    st.rerun()


if st.session_state.processing_mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.first_frame_roi = None 
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.processed_video_path = None 
    else: 
        st.session_state.uploaded_file_name = None # If no file, clear name
        # Keep first_frame_roi and processed_video_path if user deselects then reselects same file from cache
        # To fully reset on deselection:
        # st.session_state.first_frame_roi = None
        # st.session_state.processed_video_path = None


    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ Settings")
        
        if model is not None:
            st.session_state.selected_class = st.selectbox(
                "Select item type to track:",
                options=AVAILABLE_CLASSES,
                index=AVAILABLE_CLASSES.index(st.session_state.selected_class) if st.session_state.selected_class in AVAILABLE_CLASSES else 0
            )
        else:
            st.error("Model not loaded. Please ensure 'best_bag_box.pt' is available or check path.")
            st.session_state.selected_class = st.selectbox(
                "Select item type to track:", options=AVAILABLE_CLASSES, disabled=True
            )
        
        conf_threshold_st = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.55, 0.05, key="conf_thresh_upload")
        
        st.markdown("---")
        st.subheader("🎯 Define Transfer Zone (ROI)")
        
        if uploaded_file is not None and st.session_state.first_frame_roi is None:
            with st.spinner("Extracting first frame for ROI setup..."):
                # Pass uploaded_file directly as it has a 'read' method
                st.session_state.first_frame_roi = get_first_frame(uploaded_file)
                if st.session_state.first_frame_roi is None:
                    st.error("Failed to get first frame. Cannot setup ROI.")
                else: 
                    st.rerun() # Rerun to update UI with the frame
        
        roi_method_options = ["Manual Input", "Percentage Based"]
        default_roi_method_index = 0
        if st.session_state.first_frame_roi is not None: 
             default_roi_method_index = st.session_state.get('roi_method_pref_idx',0)

        roi_method = st.radio("Select ROI Method:", 
                             roi_method_options, 
                             index=default_roi_method_index,
                             horizontal=False, key="roi_method_upload")
        
        if st.session_state.first_frame_roi is not None:
            st.session_state.roi_method_pref_idx = roi_method_options.index(roi_method)

        roi_x, roi_y, roi_w, roi_h = (st.session_state.roi_coords_manual["x"],
                                      st.session_state.roi_coords_manual["y"],
                                      st.session_state.roi_coords_manual["w"],
                                      st.session_state.roi_coords_manual["h"])

        if roi_method == "Manual Input":
            roi_x = st.number_input("ROI X (top-left)", value=st.session_state.roi_coords_manual["x"], min_value=0, key="man_x")
            roi_y = st.number_input("ROI Y (top-left)", value=st.session_state.roi_coords_manual["y"], min_value=0, key="man_y")
            roi_w = st.number_input("ROI Width", value=st.session_state.roi_coords_manual["w"], min_value=10, key="man_w")
            roi_h = st.number_input("ROI Height", value=st.session_state.roi_coords_manual["h"], min_value=10, key="man_h")
            
        elif roi_method == "Percentage Based":
            if st.session_state.first_frame_roi is not None:
                frame_h_disp, frame_w_disp = st.session_state.first_frame_roi.shape[:2]
                st.info(f"Video dimensions for ROI setup: {frame_w_disp} x {frame_h_disp}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    x_percent = st.slider("X Position (%)", 0, 90, 20, key="perc_x")
                    w_percent = st.slider("Width (%)", 10, min(100-x_percent, 90), 40, key="perc_w") 
                with col_b:
                    y_percent = st.slider("Y Position (%)", 0, 90, 30, key="perc_y")
                    h_percent = st.slider("Height (%)", 10, min(100-y_percent, 90), 30, key="perc_h") 
                
                roi_x = int(frame_w_disp * x_percent / 100)
                roi_y = int(frame_h_disp * y_percent / 100)
                roi_w = int(frame_w_disp * w_percent / 100)
                roi_h = int(frame_h_disp * h_percent / 100)
            else:
                st.warning("Upload video first to use percentage-based ROI. Using default manual values.")
        
        st.session_state.roi_coords_manual = {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h}
        
        if st.session_state.first_frame_roi is not None:
            frame_with_roi_preview = draw_roi_preview(
                st.session_state.first_frame_roi, 
                (roi_x, roi_y, roi_w, roi_h), 
                show_transparent=False 
            )
            display_opencv_image(frame_with_roi_preview, "ROI Preview on First Frame", use_container_width=True)
        elif uploaded_file: 
            st.caption("First frame being extracted for ROI preview...")
        else: 
            st.caption("Upload a video to define ROI and see preview.")
    
    with col2:
        st.subheader("📊 Results")
        
        video_output_placeholder = st.empty()
        
        st.markdown("### 📈 Real-time Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        loaded_metric = metrics_col1.empty()
        unloaded_metric = metrics_col2.empty()
        unique_metric = metrics_col3.empty()
        
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Process video button logic
        process_button_disabled = not (uploaded_file is not None and model is not None)
        if st.button(f"🚀 Process Video for {st.session_state.selected_class}s", use_container_width=True, key="process_video_btn", disabled=process_button_disabled):
            st.info(f"Processing video for '{st.session_state.selected_class}'... ROI will be shown with a border.")
            st.session_state.processed_video_path = None 
            
            video_bytes = uploaded_file.getvalue() 
            transfer_zone_from_ui = (
                st.session_state.roi_coords_manual["x"],
                st.session_state.roi_coords_manual["y"],
                st.session_state.roi_coords_manual["w"],
                st.session_state.roi_coords_manual["h"]
            )
            
            progress_bar = progress_placeholder.progress(0)
            status_text_area = status_placeholder.empty() 
            status_text_area.text("Initializing processing...")
            
            final_results_from_processing = None
            for result in process_video_streamlit(video_bytes, st.session_state.selected_class, 
                                                transfer_zone_from_ui, conf_threshold_st):
                
                annotated_frame, loaded, unloaded, unique, progress, final_path_from_gen = result 
                final_results_from_processing = result # Keep track of the last yield

                if annotated_frame is not None:
                    video_output_placeholder.image(cv2_to_pil(annotated_frame), 
                                                 caption="Live Processing Preview", use_container_width=True) 
                
                if loaded is not None: 
                    loaded_metric.metric("Total Loaded", loaded)
                    unloaded_metric.metric("Total Unloaded", unloaded) 
                    unique_metric.metric("Unique Items", unique)
                    
                if progress is not None:
                    progress_bar.progress(progress)
                    status_text_area.text(f"Processing... {progress*100:.1f}% complete")
            
            status_placeholder.success("✅ Video processing completed!")
            progress_placeholder.empty() 
            video_output_placeholder.empty() 
            
            # Display thumbnail if processing was successful and path is set
            if final_results_from_processing and final_results_from_processing[5] and os.path.exists(final_results_from_processing[5]):
                 st.session_state.processed_video_path = final_results_from_processing[5] # Ensure it's set from the last yield
                 try:
                    cap_thumb = cv2.VideoCapture(st.session_state.processed_video_path)
                    if cap_thumb.isOpened():
                        ret_thumb, frame_thumb = cap_thumb.read()
                        if ret_thumb:
                            thumb_with_roi = draw_roi_preview(frame_thumb, transfer_zone_from_ui, show_transparent=True)
                            video_output_placeholder.image(cv2_to_pil(thumb_with_roi), caption="Processed Video (first frame with ROI)", use_container_width=True)
                        cap_thumb.release()
                 except Exception as e_thumb:
                    st.warning(f"Could not generate thumbnail: {e_thumb}")
            st.rerun() # Rerun to make download button visible immediately based on session state


        # Download button - visibility controlled by st.session_state.processed_video_path
        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
            try:
                with open(st.session_state.processed_video_path, "rb") as fp_dl:
                    file_name_for_download = "processed_video.mp4"
                    if st.session_state.uploaded_file_name:
                        base, ext = os.path.splitext(st.session_state.uploaded_file_name)
                        file_name_for_download = f"processed_{base}{ext if ext else '.mp4'}"

                    st.download_button(
                        label="💾 Download Processed Video",
                        data=fp_dl,
                        file_name=file_name_for_download,
                        mime="video/mp4",
                        key="download_processed_video_btn"
                    )
            except FileNotFoundError:
                st.error("Processed video file not found for download. Please reprocess.")
                st.session_state.processed_video_path = None 
            except Exception as e_dl_prep:
                st.error(f"Error preparing video for download: {e_dl_prep}")
                st.session_state.processed_video_path = None

        elif not uploaded_file: 
            st.info("📁 Upload a video file to begin processing...")
        elif not model:
             st.error("Cannot process video: Model not loaded.")


else:  # Live Stream Mode 
    st.subheader("📹 Live Stream Processing")
    col1_live, col2_live = st.columns([1, 1])

    with col1_live:
        st.subheader("⚙️ Live Stream Settings")
        if model is None:
            st.error("Model not loaded. Live stream disabled. Please check model file.")
        else:
            if st.session_state.live_stream_state == "initial":
                st.info("Setup camera and ROI before starting the stream.")
                camera_index = st.number_input("Camera Index", value=st.session_state.get('live_camera_idx',0), min_value=0, max_value=10, key="live_cam_idx_initial")
                st.session_state.live_camera_idx = camera_index 

                if st.button("📸 Capture Frame for ROI Setup", key="capture_frame_btn", use_container_width=True):
                    with st.spinner(f"Accessing camera {camera_index}..."):
                        cap_roi = cv2.VideoCapture(camera_index)
                        if cap_roi.isOpened():
                            ret_roi, frame_roi = cap_roi.read()
                            cap_roi.release()
                            if ret_roi and frame_roi is not None:
                                st.session_state.live_stream_captured_frame = frame_roi
                                st.session_state.live_stream_state = "roi_setup"
                                st.rerun()
                            else:
                                st.error(f"Failed to capture frame from camera {camera_index}.")
                        else:
                            st.error(f"Could not open camera {camera_index}. Check connection/permissions.")
            
            elif st.session_state.live_stream_state == "roi_setup":
                st.info("Adjust ROI for the live stream on the captured frame below.")
                if st.session_state.live_stream_captured_frame is not None:
                    frame_for_roi = st.session_state.live_stream_captured_frame
                    h_f, w_f = frame_for_roi.shape[:2]
                    
                    current_roi = st.session_state.live_stream_roi_coords
                    # Ensure ROI inputs are valid relative to frame dimensions
                    roi_x_live = st.slider("ROI X", 0, max(0, w_f - 50), current_roi[0], key="live_roi_x_slider", help=f"Frame width: {w_f}")
                    roi_y_live = st.slider("ROI Y", 0, max(0, h_f - 50), current_roi[1], key="live_roi_y_slider", help=f"Frame height: {h_f}")
                    roi_w_live = st.slider("ROI Width", 50, max(50, w_f - roi_x_live), current_roi[2], key="live_roi_w_slider")
                    roi_h_live = st.slider("ROI Height", 50, max(50, h_f - roi_y_live), current_roi[3], key="live_roi_h_slider")
                    st.session_state.live_stream_roi_coords = (roi_x_live, roi_y_live, roi_w_live, roi_h_live)

                    selected_class_live_idx = 0
                    if st.session_state.get('live_selected_class') in AVAILABLE_CLASSES:
                        selected_class_live_idx = AVAILABLE_CLASSES.index(st.session_state.get('live_selected_class'))
                    
                    selected_class_live = st.selectbox(
                        "Select item type for live tracking:", options=AVAILABLE_CLASSES,
                        index=selected_class_live_idx,
                        key="live_class_select_active"
                    )
                    st.session_state.live_selected_class = selected_class_live
                    
                    conf_threshold_live = st.slider("Detection Confidence", 0.1, 1.0, st.session_state.get('live_conf_thresh', 0.55), 0.05, key="live_conf_active")
                    st.session_state.live_conf_thresh = conf_threshold_live

                    if st.button("🚀 Start Live Detection", key="start_detection_btn", use_container_width=True):
                        st.session_state.live_stream_active = True 
                        st.session_state.live_stream_state = "streaming"
                        st.rerun()
                    if st.button("🔄 Recapture Frame", key="recapture_btn", use_container_width=True):
                        st.session_state.live_stream_captured_frame = None
                        st.session_state.live_stream_state = "initial"
                        st.rerun()
                else: 
                    st.error("No captured frame available for ROI setup. Please recapture.")
                    st.session_state.live_stream_state = "initial" 
                    st.rerun()

            elif st.session_state.live_stream_state == "streaming":
                st.info("Live stream is active. ROI is fixed.")
                st.markdown(f"**Tracking:** `{st.session_state.get('live_selected_class', 'N/A')}`")
                st.markdown(f"**Confidence:** `{st.session_state.get('live_conf_thresh', 'N/A')}`")
                roi_disp = st.session_state.get('live_stream_roi_coords')
                st.markdown(f"**ROI (x,y,w,h):** `{roi_disp}`")

                if st.button("⏹️ Stop Live Stream", key="stop_live_btn_active", use_container_width=True):
                    st.session_state.live_stream_active = False 
                    st.session_state.live_stream_state = "initial" 
                    time.sleep(0.1) # Short delay for loop to catch flag (may not be strictly necessary with rerun)
                    st.rerun()
    
    with col2_live:
        st.subheader("📺 Live Output")
        if st.session_state.live_stream_state == "roi_setup" and st.session_state.live_stream_captured_frame is not None:
            frame_roi_preview_live = draw_roi_preview(
                st.session_state.live_stream_captured_frame,
                st.session_state.live_stream_roi_coords,
                show_transparent=False 
            )
            display_opencv_image(frame_roi_preview_live, "ROI Preview for Live Stream", use_container_width=True)
        
        elif st.session_state.live_stream_state == "streaming" and st.session_state.live_stream_active:
            if model:
                live_stream_processing_loop(
                    st.session_state.live_selected_class, 
                    st.session_state.live_stream_roi_coords, 
                    st.session_state.live_conf_thresh, 
                    st.session_state.live_camera_idx
                )
            else:
                st.error("Model not available. Cannot start stream.")
                st.session_state.live_stream_active = False
                st.session_state.live_stream_state = "initial"

        elif st.session_state.live_stream_state == "initial":
            st.info("Camera feed and metrics will appear here once stream starts.")
        else: 
             st.info("Configure settings and start stream to see output.")


st.markdown("---")
st.caption("Enhanced App by ElevateTrust.AI Solutions")

st.sidebar.markdown("---")
st.sidebar.header("🔧 Additional Utilities") 

if st.sidebar.button("📷 Camera Test"):
    st.sidebar.info("Testing camera connection...")
    test_cap = None
    try:
        cam_idx_to_test = st.session_state.get('live_camera_idx', 0)
        test_cap = cv2.VideoCapture(cam_idx_to_test) 
        if test_cap.isOpened():
            st.sidebar.success(f"✅ Camera (index {cam_idx_to_test}) connected successfully!")
        else:
            st.sidebar.error(f"❌ Cannot connect to camera (index {cam_idx_to_test}).")
    except Exception as e_cam_test:
        st.sidebar.error(f"❌ Camera test failed: {e_cam_test}")
    finally:
        if test_cap and test_cap.isOpened():
            test_cap.release()

with st.sidebar.expander("❓ Help & Tips"):
    st.markdown("""
    **ROI Adjustment Tips (Upload Video & Live Stream):**
    - For 'Percentage Based' ROI, ensure a frame is loaded/captured first.
    - The transfer zone is semi-transparent during setup (purple box).
    - During processing (video or live), the zone is marked by a thin border.
    
    **Live Stream Tips:**
    - First, capture a frame from your camera to set up the ROI.
    - Ensure good lighting for better detection.
    - Test camera connection using the 'Camera Test' utility.
    - Adjust confidence threshold (0.5-0.7 often a good start).
    
    **General:**
    - If the model seems to miss items or detects wrong items, try adjusting the 'Detection Confidence Threshold'.
    - Ensure the `best_bag_box.pt` model file is in the same directory as the app script or provide the correct path.
    """)

if st.sidebar.checkbox("🐛 Debug Mode"):
    st.sidebar.markdown("### Debug Information")
    debug_info = {
        "Model Loaded": model is not None,
        "Available Classes": AVAILABLE_CLASSES,
        "Selected Class (Session - Upload)": st.session_state.get('selected_class'),
        "Processing Mode": st.session_state.processing_mode,
        "ROI Coords (Manual/Upload)": st.session_state.roi_coords_manual,
        "Processed Video Path (Session)": st.session_state.get('processed_video_path'),
        "---Live Stream Debug---": "---",
        "Live Stream State": st.session_state.get('live_stream_state'),
        "Live Stream Active Flag": st.session_state.get('live_stream_active'),
        "Live Selected Class": st.session_state.get('live_selected_class'),
        "Live Confidence": st.session_state.get('live_conf_thresh'),
        "Live Camera Index": st.session_state.get('live_camera_idx'),
        "Live ROI Coords": st.session_state.get('live_stream_roi_coords'),
        "Live Captured Frame Shape": st.session_state.live_stream_captured_frame.shape if st.session_state.live_stream_captured_frame is not None else "None",
    }
    
    simple_session_keys = {}
    for k, v in st.session_state.items():
        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            simple_session_keys[k] = v
        elif isinstance(v, np.ndarray): # Specifically for numpy arrays like frames
            simple_session_keys[k] = f"Numpy array, Shape: {v.shape}, Dtype: {v.dtype}"
        else:
            simple_session_keys[k] = f"Object of type: {type(v)}"
            
    debug_info["Full Session State (Simplified)"] = simple_session_keys
    st.sidebar.json(debug_info, expanded=False)