import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
from PIL import Image
import time
import threading
import queue

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

def display_opencv_image(cv2_image, caption="", use_column_width=True):
    """Display OpenCV image in Streamlit"""
    pil_image = cv2_to_pil(cv2_image)
    st.image(pil_image, caption=caption, use_column_width=use_column_width)

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
    # When show_transparent=True, we don't draw anything (completely transparent)
    
    return frame_with_roi

# --- Configuration ---
MODEL_PATH = 'best_bag_box.pt'

# Global for model to load only once
@st.cache_resource
def load_yolo_model(model_path):
    """Load YOLO model with error handling"""
    try:
        model = YOLO(model_path)
        
        # Remap class name if needed
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
            st.info(f"No class remapping needed. Available classes: {list(model.model.names.values())}")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure 'best_bag_box.pt' is in your app directory")
        return None

# Try to load model
model = load_yolo_model(MODEL_PATH)
if model is not None:
    AVAILABLE_CLASSES = list(model.model.names.values())
else:
    # Fallback classes if model fails to load
    AVAILABLE_CLASSES = ['bag', 'box']
    st.warning("Using fallback classes. Please upload your model file.")

# --- ROI Selection ---
def get_first_frame(uploaded_file):
    """Extract first frame from uploaded video"""
    if uploaded_file is not None:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
                tfile.write(uploaded_file.read())
                temp_path = tfile.name
            
            # Use VideoCapture with proper error handling
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("Could not open video file")
                return None
                
            ret, frame = cap.read()
            cap.release()
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
            if ret and frame is not None:
                return frame
            else:
                st.error("Could not read first frame from video")
                return None
                
        except Exception as e:
            st.error(f"Error extracting first frame: {str(e)}")
    return None

def interactive_roi_selector(frame):
    """Interactive ROI selection using Streamlit components"""
    st.subheader("🖱️ Interactive ROI Selection")
    st.info("Use the sliders below to adjust the ROI by dragging on the preview")
    
    h, w = frame.shape[:2]
    
    # Create sliders for ROI adjustment
    col_a, col_b = st.columns(2)
    with col_a:
        x_pos = st.slider("X Position", 0, w-50, w//4, key="interactive_x")
        width = st.slider("Width", 50, min(w-x_pos, w//2), w//3, key="interactive_w")
    with col_b:
        y_pos = st.slider("Y Position", 0, h-50, h//4, key="interactive_y") 
        height = st.slider("Height", 50, min(h-y_pos, h//2), h//3, key="interactive_h")
    
    # Show preview with current ROI
    roi_frame = draw_roi_preview(frame, (x_pos, y_pos, width, height), show_transparent=False)
    st.image(cv2_to_pil(roi_frame), caption="Drag ROI Preview - Adjust using sliders above")
    
    return x_pos, y_pos, width, height

# --- Video Processing Functions ---
def process_single_frame(frame, model, selected_class, transfer_zone_coords, conf_threshold, 
                        item_id_map, next_display_item_id, item_zone_tracking_info, 
                        loaded_count, unloaded_count, show_zone=True):
    """Process a single frame and return annotated frame with updated counts"""
    annotated_frame = frame.copy()
    
    # Draw transfer zone (completely transparent during processing if show_zone=False)
    if show_zone:
        # Only show zone during ROI adjustment, not during processing
        tx, ty, tw, th = transfer_zone_coords
        TRANSFER_ZONE_COORDS_x1y1x2y2 = (tx, ty, tx + tw, ty + th)
        
        # Completely transparent - just draw a subtle border
        cv2.rectangle(annotated_frame,
                      (TRANSFER_ZONE_COORDS_x1y1x2y2[0], TRANSFER_ZONE_COORDS_x1y1x2y2[1]),
                      (TRANSFER_ZONE_COORDS_x1y1x2y2[2], TRANSFER_ZONE_COORDS_x1y1x2y2[3]),
                      (255, 0, 255), 1)  # Thin border only
    
    tx, ty, tw, th = transfer_zone_coords
    TRANSFER_ZONE_COORDS_x1y1x2y2 = (tx, ty, tx + tw, ty + th)
    
    # Get class index for filtering
    try:
        class_index = AVAILABLE_CLASSES.index(selected_class)
    except ValueError:
        class_index = 0
    
    current_item_count_frame = 0
    
    # Run YOLO tracking
    try:
        results = model.track(frame, conf=conf_threshold, persist=True, 
                             tracker="bytetrack.yaml", verbose=False, 
                             classes=[class_index])
    except Exception as e:
        return annotated_frame, loaded_count, unloaded_count, next_display_item_id, current_item_count_frame
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes_data = results[0].boxes.xyxy.cpu().numpy()
        confs_data = results[0].boxes.conf.cpu().numpy()
        
        original_tracker_ids_data = None
        if results[0].boxes.id is not None:
            original_tracker_ids_data = results[0].boxes.id.cpu().numpy().astype(int)
        
        for i in range(len(boxes_data)):
            x1_item, y1_item, x2_item, y2_item = map(int, boxes_data[i])
            current_item_count_frame += 1
            label = f"{selected_class} {confs_data[i]:.2f}"
            
            cx = (x1_item + x2_item) // 2
            cy = (y1_item + y2_item) // 2
            
            if original_tracker_ids_data is not None:
                original_tracker_id = original_tracker_ids_data[i]
                if original_tracker_id not in item_id_map:
                    item_id_map[original_tracker_id] = next_display_item_id[0]
                    next_display_item_id[0] += 1
                display_id_for_this_item = item_id_map[original_tracker_id]
                label = f"ID:{display_id_for_this_item} {selected_class} {confs_data[i]:.2f}"
                
                currently_in_zone = is_centroid_in_zone(cx, cy, TRANSFER_ZONE_COORDS_x1y1x2y2)
                if display_id_for_this_item not in item_zone_tracking_info:
                    item_zone_tracking_info[display_id_for_this_item] = {"was_in_zone": currently_in_zone}
                else:
                    was_in_zone = item_zone_tracking_info[display_id_for_this_item]["was_in_zone"]
                    if not was_in_zone and currently_in_zone:
                        loaded_count[0] += 1
                    elif was_in_zone and not currently_in_zone:
                        unloaded_count[0] += 1
                    item_zone_tracking_info[display_id_for_this_item]["was_in_zone"] = currently_in_zone
            
            # Draw bounding box and label
            item_color = (0, 255, 0) if selected_class == 'box' else (255, 165, 0)
            cv2.rectangle(annotated_frame, (x1_item, y1_item), (x2_item, y2_item), item_color, 2)
            cv2.putText(annotated_frame, label, (x1_item, y1_item - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, item_color, 2)
            cv2.circle(annotated_frame, (cx, cy), 5, item_color, -1)
    
    # Display counts on frame
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
    draw_text_with_background(annotated_frame, f"{selected_class} in frame: {current_item_count_frame}", 
                            (10, y_offset), font_face, 0.7, 
                            (0, 255, 255), bg_color, text_thickness, padding)
    
    return annotated_frame, loaded_count[0], unloaded_count[0], next_display_item_id[0], current_item_count_frame

def process_video_streamlit(video_bytes, selected_class, transfer_zone_rect, conf_threshold, save_video=False):
    """Process video with item tracking - returns generator for streaming"""
    if model is None:
        st.error("Model not loaded. Cannot process video.")
        yield None, None, None, None, None
        return
    
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_bytes)
        temp_input_path = tfile.name
    
    # Initialize tracking variables
    item_id_map = {}
    next_display_item_id = [1]  # Use list for mutable reference
    item_zone_tracking_info = {}
    loaded_count = [0]  # Use list for mutable reference
    unloaded_count = [0]  # Use list for mutable reference
    
    output_video_path = None
    writer = None
    
    try:
        cap = cv2.VideoCapture(temp_input_path)
        
        if not cap.isOpened():
            st.error("Could not open video file for processing")
            yield None, None, None, None, None
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if save_video:
            # Create output video writer
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_file:
                output_video_path = output_file.name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, max(fps, 1), (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame, current_loaded, current_unloaded, current_next_id, items_in_frame = process_single_frame(
                frame, model, selected_class, transfer_zone_rect, conf_threshold,
                item_id_map, next_display_item_id, item_zone_tracking_info,
                loaded_count, unloaded_count, show_zone=False  # Completely transparent during processing
            )
            
            if save_video and writer is not None:
                writer.write(annotated_frame)
            
            # Yield current state for live updates
            total_unique_items = current_next_id - 1 if current_next_id > 1 else 0
            progress = frame_count / total_frames if total_frames > 0 else 0
            
            yield annotated_frame, current_loaded, current_unloaded, total_unique_items, progress
        
        cap.release()
        if writer is not None:
            writer.release()
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        yield None, None, None, None, None
    finally:
        # Clean up temp input file
        try:
            os.unlink(temp_input_path)
        except:
            pass
    
    # Return final results
    final_unique = next_display_item_id[0] - 1 if next_display_item_id[0] > 1 else 0
    yield None, loaded_count[0], unloaded_count[0], final_unique, 1.0

def live_stream_processing(selected_class, transfer_zone_rect, conf_threshold, camera_index=0):
    """Process live camera stream"""
    if model is None:
        st.error("Model not loaded. Cannot process live stream.")
        return
    
    # Initialize tracking variables
    item_id_map = {}
    next_display_item_id = [1]
    item_zone_tracking_info = {}
    loaded_count = [0]
    unloaded_count = [0]
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        st.error(f"Could not open camera {camera_index}")
        return
    
    # Create placeholders for live stream
    stream_placeholder = st.empty()
    live_metrics_col1, live_metrics_col2, live_metrics_col3 = st.columns(3)
    
    stop_button = st.button("Stop Live Stream", key="stop_live_stream")
    
    try:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            
            # Process frame
            annotated_frame, _, _, _, _ = process_single_frame(
                frame, model, selected_class, transfer_zone_rect, conf_threshold,
                item_id_map, next_display_item_id, item_zone_tracking_info,
                loaded_count, unloaded_count, show_zone=False
            )
            
            # Display frame
            stream_placeholder.image(cv2_to_pil(annotated_frame), channels="RGB", use_column_width=True)
            
            # Update metrics
            total_unique = next_display_item_id[0] - 1 if next_display_item_id[0] > 1 else 0
            live_metrics_col1.metric("Unique Items", total_unique)
            live_metrics_col2.metric("Loaded", loaded_count[0])
            live_metrics_col3.metric("Unloaded", unloaded_count[0])
            
            time.sleep(0.03)  # ~30 FPS
            
    except Exception as e:
        st.error(f"Error in live stream: {str(e)}")
    finally:
        cap.release()

# --- Streamlit UI ---

# Logo - handle missing logo gracefully
try:
    st.image("logo.png", width=150)
except:
    st.markdown("### 📦 Enhanced Item Tracking App")

st.title("📦/🛍️ Enhanced Item Tracking & Counting App")

# Initialize session state variables
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

# Mode selection
st.sidebar.header("📹 Processing Mode")
processing_mode = st.sidebar.radio(
    "Choose processing mode:",
    ["Upload Video", "Live Stream/Webcam"],
    index=0 if st.session_state.processing_mode == "Upload Video" else 1
)
st.session_state.processing_mode = processing_mode

if processing_mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    # Reset first frame when new file is uploaded
    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.first_frame_roi = None
            st.session_state.uploaded_file_name = uploaded_file.name
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ Settings")
        
        # Only show class selection if model is loaded
        if model is not None:
            st.session_state.selected_class = st.selectbox(
                "Select item type to track:",
                options=AVAILABLE_CLASSES,
                index=AVAILABLE_CLASSES.index(st.session_state.selected_class) if st.session_state.selected_class in AVAILABLE_CLASSES else 0
            )
        else:
            st.error("Model not loaded. Please ensure 'best_bag_box.pt' is available.")
            st.session_state.selected_class = st.selectbox(
                "Select item type to track:",
                options=AVAILABLE_CLASSES,
                disabled=True
            )
        
        conf_threshold_st = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.55, 0.05)
        
        st.markdown("---")
        st.subheader("🎯 Define Transfer Zone (ROI)")
        
        # Get first frame only when needed and not already cached
        if uploaded_file is not None and st.session_state.first_frame_roi is None:
            with st.spinner("Extracting first frame..."):
                st.session_state.first_frame_roi = get_first_frame(uploaded_file)
        
        # ROI Selection Method
        roi_method = st.radio("Select ROI Method:", 
                             ["Manual Input", "Percentage Based", "Interactive Drag & Drop"], 
                             horizontal=False)
        
        if roi_method == "Manual Input":
            roi_x = st.number_input("ROI X (top-left)", value=st.session_state.roi_coords_manual["x"], min_value=0)
            roi_y = st.number_input("ROI Y (top-left)", value=st.session_state.roi_coords_manual["y"], min_value=0)
            roi_w = st.number_input("ROI Width", value=st.session_state.roi_coords_manual["w"], min_value=10)
            roi_h = st.number_input("ROI Height", value=st.session_state.roi_coords_manual["h"], min_value=10)
            
        elif roi_method == "Percentage Based":
            if st.session_state.first_frame_roi is not None:
                frame_h, frame_w = st.session_state.first_frame_roi.shape[:2]
                st.info(f"Video dimensions: {frame_w} x {frame_h}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    x_percent = st.slider("X Position (%)", 0, 90, 20)
                    w_percent = st.slider("Width (%)", 10, 80, 40)
                with col_b:
                    y_percent = st.slider("Y Position (%)", 0, 90, 30)
                    h_percent = st.slider("Height (%)", 10, 80, 30)
                
                roi_x = int(frame_w * x_percent / 100)
                roi_y = int(frame_h * y_percent / 100)
                roi_w = int(frame_w * w_percent / 100)
                roi_h = int(frame_h * h_percent / 100)
            else:
                st.warning("Upload video first to use percentage-based ROI")
                roi_x, roi_y, roi_w, roi_h = 100, 100, 300, 200
                
        else:  # Interactive Drag & Drop
            if st.session_state.first_frame_roi is not None:
                roi_x, roi_y, roi_w, roi_h = interactive_roi_selector(st.session_state.first_frame_roi)
            else:
                st.warning("Upload video first to use interactive ROI selection")
                roi_x, roi_y, roi_w, roi_h = 100, 100, 300, 200
        
        # Update session state
        st.session_state.roi_coords_manual = {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h}
        
        # Show ROI preview (with semi-transparent zone for adjustment)
        if st.session_state.first_frame_roi is not None and roi_method != "Interactive Drag & Drop":
            frame_with_roi_preview = draw_roi_preview(
                st.session_state.first_frame_roi, 
                (roi_x, roi_y, roi_w, roi_h), 
                show_transparent=False  # Show semi-transparent for adjustment
            )
            display_opencv_image(frame_with_roi_preview, "ROI Preview on First Frame")
        elif roi_method != "Interactive Drag & Drop":
            st.caption("Upload a video to see ROI preview on its first frame.")
    
    with col2:
        st.subheader("📊 Results")
        
        # Video output section
        st.markdown("### 🎥 Processed Video Output")
        video_output_placeholder = st.empty()
        
        # Metrics section
        st.markdown("### 📈 Real-time Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        loaded_metric = metrics_col1.empty()
        unloaded_metric = metrics_col2.empty()
        unique_metric = metrics_col3.empty()
        
        # Progress section
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Save video option
        save_video_option = st.checkbox("💾 Save processed video", value=False)
        
        # Process video button
        if uploaded_file is not None and model is not None:
            if st.button(f"🚀 Process Video for {st.session_state.selected_class}s", use_container_width=True):
                st.info(f"Processing video for '{st.session_state.selected_class}'... Processing with transparent transfer zone.")
                
                video_bytes = uploaded_file.getvalue()
                transfer_zone_from_ui = (
                    st.session_state.roi_coords_manual["x"],
                    st.session_state.roi_coords_manual["y"],
                    st.session_state.roi_coords_manual["w"],
                    st.session_state.roi_coords_manual["h"]
                )
                
                # Process video with streaming results
                progress_bar = progress_placeholder.progress(0)
                
                for result in process_video_streamlit(video_bytes, st.session_state.selected_class, 
                                                    transfer_zone_from_ui, conf_threshold_st, 
                                                    save_video=save_video_option):
                    
                    annotated_frame, loaded, unloaded, unique, progress = result
                    
                    if annotated_frame is not None:
                        # Update live video feed
                        video_output_placeholder.image(cv2_to_pil(annotated_frame), 
                                                     caption="Live Processing", use_column_width=True)
                    
                    if loaded is not None:
                        # Update metrics
                        loaded_metric.metric("Total Loaded", loaded)
                        unloaded_metric.metric("Total Unloaded", unloaded) 
                        unique_metric.metric("Unique Items", unique)
                        
                        # Update progress
                        if progress is not None:
                            progress_bar.progress(progress)
                            status_placeholder.text(f"Processing... {progress*100:.1f}% complete")
                
                status_placeholder.success("✅ Video processing completed!")
                progress_placeholder.empty()
                
                if save_video_option:
                    st.success("💾 Video saved successfully!")
                    # Note: In a real implementation, you'd provide download link here
        
        elif uploaded_file is not None and model is None:
            st.error("Cannot process video: Model not loaded. Please ensure 'best_bag_box.pt' is available.")
        else:
            st.info("📁 Upload a video file to begin processing...")

else:  # Live Stream Mode
    st.subheader("📹 Live Stream Processing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ Live Stream Settings")
        
        # Class selection
        if model is not None:
            selected_class_live = st.selectbox(
                "Select item type to track:",
                options=AVAILABLE_CLASSES,
                key="live_class_select"
            )
        else:
            st.error("Model not loaded. Please ensure 'best_bag_box.pt' is available.")
            selected_class_live = AVAILABLE_CLASSES[0] if AVAILABLE_CLASSES else 'bag'
        
        conf_threshold_live = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.55, 0.05, key="live_conf")
        camera_index = st.number_input("Camera Index", value=0, min_value=0, max_value=10, key="camera_idx")
        
        # ROI for live stream (simplified)
        st.markdown("### 🎯 Transfer Zone for Live Stream")
        live_roi_x = st.slider("ROI X", 0, 640, 100, key="live_roi_x")
        live_roi_y = st.slider("ROI Y", 0, 480, 100, key="live_roi_y")
        live_roi_w = st.slider("ROI Width", 50, 400, 200, key="live_roi_w")
        live_roi_h = st.slider("ROI Height", 50, 300, 150, key="live_roi_h")
        
        transfer_zone_live = (live_roi_x, live_roi_y, live_roi_w, live_roi_h)
        
        if st.button("🔴 Start Live Stream", use_container_width=True, key="start_live"):
            st.session_state.live_stream_active = True
        
        if st.button("⏹️ Stop Live Stream", use_container_width=True, key="stop_live"):
            st.session_state.live_stream_active = False
    
    with col2:
        st.subheader("📺 Live Stream Output")
        
        if 'live_stream_active' not in st.session_state:
            st.session_state.live_stream_active = False
        
        if st.session_state.live_stream_active and model is not None:
            live_stream_processing(selected_class_live, transfer_zone_live, conf_threshold_live, camera_index)
        elif model is None:
            st.error("Model not loaded. Cannot start live stream.")
        else:
            st.info("Click 'Start Live Stream' to begin real-time processing")
            st.markdown("### 📊 Live Metrics")
            st.markdown("Metrics will appear here during live streaming...")

st.markdown("---")
st.caption("Enhanced App by ElevateTrust.AI Solutions")

# Additional Features Section
st.sidebar.markdown("---")
st.sidebar.header("🔧 Additional Features")

if st.sidebar.button("🎯 Test ROI Visibility"):
    st.sidebar.info("ROI will be semi-transparent during adjustment and completely transparent during processing")

if st.sidebar.button("📷 Camera Test"):
    st.sidebar.info("Testing camera connection...")
    try:
        test_cap = cv2.VideoCapture(0)
        if test_cap.isOpened():
            st.sidebar.success("✅ Camera connected successfully!")
            test_cap.release()
        else:
            st.sidebar.error("❌ Cannot connect to camera")
    except:
        st.sidebar.error("❌ Camera test failed")

# Performance monitoring
st.sidebar.markdown("### 📊 Performance Info")
if 'frame_processing_time' not in st.session_state:
    st.session_state.frame_processing_time = []

if st.sidebar.button("Clear Performance Data"):
    st.session_state.frame_processing_time = []
    st.sidebar.success("Performance data cleared!")

# Export settings
st.sidebar.markdown("### 💾 Export Options")
video_quality = st.sidebar.selectbox("Video Quality", ["High", "Medium", "Low"], index=1)
export_format = st.sidebar.selectbox("Export Format", ["MP4", "AVI"], index=0)

# Help section
with st.sidebar.expander("❓ Help & Tips"):
    st.markdown("""
    **ROI Adjustment Tips:**
    - Use 'Interactive Drag & Drop' for precise positioning
    - Transfer zone is semi-transparent during setup
    - Transfer zone becomes completely transparent during processing
    
    **Live Stream Tips:**
    - Ensure good lighting for better detection
    - Test camera connection before starting
    - Use appropriate confidence threshold (0.5-0.7 recommended)
    
    **Performance Tips:**
    - Lower confidence threshold for more detections
    - Higher confidence threshold for better accuracy
    - Use 'Medium' quality for balanced performance
    """)

# Debug information
if st.sidebar.checkbox("🐛 Debug Mode"):
    st.sidebar.markdown("### Debug Information")
    st.sidebar.json({
        "Model Loaded": model is not None,
        "Available Classes": AVAILABLE_CLASSES,
        "Current ROI": st.session_state.roi_coords_manual,
        "Processing Mode": st.session_state.processing_mode,
        "Session State Keys": list(st.session_state.keys())
    })