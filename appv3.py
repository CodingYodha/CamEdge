import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
from PIL import Image

# Set page config
st.set_page_config(layout="wide", page_title="Item Tracking App")

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

# --- Main Processing function adapted for Streamlit ---
def process_video_streamlit(video_bytes, selected_class, transfer_zone_rect, conf_threshold):
    """Process video with item tracking"""
    if model is None:
        st.error("Model not loaded. Cannot process video.")
        return None, None, None, None
    
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_bytes)
        temp_input_path = tfile.name

    # Initialize tracking variables
    item_id_map_st = {}
    next_display_item_id_st = 1
    item_zone_tracking_info_st = {}
    loaded_count_st = 0
    unloaded_count_st = 0

    try:
        cap = cv2.VideoCapture(temp_input_path)
        
        if not cap.isOpened():
            st.error("Could not open video file for processing")
            return None, None, None, None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            total_frames = 1000  # Fallback for streams
        
        # Create output video writer
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_file:
            output_video_path = output_file.name
        
        # Use H264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, max(fps, 1), (width, height))
        
        if not writer.isOpened():
            st.error("Could not create output video writer")
            return None, None, None, None

        frame_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        image_placeholder = st.empty()

        tx, ty, tw, th = transfer_zone_rect
        TRANSFER_ZONE_COORDS_x1y1x2y2 = (tx, ty, tx + tw, ty + th)

        # Get class index for filtering
        try:
            class_index = AVAILABLE_CLASSES.index(selected_class)
        except ValueError:
            class_index = 0  # Default to first class if not found

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            frame_count += 1
            annotated_frame = frame.copy()

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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Run YOLO tracking
            try:
                results_st = model.track(frame, conf=conf_threshold, persist=True, 
                                       tracker="bytetrack.yaml", verbose=False, 
                                       classes=[class_index])
            except Exception as e:
                st.warning(f"Tracking error on frame {frame_count}: {str(e)}")
                writer.write(annotated_frame)
                continue
            
            current_item_count_frame = 0

            if results_st[0].boxes is not None and len(results_st[0].boxes) > 0:
                boxes_data = results_st[0].boxes.xyxy.cpu().numpy()
                confs_data = results_st[0].boxes.conf.cpu().numpy()
                
                original_tracker_ids_data = None
                if results_st[0].boxes.id is not None:
                    original_tracker_ids_data = results_st[0].boxes.id.cpu().numpy().astype(int)

                for i in range(len(boxes_data)):
                    x1_item, y1_item, x2_item, y2_item = map(int, boxes_data[i])
                    current_item_count_frame += 1
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
                            if not was_in_zone and currently_in_zone:
                                loaded_count_st += 1
                            elif was_in_zone and not currently_in_zone:
                                unloaded_count_st += 1
                            item_zone_tracking_info_st[display_id_for_this_item]["was_in_zone"] = currently_in_zone
                    
                    # Draw bounding box and label
                    item_color = (0, 255, 0) if selected_class == 'box' else (255, 165, 0)
                    cv2.rectangle(annotated_frame, (x1_item, y1_item), (x2_item, y2_item), item_color, 2)
                    cv2.putText(annotated_frame, label_st, (x1_item, y1_item - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, item_color, 2)
                    cv2.circle(annotated_frame, (cx, cy), 5, item_color, -1)

            # Display counts on frame
            y_offset_st = 30
            font_face_st = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_st = 0.8
            text_thickness_st = 2
            bg_color_st = (50, 50, 50)
            padding_st = 5
            
            draw_text_with_background(annotated_frame, f"Loaded: {loaded_count_st}", 
                                    (10, y_offset_st), font_face_st, font_scale_st, 
                                    (0, 255, 0), bg_color_st, text_thickness_st, padding_st)
            y_offset_st += 40
            draw_text_with_background(annotated_frame, f"Unloaded: {unloaded_count_st}", 
                                    (10, y_offset_st), font_face_st, font_scale_st, 
                                    (0, 0, 255), bg_color_st, text_thickness_st, padding_st)
            y_offset_st += 40
            draw_text_with_background(annotated_frame, f"{selected_class} in frame: {current_item_count_frame}", 
                                    (10, y_offset_st), font_face_st, 0.7, 
                                    (0, 255, 255), bg_color_st, text_thickness_st, padding_st)

            # Write frame to output video
            writer.write(annotated_frame)
            
            # Update Streamlit UI every 15 frames to avoid too frequent updates
            if frame_count % 15 == 0 or frame_count >= total_frames:
                # Use custom display function for OpenCV headless compatibility
                image_placeholder.image(cv2_to_pil(annotated_frame), caption=f"Processing Frame {frame_count}")
                progress_value = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress_value)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")

        cap.release()
        writer.release()
        
        status_text.text("Video processing complete!")
        progress_bar.empty()

        total_unique_items = next_display_item_id_st - 1 if next_display_item_id_st > 1 else 0
        
        return total_unique_items, loaded_count_st, unloaded_count_st, output_video_path

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None, None, None
    finally:
        # Clean up temp input file
        try:
            os.unlink(temp_input_path)
        except:
            pass

# --- Streamlit UI ---

# Logo - handle missing logo gracefully
try:
    st.image("logo.png", width=150)
except:
    st.markdown("### 📦 Item Tracking App")

st.title("📦/🛍️ Item Tracking & Counting App")

# Initialize session state variables
if 'roi_coords_manual' not in st.session_state:
    st.session_state.roi_coords_manual = {"x": 100, "y": 100, "w": 300, "h": 200}
if 'first_frame_roi' not in st.session_state:
    st.session_state.first_frame_roi = None
if 'selected_class' not in st.session_state:
    st.session_state.selected_class = AVAILABLE_CLASSES[0] if AVAILABLE_CLASSES else 'bag'
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

# Reset first frame when new file is uploaded
if uploaded_file is not None:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.first_frame_roi = None
        st.session_state.uploaded_file_name = uploaded_file.name

col1, col2 = st.columns(2)

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
    roi_method = st.radio("Select ROI Method:", ["Manual Input", "Percentage Based"], horizontal=True)
    
    if roi_method == "Manual Input":
        roi_x = st.number_input("ROI X (top-left)", value=st.session_state.roi_coords_manual["x"], min_value=0)
        roi_y = st.number_input("ROI Y (top-left)", value=st.session_state.roi_coords_manual["y"], min_value=0)
        roi_w = st.number_input("ROI Width", value=st.session_state.roi_coords_manual["w"], min_value=10)
        roi_h = st.number_input("ROI Height", value=st.session_state.roi_coords_manual["h"], min_value=10)
    else:
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
    
    # Update session state
    st.session_state.roi_coords_manual = {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h}
    
    # Show ROI preview
    if st.session_state.first_frame_roi is not None:
        frame_with_roi_preview = st.session_state.first_frame_roi.copy()
        tx, ty, tw, th = roi_x, roi_y, roi_w, roi_h
        
        # Ensure ROI is within frame bounds
        h, w = frame_with_roi_preview.shape[:2]
        tx = max(0, min(tx, w-1))
        ty = max(0, min(ty, h-1))
        tw = max(10, min(tw, w-tx))
        th = max(10, min(th, h-ty))
        
        # Draw semi-transparent ROI
        overlay_st = frame_with_roi_preview.copy()
        cv2.rectangle(overlay_st, (tx, ty), (tx+tw, ty+th), (255, 0, 255), -1)
        alpha_st = 0.3
        cv2.addWeighted(overlay_st, alpha_st, frame_with_roi_preview, 1-alpha_st, 0, frame_with_roi_preview)
        cv2.rectangle(frame_with_roi_preview, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 2)
        
        # Use custom display function for headless compatibility
        display_opencv_image(frame_with_roi_preview, "ROI Preview on First Frame")
    else:
        st.caption("Upload a video to see ROI preview on its first frame.")

with col2:
    st.subheader("📊 Results")
    output_video_placeholder = st.empty()
    
    final_counts_col1, final_counts_col2, final_counts_col3 = st.columns(3)
    unique_items_placeholder = final_counts_col1.empty()
    loaded_placeholder = final_counts_col2.empty()
    unloaded_placeholder = final_counts_col3.empty()

# Process video button
if uploaded_file is not None and model is not None:
    if st.button(f"Process Video for {st.session_state.selected_class}s", use_container_width=True):
        # Clear previous results
        output_video_placeholder.empty()
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
            conf_threshold_st
        )

        if processed_video_path and os.path.exists(processed_video_path):
            st.success("Video processed successfully!")
            output_video_placeholder.video(processed_video_path)
            
            # Clean up processed video after a delay to allow streaming
            # Note: In production, you might want to implement proper cleanup
        elif total_unique is not None:
            st.success("Processing completed!")
        else:
            st.error("Video processing failed. Please check your video file and try again.")

        # Display metrics
        unique_items_placeholder.metric(f"Total Unique '{st.session_state.selected_class}s'", 
                                      total_unique if total_unique is not None else "N/A")
        loaded_placeholder.metric("Total Loaded", 
                                total_loaded if total_loaded is not None else "N/A")
        unloaded_placeholder.metric("Total Unloaded", 
                                  total_unloaded if total_unloaded is not None else "N/A")

elif uploaded_file is not None and model is None:
    st.error("Cannot process video: Model not loaded. Please ensure 'best_bag_box.pt' is available.")
else:
    st.info("Upload a video file to begin processing...")

st.markdown("---")
st.caption("App by ElevateTrust.AI Solutions")