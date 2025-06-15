import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
from PIL import Image
import time
import base64

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
    
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def cv2_to_pil(cv2_image):
    """Convert OpenCV image (BGR) to PIL Image (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def display_opencv_image(cv2_image, caption="", use_container_width=True):
    """Display OpenCV image in Streamlit"""
    pil_image = cv2_to_pil(cv2_image)
    st.image(pil_image, caption=caption, use_container_width=use_container_width)

def draw_roi_preview(frame, roi_coords, show_transparent=False):
    frame_with_roi = frame.copy()
    tx, ty, tw, th = roi_coords
    
    if not show_transparent: 
        overlay = frame_with_roi.copy()
        cv2.rectangle(overlay, (tx, ty), (tx+tw, ty+th), (255, 0, 255), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame_with_roi, 1-alpha, 0, frame_with_roi)
        cv2.rectangle(frame_with_roi, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 2)
        cv2.putText(frame_with_roi, "Transfer Zone", (tx, ty-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    elif show_transparent and tx >= 0 and ty >= 0 and tw > 0 and th > 0 : 
         cv2.rectangle(frame_with_roi, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 1) 
    
    return frame_with_roi

# --- Configuration ---
MODEL_PATH = 'best_bag_box.pt'

@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        original_roboflow_name = "counting  industry - v1 2024-06-18 6-04am"
        new_name_for_roboflow_class = "bag" 
        remapped = False
        if hasattr(model, 'model') and hasattr(model.model, 'names') and isinstance(model.model.names, dict):
            for idx, name_in_model in model.model.names.items():
                if name_in_model == original_roboflow_name:
                    model.model.names[idx] = new_name_for_roboflow_class
                    remapped = True
                    break
        else:
            st.warning("Model names attribute not found or not in expected format. Skipping remapping.")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        st.error(f"Please ensure '{os.path.basename(model_path)}' is in your app directory or provide the correct path.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_yolo_model(MODEL_PATH)
if model is not None and hasattr(model, 'model') and hasattr(model.model, 'names'):
    AVAILABLE_CLASSES = list(model.model.names.values())
else:
    AVAILABLE_CLASSES = ['bag', 'box'] 
    if model is None:
      st.warning("Using fallback classes as model failed to load. Please check model file and path.")
    else: 
      st.warning("Model loaded, but class names could not be determined. Using fallback classes.")

def get_first_frame(uploaded_file_or_path):
    cap = None
    temp_path = None
    try:
        if hasattr(uploaded_file_or_path, 'read'): 
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_or_path.name)[1]) as tfile:
                tfile.write(uploaded_file_or_path.read())
                temp_path = tfile.name
            video_source = temp_path
        else: 
            video_source = uploaded_file_or_path
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            # st.error(f"Could not open video source: {video_source}") # Avoid duplicate errors if called during processing
            return None
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        else:
            # st.error("Could not read first frame from video")
            return None
    except Exception as e:
        # st.error(f"Error extracting first frame: {str(e)}")
        return None
    finally:
        if cap: cap.release()
        if temp_path and os.path.exists(temp_path): 
            try: os.unlink(temp_path)
            except Exception as e: st.warning(f"Could not delete temp file {temp_path}: {e}")

def process_single_frame(frame, model_instance, selected_class, transfer_zone_coords, conf_threshold, 
                        item_id_map, next_display_item_id, item_zone_tracking_info, 
                        loaded_count, unloaded_count, show_roi_on_video_frames=False):
    annotated_frame = frame.copy()
    tx, ty, tw, th = transfer_zone_coords
    TRANSFER_ZONE_COORDS_x1y1x2y2 = (tx, ty, tx + tw, ty + th)

    if show_roi_on_video_frames and tx >= 0 and ty >= 0 and tw > 0 and th > 0:
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (tx, ty), (tx + tw, ty + th), (255, 0, 255), -1) 
        alpha = 0.3 
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
        cv2.rectangle(annotated_frame, (tx, ty), (tx + tw, ty + th), (255, 0, 255), 2) 
    
    class_index = None 
    if hasattr(model_instance, 'model') and hasattr(model_instance.model, 'names') and model_instance.model.names:
        class_names_list = list(model_instance.model.names.values())
        if selected_class in class_names_list:
            class_index = class_names_list.index(selected_class)
    current_item_count_frame = 0
    try:
        track_args = {'conf': conf_threshold, 'persist': True, 'tracker': "bytetrack.yaml", 'verbose': False}
        if class_index is not None: track_args['classes'] = [class_index]
        results = model_instance.track(frame, **track_args)
    except Exception as e:
        # st.error(f"Tracking error: {e}") # Reduce error spam during processing
        return annotated_frame, loaded_count[0], unloaded_count[0], next_display_item_id[0], current_item_count_frame
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes_data = results[0].boxes.xyxy.cpu().numpy()
        confs_data = results[0].boxes.conf.cpu().numpy()
        original_tracker_ids_data = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None
        
        for i in range(len(boxes_data)):
            x1_item, y1_item, x2_item, y2_item = map(int, boxes_data[i])
            current_item_count_frame += 1
            item_cls_id_tensor = results[0].boxes.cls[i] if results[0].boxes.cls is not None else None
            item_cls_id = int(item_cls_id_tensor.cpu()) if item_cls_id_tensor is not None else None
            item_class_name = selected_class 
            if item_cls_id is not None and hasattr(model_instance, 'model') and hasattr(model_instance.model, 'names') and item_cls_id in model_instance.model.names:
                 item_class_name = model_instance.model.names[item_cls_id]
            label_base = f"{item_class_name} {confs_data[i]:.2f}"
            cx, cy = (x1_item + x2_item) // 2, (y1_item + y2_item) // 2
            display_id_for_this_item = None 

            if original_tracker_ids_data is not None and i < len(original_tracker_ids_data): 
                original_tracker_id = original_tracker_ids_data[i]
                if original_tracker_id not in item_id_map:
                    item_id_map[original_tracker_id] = next_display_item_id[0]
                    next_display_item_id[0] += 1
                display_id_for_this_item = item_id_map[original_tracker_id]
                label = f"ID:{display_id_for_this_item} {label_base}"
                currently_in_zone = is_centroid_in_zone(cx, cy, TRANSFER_ZONE_COORDS_x1y1x2y2)
                if display_id_for_this_item not in item_zone_tracking_info:
                    item_zone_tracking_info[display_id_for_this_item] = {"was_in_zone": currently_in_zone, "has_been_counted_load": False, "has_been_counted_unload": False}
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
            else: label = label_base
            item_color = (0, 255, 0) if item_class_name.lower() == 'box' else (255, 165, 0) 
            cv2.rectangle(annotated_frame, (x1_item, y1_item), (x2_item, y2_item), item_color, 2)
            cv2.putText(annotated_frame, label, (x1_item, y1_item - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, item_color, 2)
            if display_id_for_this_item is not None : cv2.circle(annotated_frame, (cx, cy), 5, item_color, -1) 
    
    y_offset, font_face, font_scale, text_thickness, bg_color, padding = 30, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, (50,50,50), 5
    draw_text_with_background(annotated_frame, f"Loaded: {loaded_count[0]}", (10, y_offset), font_face, font_scale, (0,255,0), bg_color, text_thickness, padding)
    y_offset += 40
    draw_text_with_background(annotated_frame, f"Unloaded: {unloaded_count[0]}", (10, y_offset), font_face, font_scale, (0,0,255), bg_color, text_thickness, padding)
    y_offset += 40
    draw_text_with_background(annotated_frame, f"Items in frame: {current_item_count_frame}", (10, y_offset), font_face, 0.7, (0,255,255), bg_color, text_thickness, padding)
    return annotated_frame, loaded_count[0], unloaded_count[0], next_display_item_id[0], current_item_count_frame

def process_video_streamlit(video_bytes, selected_class, transfer_zone_rect, conf_threshold, show_roi_on_video_toggle_val):
    if model is None:
        yield None, None, None, None, None, None 
        return
    temp_input_path, output_video_path, writer, cap = None, None, None, None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(video_bytes)
            temp_input_path = tfile.name
        item_id_map, next_display_item_id, item_zone_tracking_info, loaded_count, unloaded_count = {}, [1], {}, [0], [0]
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            yield None, None, None, None, None, None; return
        fps, width, height, total_frames = max(cap.get(cv2.CAP_PROP_FPS),1), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_file_obj: output_video_path = output_file_obj.name
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            yield None, None, None, None, None, output_video_path; return
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break
            frame_count += 1
            annotated_frame, current_loaded, current_unloaded, current_next_id, items_in_frame = process_single_frame(
                frame, model, selected_class, transfer_zone_rect, conf_threshold, item_id_map, next_display_item_id, 
                item_zone_tracking_info, loaded_count, unloaded_count, 
                show_roi_on_video_frames=show_roi_on_video_toggle_val)
            if writer: writer.write(annotated_frame)
            total_unique_items = current_next_id - 1 if current_next_id > 1 else 0
            progress = frame_count / total_frames if total_frames > 0 else 0
            yield annotated_frame, current_loaded, current_unloaded, total_unique_items, progress, None 
    except Exception as e:
        st.error(f"Error processing video: {str(e)}") 
        loaded_val = loaded_count[0] if 'loaded_count' in locals() else 0
        unloaded_val = unloaded_count[0] if 'unloaded_count' in locals() else 0
        unique_val = (next_display_item_id[0]-1) if 'next_display_item_id' in locals() and next_display_item_id[0] > 1 else 0
        yield None, loaded_val, unloaded_val, unique_val, None, output_video_path 
    finally:
        if cap: cap.release()
        if writer: writer.release()
        if temp_input_path and os.path.exists(temp_input_path):
            try: os.unlink(temp_input_path)
            except Exception as e_unlink: st.warning(f"Could not delete temp input file {temp_input_path}: {e_unlink}")
        current_processed_path = None
        if output_video_path and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            current_processed_path = output_video_path
        else: 
            if output_video_path and os.path.exists(output_video_path): 
                try: os.unlink(output_video_path)
                except Exception as e_unlink_out: st.warning(f"Could not delete temp output file {output_video_path}: {e_unlink_out}")
    final_loaded = loaded_count[0]
    final_unloaded = unloaded_count[0]
    final_unique = next_display_item_id[0] - 1 if next_display_item_id[0] > 1 else 0
    yield None, final_loaded, final_unloaded, final_unique, 1.0, current_processed_path

def live_stream_processing_loop(selected_class, transfer_zone_rect, conf_threshold, source_identifier): # Modified: source_identifier
    if model is None: 
        st.error("Model not loaded.")
        st.session_state.live_stream_active = False 
        return

    cap = cv2.VideoCapture(source_identifier) # Modified: use source_identifier
    if not cap.isOpened(): 
        st.error(f"Could not open live source: {source_identifier}.")
        st.session_state.live_stream_active = False 
        return

    item_id_map = {}
    next_display_item_id = [1]
    item_zone_tracking_info = {}
    loaded_count = [0] 
    unloaded_count = [0]

    stream_placeholder = st.empty()
    live_metrics_col1, live_metrics_col2, live_metrics_col3 = st.columns(3)
    unique_ph = live_metrics_col1.empty() 
    loaded_ph = live_metrics_col2.empty() 
    unloaded_ph = live_metrics_col3.empty()
    
    try:
        while st.session_state.get('live_stream_active', False): 
            ret, frame = cap.read()
            if not ret or frame is None: 
                st.warning("Failed to read frame from live source. Stream may have ended or encountered an issue.")
                st.session_state.live_stream_active = False 
                break
            
            show_roi_on_live_feed = st.session_state.get('show_roi_on_live_stream_toggle', False)

            annotated_frame, current_total_loaded, current_total_unloaded, current_next_id, items_in_frame = process_single_frame(
                frame, model, selected_class, transfer_zone_rect, conf_threshold, 
                item_id_map, next_display_item_id, item_zone_tracking_info, 
                loaded_count, unloaded_count, 
                show_roi_on_video_frames=show_roi_on_live_feed
            )
            
            stream_placeholder.image(cv2_to_pil(annotated_frame), channels="RGB", use_container_width=True) 
            
            total_unique = current_next_id - 1 if current_next_id > 1 else 0
            
            unique_ph.metric("Unique Items", total_unique)
            loaded_ph.metric("Loaded", current_total_loaded) 
            unloaded_ph.metric("Unloaded", current_total_unloaded)
            
    except Exception as e: 
        st.error(f"Error in live stream: {str(e)}")
    finally:
        if cap.isOpened(): 
            cap.release()
        stream_placeholder.empty() 
        unique_ph.empty() 
        loaded_ph.empty() 
        unloaded_ph.empty()
        if st.session_state.get('live_stream_active', False): 
             st.info("Live stream stopped.")
        st.session_state.live_stream_active = False 

# --- Centering the Logo and Title ---
logo_path = "resources/Original logo WHITE-01.png" # Make sure this path is correct
col1, col2, col3 = st.columns([1, 2, 1]) 
with col2: 
    with st.container():
        # st.markdown('<div class="centered-header-container">', unsafe_allow_html=True) # Optional if pure st elements are used
        try:
            if os.path.exists(logo_path):
                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" alt="Logo" width="210">
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else: # Fallback if logo file not found
                st.markdown("<h3 style='text-align: center; color: #FF4B4B;'>Logo not found</h3>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>📦 CamEdge App</h3>", unsafe_allow_html=True)

        except Exception as logo_e: # Catch other potential errors with logo loading/encoding
            st.markdown(f"<h3 style='text-align: center; color: #FF4B4B;'>Error displaying logo: {logo_e}</h3>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>📦 CamEdge App</h3>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>CamEdge - Item Tracking and Counting System</h1>", unsafe_allow_html=True)
        # st.markdown('</div>', unsafe_allow_html=True) # Optional


default_states = {
    'roi_coords_manual': {"x": 100, "y": 100, "w": 300, "h": 200},
    'first_frame_roi': None,
    'selected_class': AVAILABLE_CLASSES[0] if AVAILABLE_CLASSES else 'bag',
    'uploaded_file_name': None,
    'processing_mode': "Upload Video",
    'processed_video_path': None,
    'live_stream_state': "initial",
    'live_stream_captured_frame': None,
    'live_stream_roi_coords': (100, 100, 200, 150),
    'live_stream_active': False,
    'final_loaded_count': 0, 
    'final_unloaded_count': 0, 
    'final_unique_count': 0, 
    'processing_status_message': "", 
    'is_processing': False,
    'show_roi_on_video_toggle_upload': False,
    'show_roi_on_live_stream_toggle': False,
    'live_stream_input_type': "Webcam", # New
    'rtsp_url': "rtsp://",              # New
    'live_camera_idx': 0 # Ensure this is initialized for webcam mode
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.sidebar.header("📹 Processing Mode")
processing_mode_options = ["Upload Video", "Live Stream/Webcam"]
current_mode_idx = processing_mode_options.index(st.session_state.processing_mode)
new_processing_mode = st.sidebar.radio("Choose processing mode:", processing_mode_options, index=current_mode_idx, key="processing_mode_radio")

if st.session_state.processing_mode != new_processing_mode: 
    st.session_state.processing_mode = new_processing_mode
    if new_processing_mode == "Upload Video":
        st.session_state.live_stream_state = "initial"; st.session_state.live_stream_active = False; st.session_state.live_stream_captured_frame = None
    else: 
        st.session_state.first_frame_roi = None; st.session_state.processed_video_path = None
        st.session_state.processing_status_message = ""; st.session_state.is_processing = False 
    st.rerun()

if st.session_state.processing_mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.first_frame_roi = None 
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.processed_video_path = None 
            st.session_state.processing_status_message = ""
            st.session_state.final_loaded_count = 0
            st.session_state.final_unloaded_count = 0
            st.session_state.final_unique_count = 0
            st.session_state.is_processing = False
            # st.rerun() # Consider if rerun is always needed here or only on first frame extraction
    else: 
        st.session_state.uploaded_file_name = None

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("⚙️ Settings")
        if model is not None:
            sel_cls_idx = AVAILABLE_CLASSES.index(st.session_state.selected_class) if st.session_state.selected_class in AVAILABLE_CLASSES else 0
            st.session_state.selected_class = st.selectbox("Select item type to track:", options=AVAILABLE_CLASSES, index=sel_cls_idx)
        else:
            st.error("Model not loaded."); st.session_state.selected_class = st.selectbox("Select item type:", options=AVAILABLE_CLASSES, disabled=True)
        conf_threshold_st = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.55, 0.05, key="conf_thresh_upload")
        
        st.toggle("Show ROI on Processed Video", 
                         key="show_roi_on_video_toggle_upload", 
                         value=st.session_state.show_roi_on_video_toggle_upload,
                         help="If ON, the transfer zone will be drawn semi-transparently on the processed video frames.")

        st.markdown("---"); st.subheader("🎯 Define Transfer Zone (ROI)")
        if uploaded_file is not None and st.session_state.first_frame_roi is None and not st.session_state.is_processing: 
            with st.spinner("Extracting first frame..."):
                st.session_state.first_frame_roi = get_first_frame(uploaded_file)
                if st.session_state.first_frame_roi is None: st.error("Failed to get first frame.")
                else: st.rerun() 
        
        roi_method_options = ["Manual Input", "Percentage Based"]
        roi_method_idx = st.session_state.get('roi_method_pref_idx',0) if st.session_state.first_frame_roi is not None else 0
        roi_method = st.radio("Select ROI Method:", roi_method_options, index=roi_method_idx, horizontal=False, key="roi_method_upload")
        if st.session_state.first_frame_roi is not None: st.session_state.roi_method_pref_idx = roi_method_options.index(roi_method)

        roi_x, roi_y, roi_w, roi_h = st.session_state.roi_coords_manual.values()
        if roi_method == "Manual Input":
            roi_x = st.number_input("ROI X", value=roi_x, min_value=0, key="man_x")
            roi_y = st.number_input("ROI Y", value=roi_y, min_value=0, key="man_y")
            roi_w = st.number_input("ROI Width", value=roi_w, min_value=10, key="man_w")
            roi_h = st.number_input("ROI Height", value=roi_h, min_value=10, key="man_h")
        elif roi_method == "Percentage Based" and st.session_state.first_frame_roi is not None:
            fh, fw = st.session_state.first_frame_roi.shape[:2]
            st.info(f"Video dims: {fw}x{fh}")
            col_a, col_b = st.columns(2)
            with col_a: x_pct=st.slider("X Pos (%)",0,90,20,key="px"); w_pct=st.slider("Width (%)",10,min(100-x_pct,90),40,key="pw") 
            with col_b: y_pct=st.slider("Y Pos (%)",0,90,30,key="py"); h_pct=st.slider("Height (%)",10,min(100-y_pct,90),30,key="ph") 
            roi_x, roi_y, roi_w, roi_h = int(fw*x_pct/100), int(fh*y_pct/100), int(fw*w_pct/100), int(fh*h_pct/100)
        elif roi_method == "Percentage Based": st.warning("Upload video for percentage ROI.")
        st.session_state.roi_coords_manual = {"x":roi_x, "y":roi_y, "w":roi_w, "h":roi_h}
        if st.session_state.first_frame_roi is not None:
            preview = draw_roi_preview(st.session_state.first_frame_roi, (roi_x,roi_y,roi_w,roi_h), False)
            display_opencv_image(preview, "ROI Preview", use_container_width=True)
        elif uploaded_file and not st.session_state.is_processing : st.caption("Extracting frame...")
        elif not st.session_state.is_processing: st.caption("Upload video for ROI.")
    
    with col2:
        st.subheader("📊 Results")
        video_display_placeholder = st.empty()
        st.markdown("### 📈 Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        loaded_metric_ph = m_col1.empty()
        unloaded_metric_ph = m_col2.empty()
        unique_metric_ph = m_col3.empty()
        progress_placeholder = st.empty()
        status_message_placeholder = st.empty()

        if st.session_state.is_processing:
            status_message_placeholder.info(st.session_state.processing_status_message)
        elif st.session_state.processing_status_message == "✅ Video processing completed!":
            status_message_placeholder.success(st.session_state.processing_status_message)
            loaded_metric_ph.metric("Total Loaded", st.session_state.final_loaded_count)
            unloaded_metric_ph.metric("Total Unloaded", st.session_state.final_unloaded_count)
            unique_metric_ph.metric("Total Unique Items", st.session_state.final_unique_count)
            if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                try:
                    cap_thumb = cv2.VideoCapture(st.session_state.processed_video_path)
                    if cap_thumb.isOpened():
                        ret_thumb, frame_thumb = cap_thumb.read()
                        if ret_thumb:
                            thumb_roi_coords = (st.session_state.roi_coords_manual["x"], st.session_state.roi_coords_manual["y"],
                                                st.session_state.roi_coords_manual["w"], st.session_state.roi_coords_manual["h"])
                            thumb_with_roi = draw_roi_preview(frame_thumb, thumb_roi_coords, show_transparent=True) 
                            video_display_placeholder.image(cv2_to_pil(thumb_with_roi), caption="Processed Video (first frame with ROI border)", use_container_width=True)
                        cap_thumb.release()
                except Exception as e_thumb: st.warning(f"Thumbnail error: {e_thumb}")
            else: video_display_placeholder.empty()
        else: 
            loaded_metric_ph.metric("Total Loaded", st.session_state.final_loaded_count) 
            unloaded_metric_ph.metric("Total Unloaded", st.session_state.final_unloaded_count)
            unique_metric_ph.metric("Total Unique Items", st.session_state.final_unique_count)
            if st.session_state.processing_status_message: 
                 status_message_placeholder.info(st.session_state.processing_status_message)
            else:
                 status_message_placeholder.empty() # Clears any previous message if not completed or processing
            video_display_placeholder.empty() # Clears video display if not processing or completed with path

        process_btn_disabled = not (uploaded_file and model and not st.session_state.is_processing)
        if st.button(f"🚀 Process Video for {st.session_state.selected_class}s", use_container_width=True, key="process_video_btn", disabled=process_btn_disabled):
            st.session_state.is_processing = True
            st.session_state.processing_status_message = f"Processing '{st.session_state.uploaded_file_name}'..."
            st.session_state.processed_video_path = None 
            st.session_state.final_loaded_count = 0 
            st.session_state.final_unloaded_count = 0
            st.session_state.final_unique_count = 0
            video_display_placeholder.empty() # Clear previous thumbnail
            status_message_placeholder.info(st.session_state.processing_status_message) # Show current processing message
            
            video_bytes = uploaded_file.getvalue() 
            transfer_zone = tuple(st.session_state.roi_coords_manual.values())
            show_roi_on_frames_value = st.session_state.show_roi_on_video_toggle_upload

            progress_bar = progress_placeholder.progress(0)
            final_results_from_gen = None

            for frame_data in process_video_streamlit(video_bytes, st.session_state.selected_class, 
                                                      transfer_zone, conf_threshold_st, 
                                                      show_roi_on_frames_value):
                annotated_frame, loaded, unloaded, unique, progress, temp_vid_path = frame_data
                final_results_from_gen = frame_data 

                if annotated_frame is not None:
                    video_display_placeholder.image(cv2_to_pil(annotated_frame), caption="Processing...", use_container_width=True) 
                if loaded is not None: 
                    loaded_metric_ph.metric("Loaded", loaded)
                    unloaded_metric_ph.metric("Unloaded", unloaded) 
                    unique_metric_ph.metric("Unique Items", unique)
                if progress is not None: progress_bar.progress(progress)
            
            st.session_state.is_processing = False
            if final_results_from_gen:
                _, fin_loaded, fin_unloaded, fin_unique, _, fin_path = final_results_from_gen
                st.session_state.final_loaded_count = fin_loaded if fin_loaded is not None else 0
                st.session_state.final_unloaded_count = fin_unloaded if fin_unloaded is not None else 0
                st.session_state.final_unique_count = fin_unique if fin_unique is not None else 0
                st.session_state.processing_status_message = "✅ Video processing completed!"
                if fin_path and os.path.exists(fin_path):
                    st.session_state.processed_video_path = fin_path
                else: 
                    st.session_state.processed_video_path = None
                    if fin_path is None and (final_results_from_gen[0] is None and final_results_from_gen[1] is None): # Check if generator yielded error state
                         st.session_state.processing_status_message = "⚠️ Processing error or video save failed."
            else: 
                st.session_state.processing_status_message = "❓ Processing finished with no results."

            progress_placeholder.empty()
            st.rerun()

        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path) and not st.session_state.is_processing:
            try:
                with open(st.session_state.processed_video_path, "rb") as fp_dl:
                    dl_fname = f"processed_{st.session_state.uploaded_file_name or 'video.mp4'}"
                    st.download_button("💾 Download Processed Video", data=fp_dl, file_name=dl_fname, mime="video/mp4", key="dl_btn")
            except Exception as e_dl: st.error(f"Download error: {e_dl}")
        elif not uploaded_file and not st.session_state.is_processing: st.info("📁 Upload video to process...")
        elif not model and not st.session_state.is_processing : st.error("Model not loaded.")

else:  # Live Stream Mode 
    st.subheader("📹 Live Stream Processing")
    col1_live, col2_live = st.columns([1, 1])
    with col1_live:
        st.subheader("⚙️ Live Stream Settings")
        if model is None: st.error("Model not loaded. Live stream disabled.")
        else:
            st.session_state.live_stream_input_type = st.radio(
                "Select Live Input Source:",
                ("Webcam", "RTSP Stream"),
                index=("Webcam", "RTSP Stream").index(st.session_state.live_stream_input_type),
                key="live_input_type_radio"
            )

            if st.session_state.live_stream_state == "initial":
                st.info("Setup input source and ROI before starting.")
                capture_source_display = ""
                capture_source_internal = None

                if st.session_state.live_stream_input_type == "Webcam":
                    cam_idx = st.number_input("Camera Index", value=st.session_state.get('live_camera_idx',0), min_value=0, max_value=10, key="live_cam_idx_init")
                    st.session_state.live_camera_idx = cam_idx
                    capture_source_display = f"Webcam (Index: {cam_idx})"
                    capture_source_internal = cam_idx
                else: # RTSP Stream
                    rtsp_url_input = st.text_input("RTSP Stream URL:", value=st.session_state.get('rtsp_url', 'rtsp://username:password@ip_address:port/stream'), key="rtsp_url_input")
                    st.session_state.rtsp_url = rtsp_url_input
                    capture_source_display = f"RTSP Stream ({rtsp_url_input[:30]}...)" if rtsp_url_input and rtsp_url_input != "rtsp://" else "RTSP Stream (No URL)"
                    capture_source_internal = rtsp_url_input if rtsp_url_input and rtsp_url_input != "rtsp://" else None
                
                capture_btn_disabled = (st.session_state.live_stream_input_type == "RTSP Stream" and not capture_source_internal)
                if st.button(f"📸 Capture Frame for ROI (from {capture_source_display})", key="cap_frame_btn", use_container_width=True, disabled=capture_btn_disabled):
                    if capture_source_internal is None and st.session_state.live_stream_input_type == "RTSP Stream":
                        st.error("Please enter a valid RTSP URL.")
                    else:
                        with st.spinner(f"Accessing {capture_source_display}..."):
                            cap_roi = cv2.VideoCapture(capture_source_internal)
                            if cap_roi.isOpened():
                                ret_r, frame_r = cap_roi.read(); cap_roi.release()
                                if ret_r and frame_r is not None:
                                    st.session_state.live_stream_captured_frame = frame_r
                                    st.session_state.live_stream_state = "roi_setup"; st.rerun()
                                else: st.error(f"Failed to capture from {capture_source_display}.")
                            else: st.error(f"Could not open {capture_source_display}.")

            elif st.session_state.live_stream_state == "roi_setup":
                st.info(f"Adjust ROI for live stream from {st.session_state.live_stream_input_type}.")
                if st.session_state.live_stream_captured_frame is not None:
                    frame_roi = st.session_state.live_stream_captured_frame; hf, wf = frame_roi.shape[:2]
                    cur_roi = st.session_state.live_stream_roi_coords
                    rx = st.slider("ROI X",0,max(0,wf-50),cur_roi[0],key="lrx",help=f"W:{wf}")
                    ry = st.slider("ROI Y",0,max(0,hf-50),cur_roi[1],key="lry",help=f"H:{hf}")
                    rw = st.slider("ROI W",50,max(50,wf-rx),cur_roi[2],key="lrw")
                    rh = st.slider("ROI H",50,max(50,hf-ry),cur_roi[3],key="lrh")
                    st.session_state.live_stream_roi_coords = (rx,ry,rw,rh)
                    
                    sel_cls_live_idx = AVAILABLE_CLASSES.index(st.session_state.get('live_selected_class', AVAILABLE_CLASSES[0] if AVAILABLE_CLASSES else '')) if st.session_state.get('live_selected_class') in AVAILABLE_CLASSES else 0
                    st.session_state.live_selected_class = st.selectbox("Track item:",AVAILABLE_CLASSES,index=sel_cls_live_idx,key="live_cls_sel")
                    st.session_state.live_conf_thresh = st.slider("Confidence",0.1,1.0,st.session_state.get('live_conf_thresh',0.55),0.05,key="live_conf")
                    
                    st.toggle("Show ROI on Live Stream", 
                                     key="show_roi_on_live_stream_toggle", 
                                     value=st.session_state.show_roi_on_live_stream_toggle,
                                     help="If ON, the transfer zone will be drawn semi-transparently on the live stream.")

                    start_live_disabled = (st.session_state.live_stream_input_type == "RTSP Stream" and not st.session_state.rtsp_url)
                    if st.button("🚀 Start Live Detection",key="start_live_det",use_container_width=True, disabled=start_live_disabled):
                        if start_live_disabled:
                            st.error("RTSP URL is required to start detection.")
                        else:
                            st.session_state.live_stream_active = True; st.session_state.live_stream_state = "streaming"; st.rerun()
                    if st.button("🔄 Recapture Frame",key="recap_btn",use_container_width=True):
                        st.session_state.live_stream_captured_frame=None; st.session_state.live_stream_state="initial"; st.rerun()
                else: st.error("No captured frame. Recapture."); st.session_state.live_stream_state="initial"; st.rerun()
            
            elif st.session_state.live_stream_state == "streaming":
                st.info(f"Live stream active from {st.session_state.live_stream_input_type}.")
                st.markdown(f"**Tracking:** `{st.session_state.get('live_selected_class','N/A')}` **Conf:** `{st.session_state.get('live_conf_thresh','N/A')}`")
                st.markdown(f"**ROI (Internal):** `{st.session_state.get('live_stream_roi_coords')}`")
                
                st.toggle("Show ROI on Live Stream", 
                                 key="show_roi_on_live_stream_toggle", 
                                 value=st.session_state.show_roi_on_live_stream_toggle,
                                 help="If ON, the transfer zone will be drawn semi-transparently on the live stream.")

                if st.button("⏹️ Stop Live Stream",key="stop_live_active",use_container_width=True):
                    st.session_state.live_stream_active=False; st.session_state.live_stream_state="initial"; time.sleep(0.1); st.rerun()
    
    with col2_live:
        st.subheader("📺 Live Output")
        if st.session_state.live_stream_state == "roi_setup" and st.session_state.live_stream_captured_frame is not None:
            preview_live = draw_roi_preview(st.session_state.live_stream_captured_frame, st.session_state.live_stream_roi_coords, False)
            display_opencv_image(preview_live, "ROI Preview for Live Stream", use_container_width=True)
        elif st.session_state.live_stream_state == "streaming" and st.session_state.live_stream_active:
            if model: 
                current_live_source = None
                if st.session_state.live_stream_input_type == "Webcam":
                    current_live_source = st.session_state.live_camera_idx
                elif st.session_state.live_stream_input_type == "RTSP Stream":
                    current_live_source = st.session_state.rtsp_url
                
                if current_live_source is not None and (st.session_state.live_stream_input_type == "Webcam" or (st.session_state.live_stream_input_type == "RTSP Stream" and current_live_source != "rtsp://")):
                    live_stream_processing_loop(
                        st.session_state.live_selected_class, 
                        st.session_state.live_stream_roi_coords, 
                        st.session_state.live_conf_thresh, 
                        current_live_source
                    )
                else:
                    st.error(f"Invalid source for {st.session_state.live_stream_input_type}. Please check settings.")
                    st.session_state.live_stream_active = False
                    st.session_state.live_stream_state = "initial"
            else: st.error("Model error."); st.session_state.live_stream_active=False; st.session_state.live_stream_state="initial"
        elif st.session_state.live_stream_state == "initial": st.info("Feed & metrics here on stream start.")
        else: st.info("Configure & start stream.")

footer = """
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117; /* Streamlit dark theme background */
        color: #FAFAFA; /* Streamlit dark theme text */
        text-align: center;
        padding: 10px 0; /* Adjust padding as needed */
        font-size: 12px;
        border-top: 1px solid #31333F; /* Subtle top border */
        z-index: 100; /* Ensure it's above other content if necessary */
    }
</style>
<div class="footer">
    <p>© 2024 ElevateTrust.AI. All rights reserved.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

st.sidebar.markdown("---"); st.sidebar.header("🔧 Additional Utilities") 
if st.sidebar.button("📷 Camera/RTSP Test"):
    st.sidebar.info("Testing connection...")
    cap_t = None
    test_source_display = ""
    test_source_internal = None

    input_type_for_test = st.session_state.get('live_stream_input_type', 'Webcam')
    if input_type_for_test == "Webcam":
        cam_idx_t = st.session_state.get('live_camera_idx', 0)
        test_source_internal = cam_idx_t
        test_source_display = f"Webcam (Index: {cam_idx_t})"
    else: # RTSP
        rtsp_t = st.session_state.get('rtsp_url', '')
        if not rtsp_t or rtsp_t == "rtsp://":
            st.sidebar.warning("No RTSP URL entered to test.")
        else:
            test_source_internal = rtsp_t
            test_source_display = f"RTSP ({rtsp_t[:30]}...)"

    if test_source_internal is not None:
        try:
            st.sidebar.info(f"Attempting to connect to {test_source_display}...")
            cap_t = cv2.VideoCapture(test_source_internal)
            if cap_t.isOpened():
                ret_test, frame_test = cap_t.read() # Try to read one frame
                if ret_test and frame_test is not None:
                    st.sidebar.success(f"✅ Connection to {test_source_display} OK! Frame captured.")
                else:
                    st.sidebar.error(f"❌ Connected to {test_source_display}, but failed to read frame.")
            else:
                st.sidebar.error(f"❌ Failed to open {test_source_display}.")
        except Exception as e_ct:
            st.sidebar.error(f"❌ Test failed for {test_source_display}: {e_ct}")
        finally:
            if cap_t and cap_t.isOpened():
                cap_t.release()
    elif input_type_for_test == "RTSP Stream" and (not st.session_state.get('rtsp_url') or st.session_state.get('rtsp_url') == "rtsp://"):
        pass # Warning already shown

with st.sidebar.expander("❓ Help & Tips"):
    st.markdown("""**ROI Tips:** Purple box at setup. Thin border on final thumbnail (for processed video).\n\n**Processed Video:** Toggle "Show ROI on Processed Video" to see the zone on output frames.\n\n**Live Stream:** Select Webcam or RTSP. Capture frame for ROI. Toggle "Show ROI on Live Stream" to see the zone. Good light. Test connection.\n\n**RTSP URL:** Example: `rtsp://username:password@ip_address:port/stream_path`\n\n**General:** Adjust confidence. Ensure `best_bag_box.pt` is present.""")
if st.sidebar.checkbox("🐛 Debug Mode"):
    st.sidebar.markdown("### Debug Information")
    debug_data_filtered = {}
    for k, v in st.session_state.items():
        if isinstance(v, np.ndarray) and v.ndim >= 2: 
             debug_data_filtered[k] = f"Numpy array (shape: {v.shape}, dtype: {v.dtype})"
        elif isinstance(v, bytes) and len(v) > 1024: # Avoid displaying large byte strings
             debug_data_filtered[k] = f"Bytes (length: {len(v)}) - Content omitted"
        elif isinstance(v, (list, dict)) and len(str(v)) > 1024: # Avoid large lists/dicts
            debug_data_filtered[k] = f"{type(v).__name__} (length: {len(v)}) - Content too large"
        else:
             debug_data_filtered[k] = v
    st.sidebar.json(debug_data_filtered, expanded=False)