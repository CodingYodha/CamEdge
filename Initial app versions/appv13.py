import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
from PIL import Image
import time
import base64

st.set_page_config(layout="wide", page_title="CamEdge - Item Tracking")

# --- Helper Functions (Assume these are correct and unchanged) ---
def is_centroid_in_zone(cx, cy, zone_coords):
    x1, y1, x2, y2 = zone_coords
    return x1 <= cx <= x2 and y1 <= cy <= y2

def draw_text_with_background(img, text, position, font, font_scale, text_color, bg_color, thickness, padding):
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x - padding, y - text_height - padding), (x + text_width + padding, y + baseline + padding), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def display_opencv_image(cv2_image, caption="", use_container_width=True):
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
        cv2.putText(frame_with_roi, "Transfer Zone", (tx, ty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    elif show_transparent and tx >= 0 and ty >= 0 and tw > 0 and th > 0 : 
         cv2.rectangle(frame_with_roi, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 1) 
    return frame_with_roi

MODEL_PATH = 'best_bag_box.pt'

@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        # original_roboflow_name = "counting  industry - v1 2024-06-18 6-04am" # Example
        # new_name_for_roboflow_class = "bag" # Example
        # ... (remapping logic as before) ...
        return model
    except FileNotFoundError:
        st.session_state.model_load_error = f"Model file not found: {model_path}. Please ensure it's in the app directory."
        return None
    except Exception as e:
        st.session_state.model_load_error = f"Error loading model: {str(e)}"
        return None

# Initialize model_load_error in session state
if 'model_load_error' not in st.session_state:
    st.session_state.model_load_error = None

model = load_yolo_model(MODEL_PATH)

if model is not None and hasattr(model, 'model') and hasattr(model.model, 'names'):
    AVAILABLE_CLASSES = list(model.model.names.values())
    st.session_state.model_load_error = None # Clear error if model loaded
elif model is None and not st.session_state.model_load_error: # If model is None but no specific error was set by load_yolo_model
    st.session_state.model_load_error = "Model could not be loaded. Unknown error."

if model is None: # Fallback if model loading failed
    AVAILABLE_CLASSES = ['bag', 'box']


def get_first_frame(uploaded_file_or_path):
    cap = None; temp_path = None
    try:
        video_source = uploaded_file_or_path
        if hasattr(uploaded_file_or_path, 'read'): 
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_or_path.name)[1]) as tfile:
                tfile.write(uploaded_file_or_path.read()); temp_path = tfile.name
            video_source = temp_path
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened(): return None
        ret, frame = cap.read()
        return frame if ret and frame is not None else None
    except Exception: return None
    finally:
        if cap: cap.release()
        if temp_path and os.path.exists(temp_path): 
            try: os.unlink(temp_path)
            except Exception: pass

# --- process_single_frame (with dynamic font scaling from previous answer) ---
def process_single_frame(frame, model_instance, selected_class, transfer_zone_coords, conf_threshold, 
                        item_id_map, next_display_item_id, item_zone_tracking_info, 
                        loaded_count, unloaded_count, show_roi_on_video_frames=False):
    annotated_frame = frame.copy()
    frame_height, frame_width = frame.shape[:2] 

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
        if selected_class in class_names_list: class_index = class_names_list.index(selected_class)
    
    current_item_count_frame = 0
    detections = None # To store the specific result object

    try:
        track_args = {'conf': conf_threshold, 'persist': True, 'tracker': "bytetrack.yaml", 'verbose': False}
        if class_index is not None: track_args['classes'] = [class_index]
        results_list = model_instance.track(frame, **track_args) # track() returns a list
        if results_list: # Check if the list is not empty
            detections = results_list[0] # Get the first (and likely only) Results object
    except Exception as e:
        # st.error(f"Tracking error: {e}") # Suppress for smoother UI during processing
        return annotated_frame, loaded_count[0], unloaded_count[0], next_display_item_id[0], current_item_count_frame
    
    REFERENCE_FRAME_WIDTH = 640; BASE_FONT_SCALE_INFO = 0.8; BASE_FONT_SCALE_ITEMS = 0.7
    MIN_FONT_SCALE = 0.4; MAX_FONT_SCALE = 1.2
    scale_factor = frame_width / REFERENCE_FRAME_WIDTH
    dynamic_font_scale_info = max(MIN_FONT_SCALE, min(MAX_FONT_SCALE, BASE_FONT_SCALE_INFO * scale_factor))
    dynamic_font_scale_items = max(MIN_FONT_SCALE, min(MAX_FONT_SCALE, BASE_FONT_SCALE_ITEMS * scale_factor))
    bbox_label_font_scale = max(0.3, min(0.8, 0.6 * scale_factor))

    # **** THIS IS THE CORRECTED SECTION ****
    if detections and detections.boxes is not None and len(detections.boxes) > 0:
        boxes_data = detections.boxes.xyxy.cpu().numpy()
        confs_data = detections.boxes.conf.cpu().numpy()
        # Ensure 'id' attribute exists and is not None before trying to access it
        original_tracker_ids_data = None
        if hasattr(detections.boxes, 'id') and detections.boxes.id is not None:
            original_tracker_ids_data = detections.boxes.id.cpu().numpy().astype(int)
        
        for i in range(len(boxes_data)):
            x1_item, y1_item, x2_item, y2_item = map(int, boxes_data[i])
            current_item_count_frame += 1
            
            # Ensure 'cls' attribute exists for class ID
            item_cls_id = None
            if hasattr(detections.boxes, 'cls') and detections.boxes.cls is not None and i < len(detections.boxes.cls):
                item_cls_id_tensor = detections.boxes.cls[i]
                item_cls_id = int(item_cls_id_tensor.cpu())

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
                    loaded_count[0] += 1; info["has_been_counted_load"] = True; info["has_been_counted_unload"] = False 
                elif was_in_zone and not currently_in_zone and not info["has_been_counted_unload"]:
                    unloaded_count[0] += 1; info["has_been_counted_unload"] = True; info["has_been_counted_load"] = False 
                info["was_in_zone"] = currently_in_zone
            else: label = label_base
            item_color = (0, 255, 0) if item_class_name.lower() == 'box' else (255, 165, 0) 
            cv2.rectangle(annotated_frame, (x1_item, y1_item), (x2_item, y2_item), item_color, 2)
            cv2.putText(annotated_frame, label, (x1_item, y1_item - 10), cv2.FONT_HERSHEY_SIMPLEX, bbox_label_font_scale, item_color, 2)
            if display_id_for_this_item is not None : cv2.circle(annotated_frame, (cx, cy), 5, item_color, -1)
    
    font_face = cv2.FONT_HERSHEY_SIMPLEX; text_thickness = 2; bg_color = (50,50,50)
    base_y_offset_increment = 40; base_padding = 5
    ui_element_scale_factor = max(0.7, min(1.3, scale_factor)) 
    dynamic_y_offset_increment = int(base_y_offset_increment * ui_element_scale_factor)
    dynamic_padding = int(base_padding * ui_element_scale_factor)
    initial_y_pos = int(30 * ui_element_scale_factor) 
    y_offset = initial_y_pos
    
    draw_text_with_background(annotated_frame, f"Loaded: {loaded_count[0]}", (10,y_offset), font_face, dynamic_font_scale_info, (0,255,0), bg_color, text_thickness, dynamic_padding)
    y_offset += dynamic_y_offset_increment
    draw_text_with_background(annotated_frame, f"Unloaded: {unloaded_count[0]}", (10,y_offset), font_face, dynamic_font_scale_info, (0,0,255), bg_color, text_thickness, dynamic_padding)
    y_offset += dynamic_y_offset_increment
    draw_text_with_background(annotated_frame, f"Items in frame: {current_item_count_frame}", (10,y_offset), font_face, dynamic_font_scale_items, (0,255,255), bg_color, text_thickness, dynamic_padding)
    return annotated_frame, loaded_count[0], unloaded_count[0], next_display_item_id[0], current_item_count_frame
# --- process_video_streamlit (no changes needed here for these UI updates) ---
def process_video_streamlit(video_bytes, selected_class, transfer_zone_rect, conf_threshold, show_roi_on_video_toggle_val):
    # ... (function remains the same) ...
    if model is None: yield None, None, None, None, None, None; return
    temp_input_path, output_video_path, writer, cap = None, None, None, None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(video_bytes); temp_input_path = tfile.name
        item_id_map,next_display_item_id,item_zone_tracking_info,loaded_count,unloaded_count = {},[1],{},[0],[0]
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened(): yield None,None,None,None,None,None; return
        fps,width,height,total_frames = max(cap.get(cv2.CAP_PROP_FPS),1),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4') as out_f: output_video_path = out_f.name
        writer = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened(): yield None,None,None,None,None,output_video_path; return
        frame_count = 0
        while True:
            ret,frame = cap.read()
            if not ret or frame is None: break
            frame_count +=1
            annotated_frame,cur_loaded,cur_unloaded,cur_next_id,items_in_frame = process_single_frame(
                frame, model, selected_class, transfer_zone_rect, conf_threshold, item_id_map, next_display_item_id,
                item_zone_tracking_info, loaded_count, unloaded_count, show_roi_on_video_frames=show_roi_on_video_toggle_val)
            if writer: writer.write(annotated_frame)
            total_unique = cur_next_id -1 if cur_next_id > 1 else 0
            progress = frame_count / total_frames if total_frames > 0 else 0
            yield annotated_frame, cur_loaded, cur_unloaded, total_unique, progress, None
    except Exception as e:
        st.error(f"Err proc video: {e}")
        loaded_val = loaded_count[0] if 'loaded_count' in locals() else 0
        unloaded_val = unloaded_count[0] if 'unloaded_count' in locals() else 0
        unique_val = (next_display_item_id[0]-1) if 'next_display_item_id' in locals() and next_display_item_id[0] > 1 else 0
        yield None, loaded_val, unloaded_val, unique_val, None, output_video_path
    finally:
        if cap: cap.release()
        if writer: writer.release()
        if temp_input_path and os.path.exists(temp_input_path):
            try: os.unlink(temp_input_path)
            except Exception: pass
        current_processed_path = None
        if output_video_path and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0 :
            current_processed_path = output_video_path
        elif output_video_path and os.path.exists(output_video_path):
            try: os.unlink(output_video_path)
            except Exception: pass
    final_loaded = loaded_count[0]; final_unloaded = unloaded_count[0]
    final_unique = next_display_item_id[0] -1 if next_display_item_id[0] > 1 else 0
    yield None, final_loaded, final_unloaded, final_unique, 1.0, current_processed_path


# --- live_stream_processing_loop (no changes needed here for these UI updates) ---
def live_stream_processing_loop(selected_class, transfer_zone_rect, conf_threshold, source_identifier):
    # ... (function remains the same) ...
    if model is None: st.error("Model not loaded."); st.session_state.live_stream_active = False; return
    cap = cv2.VideoCapture(source_identifier)
    if not cap.isOpened(): st.error(f"Could not open live source: {source_identifier}."); st.session_state.live_stream_active=False; return
    item_id_map,next_disp_id,item_zone_info,loaded_ct,unloaded_ct = {},[1],{},[0],[0]
    stream_ph = st.empty()
    live_met_col1,live_met_col2,live_met_col3 = st.columns(3)
    unique_ph,loaded_ph,unloaded_ph = live_met_col1.empty(),live_met_col2.empty(),live_met_col3.empty()
    try:
        while st.session_state.get('live_stream_active', False):
            ret,frame = cap.read()
            if not ret or frame is None: st.warning("Fail to read frame."); st.session_state.live_stream_active=False; break
            show_roi_live = st.session_state.get('show_roi_on_live_stream_toggle', False)
            annot_fr,cur_tot_load,cur_tot_unload,cur_next_id,items_fr = process_single_frame(
                frame, model, selected_class, transfer_zone_rect, conf_threshold,
                item_id_map, next_disp_id, item_zone_info, loaded_ct, unloaded_ct, show_roi_on_video_frames=show_roi_live)
            stream_ph.image(cv2_to_pil(annot_fr), channels="RGB", use_container_width=True)
            total_unique = cur_next_id -1 if cur_next_id > 1 else 0
            unique_ph.metric("Unique Items", total_unique)
            loaded_ph.metric("Loaded", cur_tot_load)
            unloaded_ph.metric("Unloaded", cur_tot_unload)
    except Exception as e: st.error(f"Err live stream: {e}")
    finally:
        if cap.isOpened(): cap.release()
        stream_ph.empty(); unique_ph.empty(); loaded_ph.empty(); unloaded_ph.empty()
        if st.session_state.get('live_stream_active',False): st.info("Live stream stopped.")
        st.session_state.live_stream_active = False


# --- Centered Header Section (no changes needed) ---
logo_path = "resources/Original logo WHITE-01.png"
col1_title, col2_title, col3_title = st.columns([1, 2, 1])
with col2_title:
    with st.container():
        try:
            if os.path.exists(logo_path):
                st.markdown(f"""<div style="text-align: center;"><img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" alt="Logo" width="210"></div>""", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='text-align: center; color: #FF4B4B;'>Logo not found</h3>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>📦 CamEdge App</h3>", unsafe_allow_html=True)
        except Exception as logo_e:
            st.markdown(f"<h3 style='text-align: center; color: #FF4B4B;'>Error displaying logo: {logo_e}</h3>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>📦 CamEdge App</h3>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>CamEdge - Item Tracking and Counting System</h1>", unsafe_allow_html=True)

# --- Default Session States ---
default_roi_coords = {"x": 100, "y": 100, "w": 300, "h": 200}
default_live_roi_coords = (100, 100, 200, 150)

default_states = {
    'roi_coords_manual': default_roi_coords.copy(),
    'first_frame_roi': None,
    'selected_class': AVAILABLE_CLASSES[0] if AVAILABLE_CLASSES else 'bag',
    'uploaded_file_name': None,
    'processing_mode': "Upload Video",
    'processed_video_path': None,
    'live_stream_state': "initial",
    'live_stream_captured_frame': None,
    'live_stream_roi_coords': default_live_roi_coords,
    'live_stream_active': False,
    'final_loaded_count': 0, 'final_unloaded_count': 0, 'final_unique_count': 0,
    'processing_status_message': "", 'is_processing': False,
    'show_roi_on_video_toggle_upload': False,
    'show_roi_on_live_stream_toggle': False,
    'live_stream_input_type': "Webcam",
    'rtsp_url': "rtsp://",
    'live_camera_idx': 0,
    'model_load_error': None
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Mode Selection and Global Error Display ---
if st.session_state.model_load_error:
    st.error(f"🚨 Critical Error: {st.session_state.model_load_error} Application functionality will be limited.")

st.sidebar.header("📹 Processing Mode")
# ... (mode selection logic as before) ...
processing_mode_options = ["Upload Video", "Live Stream/Webcam"]
current_mode_idx = processing_mode_options.index(st.session_state.processing_mode)
new_processing_mode = st.sidebar.radio("Choose processing mode:", processing_mode_options, index=current_mode_idx, key="processing_mode_radio")

if st.session_state.processing_mode != new_processing_mode: 
    st.session_state.processing_mode = new_processing_mode
    # Reset states specific to the other mode
    if new_processing_mode == "Upload Video":
        st.session_state.live_stream_state = "initial"; st.session_state.live_stream_active = False; st.session_state.live_stream_captured_frame = None
    else: # Switching to Live Stream
        st.session_state.first_frame_roi = None; st.session_state.processed_video_path = None
        st.session_state.processing_status_message = ""; st.session_state.is_processing = False 
        st.session_state.final_loaded_count = 0; st.session_state.final_unloaded_count = 0; st.session_state.final_unique_count = 0
    st.rerun()


# ------------------------------ UPLOAD VIDEO MODE ------------------------------
if st.session_state.processing_mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
    
    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.first_frame_roi = None 
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.processed_video_path = None 
            st.session_state.processing_status_message = ""
            st.session_state.final_loaded_count = 0; st.session_state.final_unloaded_count = 0; st.session_state.final_unique_count = 0
            st.session_state.is_processing = False
    else: 
        st.session_state.uploaded_file_name = None

    col1, col2 = st.columns([1, 1])
    with col1:
        # --- Expander for General Settings ---
        with st.expander("⚙️ General Settings", expanded=True):
            if model is not None:
                sel_cls_idx = AVAILABLE_CLASSES.index(st.session_state.selected_class) if st.session_state.selected_class in AVAILABLE_CLASSES else 0
                st.session_state.selected_class = st.selectbox("Select item type to track:", options=AVAILABLE_CLASSES, index=sel_cls_idx, key="upload_class_select")
            else:
                st.selectbox("Select item type to track:", options=AVAILABLE_CLASSES, disabled=True, key="upload_class_select_disabled", help="Model not loaded.")
            conf_threshold_st = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.55, 0.05, key="conf_thresh_upload")
            st.toggle("Show ROI on Processed Video", 
                             key="show_roi_on_video_toggle_upload", 
                             value=st.session_state.show_roi_on_video_toggle_upload,
                             help="If ON, the transfer zone will be drawn on processed video frames.")

        st.markdown("---") 

        # --- Expander for ROI Definition ---
        with st.expander("🎯 Define Transfer Zone (ROI)", expanded=True):
            if uploaded_file is not None and st.session_state.first_frame_roi is None and not st.session_state.is_processing: 
                with st.spinner("Extracting first frame for ROI setup..."):
                    st.session_state.first_frame_roi = get_first_frame(uploaded_file)
                    if st.session_state.first_frame_roi is None: st.error("⚠️ Failed to extract first frame. Please try a different video.")
                    else: st.rerun() 
            
            roi_method_options = ["Manual Input", "Percentage Based"]
            roi_method_idx = st.session_state.get('roi_method_pref_idx',0) if st.session_state.first_frame_roi is not None else 0
            roi_method = st.radio("Select ROI Method:", roi_method_options, index=roi_method_idx, horizontal=False, key="roi_method_upload")
            if st.session_state.first_frame_roi is not None: st.session_state.roi_method_pref_idx = roi_method_options.index(roi_method)

            current_roi_for_input = st.session_state.roi_coords_manual
            roi_x, roi_y, roi_w, roi_h = current_roi_for_input["x"], current_roi_for_input["y"], current_roi_for_input["w"], current_roi_for_input["h"]

            if roi_method == "Manual Input":
                roi_x = st.number_input("ROI X", value=roi_x, min_value=0, key="man_x_upload")
                roi_y = st.number_input("ROI Y", value=roi_y, min_value=0, key="man_y_upload")
                roi_w = st.number_input("ROI Width", value=roi_w, min_value=10, key="man_w_upload")
                roi_h = st.number_input("ROI Height", value=roi_h, min_value=10, key="man_h_upload")
            elif roi_method == "Percentage Based":
                if st.session_state.first_frame_roi is not None:
                    fh, fw = st.session_state.first_frame_roi.shape[:2]
                    st.info(f"Video dimensions: {fw}x{fh}")
                    col_a_roi, col_b_roi = st.columns(2)
                    with col_a_roi: x_pct=st.slider("X Pos (%)",0,90,20,key="px_upload"); w_pct=st.slider("Width (%)",10,min(100-x_pct,90),40,key="pw_upload") 
                    with col_b_roi: y_pct=st.slider("Y Pos (%)",0,90,30,key="py_upload"); h_pct=st.slider("Height (%)",10,min(100-y_pct,90),30,key="ph_upload") 
                    roi_x, roi_y, roi_w, roi_h = int(fw*x_pct/100), int(fh*y_pct/100), int(fw*w_pct/100), int(fh*h_pct/100)
                else: st.warning("⚠️ Upload video first to use percentage-based ROI.")
            
            st.session_state.roi_coords_manual = {"x":roi_x, "y":roi_y, "w":roi_w, "h":roi_h}

            if st.button("🔄 Reset ROI to Default", key="reset_roi_upload", use_container_width=True):
                st.session_state.roi_coords_manual = default_roi_coords.copy()
                st.rerun()

            if st.session_state.first_frame_roi is not None:
                preview_frame_roi = draw_roi_preview(st.session_state.first_frame_roi, tuple(st.session_state.roi_coords_manual.values()), False)
                display_opencv_image(preview_frame_roi, "ROI Preview on First Frame", use_container_width=True)
            elif uploaded_file and not st.session_state.is_processing : st.caption("⏳ Extracting frame for ROI preview...")
            elif not st.session_state.is_processing: st.caption("ℹ️ Upload a video to define and preview the ROI.")
    
    with col2:
        st.subheader("📊 Results & Output")
        video_display_placeholder = st.empty()
        st.markdown("### 📈 Metrics")
        # ... (Metrics display logic as before) ...
        m_col1, m_col2, m_col3 = st.columns(3)
        loaded_metric_ph = m_col1.empty()
        unloaded_metric_ph = m_col2.empty()
        unique_metric_ph = m_col3.empty()
        progress_placeholder = st.empty()
        status_message_placeholder = st.empty()

        if model is None:
            status_message_placeholder.error("🚨 Model not loaded! Processing is disabled.")
            video_display_placeholder.warning("🖼️ Video processing requires a loaded model.")
        elif not uploaded_file and not st.session_state.is_processing:
            status_message_placeholder.info("📁 Please upload a video file to begin processing.")
            video_display_placeholder.info("🖼️ Video output will appear here after processing.")


        if st.session_state.is_processing:
            status_message_placeholder.info(st.session_state.processing_status_message)
        elif st.session_state.processing_status_message == "✅ Video processing completed!":
            # ... (post-processing display logic as before) ...
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
                            thumb_roi_coords = tuple(st.session_state.roi_coords_manual.values())
                            thumb_with_roi = draw_roi_preview(frame_thumb, thumb_roi_coords, show_transparent=True) 
                            video_display_placeholder.image(cv2_to_pil(thumb_with_roi), caption="Processed Video (first frame with ROI border)", use_container_width=True)
                        cap_thumb.release()
                except Exception as e_thumb: st.warning(f"Thumbnail error: {e_thumb}")
        else: 
            loaded_metric_ph.metric("Total Loaded", st.session_state.final_loaded_count) 
            unloaded_metric_ph.metric("Total Unloaded", st.session_state.final_unloaded_count)
            unique_metric_ph.metric("Total Unique Items", st.session_state.final_unique_count)
            if st.session_state.processing_status_message and model and uploaded_file: # Only show status if relevant
                 status_message_placeholder.info(st.session_state.processing_status_message)


        process_btn_disabled = not (uploaded_file and model and not st.session_state.is_processing)
        if st.button(f"🚀 Process Video for {st.session_state.selected_class}s", use_container_width=True, key="process_video_btn", disabled=process_btn_disabled):
            # ... (processing logic as before) ...
            st.session_state.is_processing = True
            st.session_state.processing_status_message = f"⏳ Processing '{st.session_state.uploaded_file_name}'..."
            st.session_state.processed_video_path = None; st.session_state.final_loaded_count=0; st.session_state.final_unloaded_count=0; st.session_state.final_unique_count=0
            video_display_placeholder.empty(); status_message_placeholder.info(st.session_state.processing_status_message)
            video_bytes = uploaded_file.getvalue(); transfer_zone = tuple(st.session_state.roi_coords_manual.values())
            show_roi_toggle = st.session_state.show_roi_on_video_toggle_upload
            progress_bar = progress_placeholder.progress(0); final_res_gen = None
            for frame_data in process_video_streamlit(video_bytes,st.session_state.selected_class,transfer_zone,conf_threshold_st,show_roi_toggle):
                annot_fr,load,unload,uniq,prog,tmp_vid_path = frame_data; final_res_gen = frame_data
                if annot_fr is not None: video_display_placeholder.image(cv2_to_pil(annot_fr), caption="Processing...", use_container_width=True)
                if load is not None: loaded_metric_ph.metric("Loaded",load); unloaded_metric_ph.metric("Unloaded",unload); unique_metric_ph.metric("Unique Items",uniq)
                if prog is not None: progress_bar.progress(prog)
            st.session_state.is_processing = False
            if final_res_gen:
                _,fin_load,fin_unload,fin_uniq,_,fin_path = final_res_gen
                st.session_state.final_loaded_count=fin_load if fin_load is not None else 0
                st.session_state.final_unloaded_count=fin_unload if fin_unload is not None else 0
                st.session_state.final_unique_count=fin_uniq if fin_uniq is not None else 0
                st.session_state.processing_status_message = "✅ Video processing completed!"
                if fin_path and os.path.exists(fin_path): st.session_state.processed_video_path = fin_path
                else:
                    st.session_state.processed_video_path = None
                    if fin_path is None and (final_res_gen[0] is None and final_res_gen[1] is None):
                        st.session_state.processing_status_message = "⚠️ Processing error or video save failed."
            else: st.session_state.processing_status_message = "❓ Processing finished with no results."
            progress_placeholder.empty(); st.rerun()


        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path) and not st.session_state.is_processing:
            # ... (download button logic as before) ...
            try:
                with open(st.session_state.processed_video_path, "rb") as fp_dl:
                    dl_fname = f"processed_{st.session_state.uploaded_file_name or 'video.mp4'}"
                    st.download_button("💾 Download Processed Video", data=fp_dl, file_name=dl_fname, mime="video/mp4", key="dl_btn")
            except Exception as e_dl: st.error(f"Download error: {e_dl}")


# ------------------------------ LIVE STREAM MODE ------------------------------
else:
    st.subheader("📹 Live Stream / Webcam Processing")
    col1_live, col2_live = st.columns([1, 1])
    with col1_live:
        st.subheader("⚙️ Live Stream Configuration")
        if model is None: st.error("🚨 Model not loaded! Live stream functionality is disabled.")
        else:
            st.session_state.live_stream_input_type = st.radio(
                "Select Live Input Source:", ("Webcam", "RTSP Stream"),
                index=("Webcam", "RTSP Stream").index(st.session_state.live_stream_input_type),
                key="live_input_type_radio_main"
            )

            if st.session_state.live_stream_state == "initial":
                st.info("ℹ️ Setup input source and ROI, then start the stream.")
                capture_source_display = ""; capture_source_internal = None
                if st.session_state.live_stream_input_type == "Webcam":
                    cam_idx = st.number_input("Select Camera Index:", value=st.session_state.live_camera_idx, min_value=0, max_value=10, key="live_cam_idx_initial")
                    st.session_state.live_camera_idx = cam_idx
                    capture_source_display = f"Webcam (Index: {cam_idx})"; capture_source_internal = cam_idx
                else: # RTSP Stream
                    rtsp_url_input = st.text_input("Enter RTSP Stream URL:", value=st.session_state.rtsp_url, placeholder="e.g., rtsp://user:pass@ip:port/path", key="rtsp_url_input_main")
                    st.session_state.rtsp_url = rtsp_url_input
                    capture_source_display = f"RTSP ({rtsp_url_input[:30]}...)" if rtsp_url_input and rtsp_url_input != "rtsp://" else "RTSP (No URL provided)"
                    capture_source_internal = rtsp_url_input if rtsp_url_input and rtsp_url_input != "rtsp://" else None
                
                capture_btn_disabled = (st.session_state.live_stream_input_type == "RTSP Stream" and not capture_source_internal)
                if st.button(f"📸 Capture Frame for ROI (from {capture_source_display})", key="capture_frame_btn_live", use_container_width=True, disabled=capture_btn_disabled):
                    # ... (capture logic as before) ...
                    if capture_source_internal is None and st.session_state.live_stream_input_type == "RTSP Stream":
                        st.error("⚠️ Please enter a valid RTSP URL.")
                    else:
                        with st.spinner(f"Accessing {capture_source_display}..."):
                            cap_roi = cv2.VideoCapture(capture_source_internal)
                            if cap_roi.isOpened():
                                ret_r, frame_r = cap_roi.read(); cap_roi.release()
                                if ret_r and frame_r is not None:
                                    st.session_state.live_stream_captured_frame = frame_r
                                    st.session_state.live_stream_state = "roi_setup"; st.rerun()
                                else: st.error(f"⚠️ Failed to capture from {capture_source_display}.")
                            else: st.error(f"⚠️ Could not open {capture_source_display}.")


            elif st.session_state.live_stream_state == "roi_setup":
                st.info(f"🎯 Adjust ROI for the live stream from {st.session_state.live_stream_input_type}.")
                if st.session_state.live_stream_captured_frame is not None:
                    # ... (ROI adjustment sliders as before) ...
                    frame_roi = st.session_state.live_stream_captured_frame; hf, wf = frame_roi.shape[:2]
                    cur_roi_live = st.session_state.live_stream_roi_coords
                    rx = st.slider("ROI X",0,max(0,wf-50),cur_roi_live[0],key="lrx_main",help=f"Frame Width: {wf}")
                    ry = st.slider("ROI Y",0,max(0,hf-50),cur_roi_live[1],key="lry_main",help=f"Frame Height: {hf}")
                    rw = st.slider("ROI Width",50,max(50,wf-rx),cur_roi_live[2],key="lrw_main")
                    rh = st.slider("ROI Height",50,max(50,hf-ry),cur_roi_live[3],key="lrh_main")
                    st.session_state.live_stream_roi_coords = (rx,ry,rw,rh)

                    if st.button("🔄 Reset ROI to Default", key="reset_roi_live", use_container_width=True):
                        st.session_state.live_stream_roi_coords = default_live_roi_coords
                        st.rerun()
                    
                    sel_cls_live_idx = AVAILABLE_CLASSES.index(st.session_state.get('live_selected_class', AVAILABLE_CLASSES[0])) if st.session_state.get('live_selected_class') in AVAILABLE_CLASSES else 0
                    st.session_state.live_selected_class = st.selectbox("Select item type to track:",AVAILABLE_CLASSES,index=sel_cls_live_idx,key="live_class_select_main")
                    st.session_state.live_conf_thresh = st.slider("Detection Confidence:",0.1,1.0,st.session_state.get('live_conf_thresh',0.55),0.05,key="live_conf_main")
                    st.toggle("Show ROI on Live Stream", key="show_roi_on_live_stream_toggle", value=st.session_state.show_roi_on_live_stream_toggle, help="If ON, the transfer zone will be drawn on the live feed.")

                    start_live_disabled = (st.session_state.live_stream_input_type == "RTSP Stream" and (not st.session_state.rtsp_url or st.session_state.rtsp_url == "rtsp://"))
                    if st.button("🚀 Start Live Detection",key="start_live_detection_btn",use_container_width=True, disabled=start_live_disabled):
                        # ... (start detection logic as before) ...
                        if start_live_disabled: st.error("⚠️ RTSP URL is required to start detection.")
                        else: st.session_state.live_stream_active = True; st.session_state.live_stream_state = "streaming"; st.rerun()
                    if st.button("📸 Recapture Frame",key="recapture_frame_btn_live",use_container_width=True):
                        st.session_state.live_stream_captured_frame=None; st.session_state.live_stream_state="initial"; st.rerun()
                else: 
                    st.error("⚠️ No captured frame available for ROI setup. Please recapture."); st.session_state.live_stream_state="initial"; st.rerun()
            
            elif st.session_state.live_stream_state == "streaming":
                st.success(f"🟢 Live Stream Active from {st.session_state.live_stream_input_type}")
                st.markdown(f"**Tracking:** `{st.session_state.get('live_selected_class','N/A')}`")
                st.markdown(f"**Confidence:** `{st.session_state.get('live_conf_thresh','N/A')}`")
                st.markdown(f"**ROI (x,y,w,h):** `{st.session_state.get('live_stream_roi_coords')}`")
                st.toggle("Show ROI on Live Stream", key="show_roi_on_live_stream_toggle", value=st.session_state.show_roi_on_live_stream_toggle, help="Toggle ROI visibility on the live feed.")
                if st.button("⏹️ Stop Live Stream",key="stop_live_stream_btn_active",use_container_width=True):
                    st.session_state.live_stream_active=False; st.session_state.live_stream_state="initial"; time.sleep(0.1); st.rerun()
    
    with col2_live:
        st.subheader("📺 Live Output")
        if st.session_state.live_stream_state == "streaming" and st.session_state.live_stream_active:
            st.markdown("<p style='text-align: center; font-weight: bold; color: red;'>🔴 LIVE</p>", unsafe_allow_html=True)
        # ... (Live output display logic as before) ...
        if st.session_state.live_stream_state == "roi_setup" and st.session_state.live_stream_captured_frame is not None:
            preview_live_roi = draw_roi_preview(st.session_state.live_stream_captured_frame, st.session_state.live_stream_roi_coords, False)
            display_opencv_image(preview_live_roi, "ROI Preview for Live Stream", use_container_width=True)
        elif st.session_state.live_stream_state == "streaming" and st.session_state.live_stream_active:
            if model: 
                current_live_source = st.session_state.live_camera_idx if st.session_state.live_stream_input_type == "Webcam" else st.session_state.rtsp_url
                if current_live_source is not None and (st.session_state.live_stream_input_type == "Webcam" or (st.session_state.live_stream_input_type == "RTSP Stream" and current_live_source != "rtsp://")):
                    live_stream_processing_loop(st.session_state.live_selected_class, st.session_state.live_stream_roi_coords, st.session_state.live_conf_thresh, current_live_source)
                else:
                    st.error(f"⚠️ Invalid source for {st.session_state.live_stream_input_type}. Please check settings."); st.session_state.live_stream_active=False; st.session_state.live_stream_state="initial"
            else: st.error("🚨 Model error. Cannot start stream."); st.session_state.live_stream_active=False; st.session_state.live_stream_state="initial"
        elif st.session_state.live_stream_state == "initial": st.info("ℹ️ Camera feed and metrics will appear here once the stream starts.")
        else: st.info("⚙️ Configure settings and start the stream to see output.")


# --- Footer (no changes needed) ---
footer = """<style>.footer{position:fixed;left:0;bottom:0;width:100%;background-color:#0E1117;color:#FAFAFA;text-align:center;padding:10px 0;font-size:12px;border-top:1px solid #31333F;z-index:100;}</style><div class="footer"><p>© 2024 ElevateTrust.AI. All rights reserved.</p></div>"""
st.markdown(footer, unsafe_allow_html=True)

# --- Sidebar Utilities ---
st.sidebar.markdown("---"); st.sidebar.header("🔧 Additional Utilities") 
# ... (Camera/RTSP Test logic as before) ...
if st.sidebar.button("📷 Camera/RTSP Test", key="sidebar_test_conn"):
    st.sidebar.info("🧪 Testing connection...")
    cap_t = None; test_source_display = ""; test_source_internal = None
    input_type_for_test = st.session_state.get('live_stream_input_type', 'Webcam')
    if input_type_for_test == "Webcam":
        cam_idx_t = st.session_state.get('live_camera_idx', 0)
        test_source_internal = cam_idx_t; test_source_display = f"Webcam (Index: {cam_idx_t})"
    else: 
        rtsp_t = st.session_state.get('rtsp_url', '')
        if not rtsp_t or rtsp_t == "rtsp://": st.sidebar.warning("⚠️ No RTSP URL entered to test.")
        else: test_source_internal = rtsp_t; test_source_display = f"RTSP ({rtsp_t[:30]}...)"
    if test_source_internal is not None:
        try:
            st.sidebar.info(f"Attempting to connect to {test_source_display}...")
            cap_t = cv2.VideoCapture(test_source_internal)
            if cap_t.isOpened():
                ret_test, frame_test = cap_t.read() 
                if ret_test and frame_test is not None: st.sidebar.success(f"✅ Connection to {test_source_display} OK! Frame captured.")
                else: st.sidebar.error(f"❌ Connected to {test_source_display}, but failed to read frame.")
            else: st.sidebar.error(f"❌ Failed to open {test_source_display}.")
        except Exception as e_ct: st.sidebar.error(f"❌ Test failed for {test_source_display}: {e_ct}")
        finally:
            if cap_t and cap_t.isOpened(): cap_t.release()

with st.sidebar.expander("❓ Help & Tips", expanded=False):
    st.markdown("""
    **General:**
    - Ensure `best_bag_box.pt` model is in the app directory.
    - Adjust "Detection Confidence" for optimal results.

    **Upload Video Mode:**
    - Use "⚙️ General Settings" for class, confidence, and ROI display toggle.
    - Use "🎯 Define Transfer Zone (ROI)" to set the counting area.
    - "Show ROI on Processed Video" toggle applies to the output video frames.

    **Live Stream/Webcam Mode:**
    - Select "Webcam" or "RTSP Stream" as your input.
    - For Webcam, select the correct "Camera Index".
    - For RTSP, enter the full URL (e.g., `rtsp://user:pass@ip:port/path`).
    - "Capture Frame for ROI" to set the counting zone on a snapshot.
    - "Show ROI on Live Stream" toggle applies to the live feed.
    - Use "📷 Camera/RTSP Test" to verify your source connection.
    """)
if st.sidebar.checkbox("🐛 Debug Mode", key="debug_mode_sidebar"):
    # ... (Debug mode display logic as before) ...
    st.sidebar.markdown("### Session State Debug Information")
    debug_data_filtered = {}
    for k, v in st.session_state.items():
        if isinstance(v, np.ndarray) and v.ndim >=2: debug_data_filtered[k] = f"Numpy array (shape:{v.shape},dtype:{v.dtype})"
        elif isinstance(v, bytes) and len(v) > 1024: debug_data_filtered[k] = f"Bytes (len:{len(v)}) - Omitted"
        elif isinstance(v,(list,dict)) and len(str(v)) > 1024: debug_data_filtered[k] = f"{type(v).__name__} (len:{len(v)}) - Too large"
        else: debug_data_filtered[k] = v
    st.sidebar.json(debug_data_filtered, expanded=False)