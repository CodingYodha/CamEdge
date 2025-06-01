import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
from PIL import Image
import time

st.set_page_config(layout="wide", page_title="Enhanced Item Tracking App")

# --- Helper Functions ---
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

def draw_roi_preview(frame, roi_coords, show_transparent_for_thumbnail=False, show_semi_transparent_for_processing=False):
    frame_with_roi = frame.copy()
    tx, ty, tw, th = roi_coords
    if tx < 0 or ty < 0 or tw <= 0 or th <= 0: return frame_with_roi

    if show_semi_transparent_for_processing:
        overlay = frame_with_roi.copy()
        roi_color = (255, 0, 255)
        cv2.rectangle(overlay, (tx, ty), (tx + tw, ty + th), roi_color, -1)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame_with_roi, 1 - alpha, 0, frame_with_roi)
        cv2.rectangle(frame_with_roi, (tx, ty), (tx + tw, ty + th), roi_color, 1)
    elif show_transparent_for_thumbnail:
         cv2.rectangle(frame_with_roi, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 1)
    else:
        overlay = frame_with_roi.copy()
        cv2.rectangle(overlay, (tx, ty), (tx+tw, ty+th), (255, 0, 255), -1)
        alpha_setup = 0.3
        cv2.addWeighted(overlay, alpha_setup, frame_with_roi, 1-alpha_setup, 0, frame_with_roi)
        cv2.rectangle(frame_with_roi, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 2)
        cv2.putText(frame_with_roi, "Transfer Zone", (tx, ty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255),2)
    return frame_with_roi

MODEL_PATH = 'best_bag_box.pt'

@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        # Class remapping can be done here if needed, example:
        # original_name = "old_class_name"
        # new_name = "new_class_name"
        # if hasattr(model, 'model') and hasattr(model.model, 'names'):
        #     for idx, name_in_model in model.model.names.items():
        #         if name_in_model == original_name:
        #             model.model.names[idx] = new_name
        #             st.info(f"Remapped '{original_name}' to '{new_name}'")
        #             break
        return model
    except FileNotFoundError: st.error(f"Model not found: {model_path}. Ensure '{os.path.basename(model_path)}' is present."); return None
    except Exception as e: st.error(f"Error loading model: {e}"); return None

model = load_yolo_model(MODEL_PATH)
AVAILABLE_CLASSES = list(model.model.names.values()) if model and hasattr(model, 'model') and hasattr(model.model, 'names') else ['bag', 'box']
if not (model and hasattr(model, 'model') and hasattr(model.model, 'names')):
    st.warning("Using fallback classes. Issue with model loading or class names.")

def get_first_frame(uploaded_file_or_path):
    cap = None; temp_path = None
    try:
        if hasattr(uploaded_file_or_path, 'read'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_or_path.name)[1]) as tfile:
                tfile.write(uploaded_file_or_path.read()); temp_path = tfile.name
            video_source = temp_path
        else: video_source = uploaded_file_or_path
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened(): st.error(f"Could not open video: {video_source}"); return None
        ret, frame = cap.read()
        if ret and frame is not None: return frame
        else: st.error("Could not read first frame from video."); return None
    except Exception as e: st.error(f"Error extracting first frame: {e}"); return None
    finally:
        if cap: cap.release()
        if temp_path and os.path.exists(temp_path): try: os.unlink(temp_path)
        except Exception: pass

def process_single_frame(frame, model_instance, selected_class, transfer_zone_coords, conf_threshold,
                        item_id_map, next_display_item_id, item_zone_tracking_info,
                        loaded_count, unloaded_count, show_roi_overlay_during_processing=False):
    annotated_frame = frame.copy()
    tx, ty, tw, th = transfer_zone_coords
    TRANSFER_ZONE_COORDS_x1y1x2y2 = (tx, ty, tx + tw, ty + th)

    if show_roi_overlay_during_processing:
        annotated_frame = draw_roi_preview(annotated_frame, transfer_zone_coords, show_semi_transparent_for_processing=True)

    class_index = None
    if hasattr(model_instance, 'model') and hasattr(model_instance.model, 'names') and model_instance.model.names:
        class_names_list = list(model_instance.model.names.values())
        if selected_class in class_names_list: class_index = class_names_list.index(selected_class)
    current_item_count_frame = 0
    try:
        track_args = {'conf': conf_threshold, 'persist': True, 'tracker': "bytetrack.yaml", 'verbose': False}
        if class_index is not None: track_args['classes'] = [class_index]
        results = model_instance.track(frame, **track_args) # Process original frame
    except Exception:
        return annotated_frame, loaded_count[0], unloaded_count[0], next_display_item_id[0], current_item_count_frame

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes_data = results[0].boxes.xyxy.cpu().numpy()
        confs_data = results[0].boxes.conf.cpu().numpy()
        original_tracker_ids_data = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None

        for i in range(len(boxes_data)):
            x1,y1,x2,y2 = map(int, boxes_data[i]); current_item_count_frame+=1
            cls_id_tensor = results[0].boxes.cls[i] if results[0].boxes.cls is not None else None
            cls_id = int(cls_id_tensor.cpu()) if cls_id_tensor is not None else None
            item_cls_name = selected_class
            if cls_id is not None and hasattr(model_instance,'model') and hasattr(model_instance.model,'names') and cls_id in model_instance.model.names:
                 item_cls_name = model_instance.model.names[cls_id]
            lbl_base = f"{item_cls_name} {confs_data[i]:.2f}"
            cx,cy = (x1+x2)//2, (y1+y2)//2
            disp_id = None
            if original_tracker_ids_data is not None and i < len(original_tracker_ids_data):
                orig_id = original_tracker_ids_data[i]
                if orig_id not in item_id_map: item_id_map[orig_id]=next_display_item_id[0]; next_display_item_id[0]+=1
                disp_id = item_id_map[orig_id]; label = f"ID:{disp_id} {lbl_base}"
                in_zone = is_centroid_in_zone(cx,cy,TRANSFER_ZONE_COORDS_x1y1x2y2)
                if disp_id not in item_zone_tracking_info: item_zone_tracking_info[disp_id]={"was_in_zone":in_zone,"load_counted":False,"unload_counted":False}
                info=item_zone_tracking_info[disp_id]; was_in=info["was_in_zone"]
                if not was_in and in_zone and not info["load_counted"]: loaded_count[0]+=1; info["load_counted"]=True; info["unload_counted"]=False
                elif was_in and not in_zone and not info["unload_counted"]: unloaded_count[0]+=1; info["unload_counted"]=True; info["load_counted"]=False
                info["was_in_zone"]=in_zone
            else: label=lbl_base
            item_col=(0,255,0) if item_cls_name.lower()=='box' else (255,165,0)
            cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),item_col,2)
            cv2.putText(annotated_frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,item_col,2)
            if disp_id: cv2.circle(annotated_frame,(cx,cy),5,item_col,-1)

    y_off, fnt, f_sc, thk, bg, pd = 30,cv2.FONT_HERSHEY_SIMPLEX,0.8,2,(50,50,50),5
    draw_text_with_background(annotated_frame,f"Loaded: {loaded_count[0]}",(10,y_off),fnt,f_sc,(0,255,0),bg,thk,pd)
    y_off+=40; draw_text_with_background(annotated_frame,f"Unloaded: {unloaded_count[0]}",(10,y_off),fnt,f_sc,(0,0,255),bg,thk,pd)
    y_off+=40; draw_text_with_background(annotated_frame,f"Items: {current_item_count_frame}",(10,y_off),fnt,0.7,(0,255,255),bg,thk,pd)
    return annotated_frame, loaded_count[0], unloaded_count[0], next_display_item_id[0], current_item_count_frame

def process_video_streamlit(video_bytes, selected_class, transfer_zone_rect, conf_threshold, show_roi_overlay):
    if model is None: yield None,0,0,0,None,None; return
    temp_in,temp_out,writer,cap=None,None,None,None
    lc,ulc,nid=[0],[0],[1]; imap,izinfo={},{}
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4') as t_in: t_in.write(video_bytes);temp_in=t_in.name
        cap=cv2.VideoCapture(temp_in)
        if not cap.isOpened(): yield None,0,0,0,None,None; return
        fps,w,h,tf=max(cap.get(cv2.CAP_PROP_FPS),1),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4') as t_out: temp_out=t_out.name
        writer=cv2.VideoWriter(temp_out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not writer.isOpened(): yield None,0,0,0,None,temp_out; return
        fc=0
        while True:
            ret,frm=cap.read()
            if not ret or frm is None: break
            fc+=1
            ann_f,cl,cul,cnid,ifr=process_single_frame(frm,model,selected_class,transfer_zone_rect,conf_threshold,imap,nid,izinfo,lc,ulc,show_roi_overlay_during_processing=show_roi_overlay)
            if ann_f is None: # Handle if a frame processing fails critically
                st.warning(f"Warning: Frame {fc} could not be processed.")
                if writer and frm is not None: writer.write(frm) # Write original frame if annotation failed
                yield frm, lc[0], ulc[0], (nid[0]-1 if nid[0]>1 else 0), fc/tf if tf>0 else 0, None
                continue

            if writer:writer.write(ann_f)
            tu=cnid-1 if cnid>1 else 0; prg=fc/tf if tf>0 else 0
            yield ann_f,cl,cul,tu,prg,None
    except Exception as e: st.error(f"Video processing error: {e}")
    finally:
        if cap:cap.release()
        if writer:writer.release()
        if temp_in and os.path.exists(temp_in): try:os.unlink(temp_in)
        except Exception:pass
        fvp=None
        if temp_out and os.path.exists(temp_out) and os.path.getsize(temp_out)>0: fvp=temp_out
        else:
            if temp_out and os.path.exists(temp_out): try:os.unlink(temp_out)
            except Exception:pass
    final_l,final_ul,final_u=lc[0],ulc[0],(nid[0]-1 if nid[0]>1 else 0)
    yield None,final_l,final_ul,final_u,1.0,fvp

def live_stream_processing_loop(selected_class,transfer_zone_rect,conf_threshold,camera_index,show_roi_overlay):
    if not model:st.error("Model not loaded.");st.session_state.live_stream_active=False;return
    cap=cv2.VideoCapture(camera_index)
    if not cap.isOpened():st.error(f"Cam {camera_index} open fail.");st.session_state.live_stream_active=False;return
    im,ni,iz,lc,ulc={},[1],{},[0],[0]
    sph=st.empty();mc1,mc2,mc3=st.columns(3);uph,lph,ulph=mc1.empty(),mc2.empty(),mc3.empty()
    try:
        while st.session_state.get('live_stream_active',False):
            ret,frm=cap.read()
            if not ret or frm is None:st.warning("Cam read fail.");st.session_state.live_stream_active=False;break
            anf,ctl,ctul,cnid,_=process_single_frame(frm,model,selected_class,transfer_zone_rect,conf_threshold,im,ni,iz,lc,ulc,show_roi_overlay_during_processing=show_roi_overlay)
            sph.image(cv2_to_pil(anf),channels="RGB",use_container_width=True)
            tu=cnid-1 if cnid>1 else 0
            uph.metric("Unique",tu);lph.metric("Loaded",ctl);ulph.metric("Unloaded",ctul)
    except Exception as e:st.error(f"Live stream err: {e}")
    finally:
        if cap.isOpened():cap.release()
        sph.empty();uph.empty();lph.empty();ulph.empty()
        if st.session_state.get('live_stream_active',False):st.info("Stream stopped.")
        st.session_state.live_stream_active=False

try:st.image("logo.png",width=150)
except:st.markdown("### 📦 CamEdge App")
st.title("CamEdge")

default_states={'roi_coords_manual':{"x":100,"y":100,"w":300,"h":200},'first_frame_roi':None,'selected_class':AVAILABLE_CLASSES[0] if AVAILABLE_CLASSES else 'bag','uploaded_file_name':None,'processing_mode':"Upload Video",'processed_video_path':None,'live_stream_state':"initial",'live_stream_captured_frame':None,'live_stream_roi_coords':(100,100,200,150),'live_stream_active':False,'final_loaded_count':0,'final_unloaded_count':0,'final_unique_count':0,'processing_status_message':"",'is_processing':False,'show_roi_during_processing':False,'show_roi_during_live_processing':False}
for k,v in default_states.items():
    if k not in st.session_state:st.session_state[k]=v

st.sidebar.header("📹 Processing Mode")
pms=["Upload Video","Live Stream/Webcam"];cmi=pms.index(st.session_state.processing_mode);npm=st.sidebar.radio("Mode:",pms,index=cmi,key="pmr")
if st.session_state.processing_mode!=npm:
    st.session_state.processing_mode=npm
    if npm=="Upload Video":st.session_state.live_stream_state,st.session_state.live_stream_active,st.session_state.live_stream_captured_frame="initial",False,None
    else:st.session_state.first_frame_roi,st.session_state.processed_video_path,st.session_state.processing_status_message,st.session_state.is_processing=None,None,"",False
    st.rerun()

if st.session_state.processing_mode=="Upload Video":
    upf=st.file_uploader("Upload video",type=["mp4","avi","mov","mkv"])
    if upf:
        if st.session_state.uploaded_file_name!=upf.name:
            st.session_state.first_frame_roi,st.session_state.uploaded_file_name=None,upf.name
            st.session_state.processed_video_path,st.session_state.processing_status_message=None,"" # Reset path
            st.session_state.final_loaded_count,st.session_state.final_unloaded_count,st.session_state.final_unique_count=0,0,0
            st.session_state.is_processing=False;st.rerun()
    else:st.session_state.uploaded_file_name=None

    cl1,cl2=st.columns([1,1])
    with cl1:
        st.subheader("⚙️ Settings")
        if model:scidx=AVAILABLE_CLASSES.index(st.session_state.selected_class) if st.session_state.selected_class in AVAILABLE_CLASSES else 0;st.session_state.selected_class=st.selectbox("Track item:",AVAILABLE_CLASSES,index=scidx)
        else:st.error("Model not loaded.");st.selectbox("Track item:",AVAILABLE_CLASSES,disabled=True)
        cfth_upload=st.slider("Confidence:",0.1,1.0,0.55,0.05,key="cfupl")
        st.session_state.show_roi_during_processing=st.toggle("Show ROI in process",value=st.session_state.show_roi_during_processing,key="roitglupl",help="Overlay semi-transparent ROI during video processing.")
        st.markdown("---");st.subheader("🎯 Define ROI")
        if upf and st.session_state.first_frame_roi is None and not st.session_state.is_processing:
            with st.spinner("Extracting..."):st.session_state.first_frame_roi=get_first_frame(upf)
            if st.session_state.first_frame_roi is None:st.error("Frame extract fail.")
            else:st.rerun()

        rms=["Manual Input","Percentage Based"];rmi=st.session_state.get('rmpidx',0) if st.session_state.first_frame_roi is not None else 0
        rm=st.radio("ROI Method:",rms,index=rmi,key="rmupl")
        if st.session_state.first_frame_roi is not None:st.session_state.rmpidx=rms.index(rm)

        rx,ry,rw,rh=st.session_state.roi_coords_manual.values()
        if rm=="Manual Input":rx=st.number_input("X",rx,0,key="mx");ry=st.number_input("Y",ry,0,key="my");rw=st.number_input("W",rw,10,key="mw");rh=st.number_input("H",rh,10,key="mh")
        elif rm=="Percentage Based" and st.session_state.first_frame_roi is not None:
            fh,fw=st.session_state.first_frame_roi.shape[:2];st.info(f"Dims:{fw}x{fh}")
            ca,cb=st.columns(2)
            with ca:xp=st.slider("X%",0,90,20,key="pcx");wp=st.slider("W%",10,min(100-xp,90),40,key="pcw")
            with cb:yp=st.slider("Y%",0,90,30,key="pcy");hp=st.slider("H%",10,min(100-yp,90),30,key="pch")
            rx,ry,rw,rh=int(fw*xp/100),int(fh*yp/100),int(fw*wp/100),int(fh*hp/100)
        elif rm=="Percentage Based":st.warning("Upload video for % ROI.")
        st.session_state.roi_coords_manual={"x":rx,"y":ry,"w":rw,"h":rh}
        if st.session_state.first_frame_roi is not None:
            pimg=draw_roi_preview(st.session_state.first_frame_roi,(rx,ry,rw,rh));display_opencv_image(pimg,"ROI Preview",True)
        elif upf and not st.session_state.is_processing:st.caption("Extracting...")
        elif not st.session_state.is_processing:st.caption("Upload video for ROI.")

    with cl2:
        st.subheader("📊 Results")
        vdph=st.empty();st.markdown("### 📈 Metrics");mc1,mc2,mc3=st.columns(3);lph,ulph,uqph=mc1.empty(),mc2.empty(),mc3.empty();prph,stmph=st.empty(),st.empty()
        if st.session_state.is_processing:stmph.info(st.session_state.processing_status_message)
        elif st.session_state.processing_status_message=="✅ Video processing completed!":
            stmph.success(st.session_state.processing_status_message);lph.metric("Loaded",st.session_state.final_loaded_count);ulph.metric("Unloaded",st.session_state.final_unloaded_count);uqph.metric("Unique",st.session_state.final_unique_count)
            if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                try:
                    cth=cv2.VideoCapture(st.session_state.processed_video_path)
                    if cth.isOpened():
                        ret_th,fth=cth.read()
                        if ret_th:thrc=tuple(st.session_state.roi_coords_manual.values());thimg=draw_roi_preview(fth,thrc,show_transparent_for_thumbnail=True);vdph.image(cv2_to_pil(thimg),"Processed (ROI border)",True)
                        cth.release()
                except Exception as eth:st.warning(f"Thumb err:{eth}")
            else:vdph.empty()
        else:
            lph.metric("Loaded",st.session_state.final_loaded_count);ulph.metric("Unloaded",st.session_state.final_unloaded_count);uqph.metric("Unique",st.session_state.final_unique_count)
            if st.session_state.processing_status_message:stmph.info(st.session_state.processing_status_message)
            else:stmph.empty()
            vdph.empty()

        pbd=not(upf and model and not st.session_state.is_processing)
        if st.button("🚀 Process Video",use_container_width=True,key="pvb",disabled=pbd):
            st.session_state.is_processing=True;st.session_state.processing_status_message=f"Processing '{st.session_state.uploaded_file_name}'..."
            st.session_state.processed_video_path,st.session_state.final_loaded_count,st.session_state.final_unloaded_count,st.session_state.final_unique_count=None,0,0,0
            vdph.empty();stmph.info(st.session_state.processing_status_message)
            vb=upf.getvalue();tz=tuple(st.session_state.roi_coords_manual.values());pb=prph.progress(0);frg=None;sro=st.session_state.show_roi_during_processing
            for frd in process_video_streamlit(vb,st.session_state.selected_class,tz,cfth_upload,sro):
                anf,ld,uld,unq,prg_val,tvp=frd;frg=frd # Use prg_val to avoid conflict
                if anf:vdph.image(cv2_to_pil(anf),"Processing...",True)
                if ld is not None:lph.metric("Loaded",ld);ulph.metric("Unloaded",uld);uqph.metric("Unique",unq)
                if prg_val is not None:pb.progress(prg_val)
            st.session_state.is_processing=False
            if frg:
                _,fl,ful,fu,_,fp=frg;st.session_state.final_loaded_count,st.session_state.final_unloaded_count,st.session_state.final_unique_count=fl or 0,ful or 0,fu or 0
                st.session_state.processing_status_message="✅ Video processing completed!"
                if fp and os.path.exists(fp):st.session_state.processed_video_path=fp
                else:st.session_state.processed_video_path=None;
                if fp is None and frg[0] is None:st.session_state.processing_status_message="⚠️ Proc/save error."
            else:st.session_state.processing_status_message="❓ No results."
            prph.empty();st.rerun()

        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path) and not st.session_state.is_processing:
            try:
                with open(st.session_state.processed_video_path,"rb") as fp_dl:dlfn=f"processed_{st.session_state.uploaded_file_name or 'video.mp4'}";st.download_button("💾 Download",data=fp_dl,file_name=dlfn,mime="video/mp4",key="dlbm")
            except Exception as e_dl:st.error(f"DL err:{e_dl}")
        elif not upf and not st.session_state.is_processing:st.info("📁 Upload video...")
        elif not model and not st.session_state.is_processing:st.error("Model not loaded.")

else: # Live Stream Mode
    st.subheader("📹 Live Stream")
    c1l,c2l=st.columns([1,1])
    with c1l:
        st.subheader("⚙️ Live Settings")
        if not model:st.error("Model not loaded. Live disabled.")
        else:
            if st.session_state.live_stream_state=="initial":
                st.info("Setup cam & ROI.");ci_l=st.number_input("Cam Idx",value=st.session_state.get('live_cam_idx',0),min_value=0,max_value=10,key="lici");st.session_state.live_cam_idx=ci_l # Use live_cam_idx
                if st.button("📸 Capture Frame",key="cfbl",use_container_width=True):
                    with st.spinner(f"Accessing {ci_l}..."):
                        cap_r=cv2.VideoCapture(ci_l)
                        if cap_r.isOpened():
                            ret_r,fr_r=cap_r.read();cap_r.release()
                            if ret_r and fr_r is not None:st.session_state.live_stream_captured_frame=fr_r;st.session_state.live_stream_state="roi_setup";st.rerun()
                            else:st.error(f"Capture fail cam {ci_l}.")
                        else:st.error(f"Open cam {ci_l} fail.")
            elif st.session_state.live_stream_state=="roi_setup":
                st.info("Adjust ROI.")
                if st.session_state.live_stream_captured_frame is not None:
                    frl=st.session_state.live_stream_captured_frame;hfl,wfl=frl.shape[:2];crl=st.session_state.live_stream_roi_coords
                    rxl=st.slider("X",0,max(0,wfl-50),crl[0],key="lrxl",help=f"W:{wfl}");ryl=st.slider("Y",0,max(0,hfl-50),crl[1],key="lryl",help=f"H:{hfl}")
                    rwl=st.slider("W",50,max(50,wfl-rxl),crl[2],key="lrwl");rhl=st.slider("H",50,max(50,hfl-ryl),crl[3],key="lrhl")
                    st.session_state.live_stream_roi_coords=(rxl,ryl,rwl,rhl)
                    sclli_live=AVAILABLE_CLASSES.index(st.session_state.get('live_selected_class')) if st.session_state.get('live_selected_class') in AVAILABLE_CLASSES else 0;st.session_state.live_selected_class=st.selectbox("Track:",AVAILABLE_CLASSES,index=sclli_live,key="live_cls_sel") # Use live_selected_class
                    st.session_state.live_conf_thresh=st.slider("Confidence",0.1,1.0,st.session_state.get('live_conf_thresh',0.55),0.05,key="live_conf_s") # Use live_conf_thresh
                    st.session_state.show_roi_during_live_processing=st.toggle("Show ROI in live",value=st.session_state.show_roi_during_live_processing,key="roitgllive")
                    if st.button("🚀 Start Live",key="sld",use_container_width=True):st.session_state.live_stream_active=True;st.session_state.live_stream_state="streaming";st.rerun()
                    if st.button("🔄 Recapture",key="rcb",use_container_width=True):st.session_state.live_stream_captured_frame=None;st.session_state.live_stream_state="initial";st.rerun()
                else:st.error("No frame. Recapture.");st.session_state.live_stream_state="initial";st.rerun()
            elif st.session_state.live_stream_state=="streaming":
                st.info(f"Live active. ROI {'shown' if st.session_state.show_roi_during_live_processing else 'hidden'}.")
                st.markdown(f"**Track:**`{st.session_state.get('live_selected_class','N/A')}`**Conf:**`{st.session_state.get('live_conf_thresh','N/A')}`\n**ROI:**`{st.session_state.get('live_stream_roi_coords')}`") # Use live_selected_class, live_conf_thresh
                if st.button("⏹️ Stop Live",key="stla",use_container_width=True):st.session_state.live_stream_active=False;st.session_state.live_stream_state="initial";time.sleep(0.1);st.rerun()
    with c2l:
        st.subheader("📺 Live Output")
        if st.session_state.live_stream_state=="roi_setup" and st.session_state.live_stream_captured_frame is not None:
            pli=draw_roi_preview(st.session_state.live_stream_captured_frame,st.session_state.live_stream_roi_coords);display_opencv_image(pli,"ROI Preview Live",True)
        elif st.session_state.live_stream_state=="streaming" and st.session_state.live_stream_active:
            if model:live_stream_processing_loop(st.session_state.live_selected_class,st.session_state.live_stream_roi_coords,st.session_state.live_conf_thresh,st.session_state.live_cam_idx,st.session_state.show_roi_during_live_processing) # Use live_selected_class, live_conf_thresh, live_cam_idx
            else:st.error("Model err.");st.session_state.live_stream_active,st.session_state.live_stream_state=False,"initial"
        elif st.session_state.live_stream_state=="initial":st.info("Feed & metrics here.")
        else:st.info("Configure & start.")

st.markdown("---");st.caption("CamEdge by ElevateTrust.AI")
st.sidebar.markdown("---");st.sidebar.header("🔧 Utilities")
if st.sidebar.button("📷 Cam Test"):
    st.sidebar.info("Testing...");ct=None
    try:
        cit=st.session_state.get('live_cam_idx',0);ct=cv2.VideoCapture(cit) # Use live_cam_idx
        if ct.isOpened():st.sidebar.success(f"✅ Cam({cit}) OK!")
        else:st.sidebar.error(f"❌ No cam({cit}).")
    except Exception as e_cam_test:st.sidebar.error(f"❌ Test fail:{e_cam_test}")
    finally:
        if ct and ct.isOpened():ct.release()
with st.sidebar.expander("❓ Help & Tips"):st.markdown("""**ROI:** Purple setup. Opt. semi-trans overlay (toggle). Border on final thumb.\n\n**Live:** Capture for ROI. Toggle ROI vis.\n\n**General:** Adj conf. Check `best_bag_box.pt`.""")
if st.sidebar.checkbox("🐛 Debug"):
    st.sidebar.markdown("### Debug Info");dd={k:(f"Shape:{v.shape}" if isinstance(v,np.ndarray) else v) for k,v in st.session_state.items()};st.sidebar.json(dd,False)