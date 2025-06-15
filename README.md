# CamEdge - Enhanced Item Tracking & Counting App

**Developed by: Shivaprasad**
*(Developed during work at ElevateTrust.AI Solutions)*

## Project Description

CamEdge is a powerful and user-friendly Streamlit application designed for real-time object detection, tracking, and counting of items within video feeds. It leverages state-of-the-art computer vision techniques to provide insights into item movement, making it suitable for various inventory management, logistics, and monitoring scenarios.

**Key capabilities include:**

*   **Versatile Processing:** Handles both uploaded video files (MP4, AVI, MOV, MKV) and live webcam streams.
*   **Advanced Object Detection & Tracking:** Utilizes a pre-trained YOLO model (specifically `best_bag_box.pt` for "bags" and "boxes") coupled with ByteTrack for robust item tracking and unique ID assignment.
*   **Intelligent Transfer Zone Counting:** Allows users to define a Region of Interest (ROI) or "Transfer Zone" to count items as "Loaded" (entering the zone) or "Unloaded" (exiting the zone after being inside).
*   **Flexible ROI Configuration:** Offers manual input and percentage-based methods for ROI definition, including a frame capture feature for live stream ROI setup.
*   **User-Friendly Interface:** Provides intuitive controls, real-time visual feedback with detections, and live metrics for item counts.
*   **Utilities:** Includes options to download processed videos, test camera connectivity, and a debug mode for advanced insights.

To run the following appv14.py
Make sure to install all the necessary libraries mentioned in requirements.txt
then run the following code
python -m streamlit run appv14.py
then Enjoy!
CamEdge aims to simplify the process of monitoring and quantifying item flow through a defined space, offering a practical tool for various operational needs.
