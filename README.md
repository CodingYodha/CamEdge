# CamEdge - Enhanced Item Tracking & Counting App

**Developed by [ElevateTrust.AI Solutions](https://www.elevatetrust.ai)** <!-- Optional: Link to your company website -->

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-camedge-streamlit-app-url.com) <!-- Replace with your actual Streamlit sharing URL if deployed -->

CamEdge is a powerful and user-friendly Streamlit application designed for real-time object detection, tracking, and counting of items within video feeds. It leverages state-of-the-art computer vision techniques to provide insights into item movement, making it suitable for various inventory management, logistics, and monitoring scenarios.

## Key Features

*   **Versatile Processing Modes:**
    *   **Video File Analysis:** Process pre-recorded videos (supports MP4, AVI, MOV, MKV formats) to analyze past events.
    *   **Live Webcam/Stream Processing:** Connect to a webcam for real-time item detection, tracking, and counting.
*   **Advanced Object Detection:**
    *   Powered by a pre-trained YOLO (You Only Look Once) model (specifically `best_bag_box.pt` for "bags" and "boxes").
    *   Includes logic for model class name remapping for flexibility with different model versions.
*   **Robust Item Tracking:**
    *   Utilizes ByteTrack for persistent and accurate tracking of detected items across video frames, assigning unique IDs.
*   **Intelligent Transfer Zone Counting:**
    *   **Define a Region of Interest (ROI) / "Transfer Zone":** Users can visually or manually define a specific area in the video frame.
    *   **Loading/Unloading Logic:**
        *   Items are counted as "Loaded" when their centroid enters the defined Transfer Zone.
        *   Items are counted as "Unloaded" when their centroid exits the Transfer Zone after having been inside.
    *   Tracks unique items, total loaded, and total unloaded.
*   **Flexible ROI Configuration:**
    *   **Manual Input:** Precisely define ROI coordinates (X, Y, Width, Height).
    *   **Percentage Based:** Conveniently set ROI dimensions and position relative to the video frame's size.
    *   **Live Stream ROI Setup:** Capture a frame from the live camera to visually define the ROI before starting detection.
*   **User-Friendly Interface:**
    *   Intuitive controls for file uploads, camera selection, and parameter adjustments (e.g., detection confidence).
    *   Real-time display of processed video frames with bounding boxes, labels, and item IDs.
    *   Live metrics for loaded, unloaded, and unique items.
*   **Output & Utilities:**
    *   **Download Processed Video:** Option to save the annotated video after processing.
    *   **Camera Test Utility:** Easily check webcam connectivity.
    *   **Debug Mode:** Provides insights into application state for troubleshooting and development.

## Target Use Cases

*   Warehouse inventory management (items moving in/out of storage areas).
*   Retail analytics (customer footfall, product interaction in specific zones).
*   Manufacturing process monitoring (parts moving along a conveyor belt).
*   Security and surveillance (object tracking in restricted areas).
*   Logistics and supply chain (tracking packages at sorting facilities).

## Technical Stack

*   **Python:** Core programming language.
*   **Streamlit:** Web application framework for the UI.
*   **OpenCV (cv2):** For image and video processing.
*   **Ultralytics YOLO:** For object detection and tracking.
*   **Pillow (PIL):** For image manipulation.
*   **NumPy:** For numerical operations.

## Model

The application currently uses a pre-trained YOLO model named `best_bag_box.pt`. This model is expected to be in the same directory as the main application script (`camedge.txt` or `app.py`). It is trained to detect "bags" and "boxes". To use a different model, you would need to:
1.  Place your new `.pt` model file in the application directory.
2.  Update the `MODEL_PATH` variable in the script.
3.  Potentially adjust the class name remapping logic or `AVAILABLE_CLASSES` if your model has different class names.

## Setup and Installation

1.  **Clone the repository (or download the script):**
    ```bash
    git clone https://github.com/your-username/CamEdge.git # Replace with your actual repo URL
    cd CamEdge
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    Make sure you have a `requirements.txt` file with the necessary packages. If not, you'll need to create one based on the imports in the script (e.g., `streamlit`, `opencv-python`, `ultralytics`, `numpy`, `Pillow`).
    ```bash
    pip install -r requirements.txt
    # Or install manually:
    # pip install streamlit opencv-python ultralytics numpy Pillow
    ```
    *Note: `ultralytics` often has specific PyTorch version dependencies. Ensure your environment is compatible or install PyTorch separately first if needed.*

4.  **Ensure the YOLO model file (`best_bag_box.pt`) is present** in the root directory of the project (or update `MODEL_PATH` in the script if it's located elsewhere).

## Running the Application

1.  Navigate to the project directory in your terminal.
2.  Run the Streamlit application:
    ```bash
    streamlit run your_script_name.py
    ```
    (If you've named your main Python file `camedge.py` or `app.py`, use that name).
3.  The application will open in your default web browser.

## Usage Guide

1.  **Select Processing Mode:**
    *   **Upload Video:** Choose this to process a video file from your computer.
    *   **Live Stream/Webcam:** Choose this to use your connected webcam.
2.  **Configure Settings (common for both modes):**
    *   **Select item type to track:** Choose between "bag" or "box" (or other classes if your model supports them).
    *   **Detection Confidence Threshold:** Adjust how confident the model needs to be to detect an item (0.1 to 1.0).
3.  **Define Transfer Zone (ROI):**
    *   **For Upload Video:**
        *   Upload a video. A preview of the first frame will appear.
        *   Choose an ROI definition method (Manual Input or Percentage Based).
        *   Adjust the parameters to position the purple ROI box on the preview.
    *   **For Live Stream/Webcam:**
        *   Select your camera index.
        *   Click "Capture Frame for ROI Setup".
        *   Adjust the ROI sliders on the captured frame.
4.  **Process:**
    *   **Upload Video:** Click "Process Video". Progress and live metrics will be displayed. After completion, a "Download Processed Video" button will appear.
    *   **Live Stream/Webcam:** After setting up the ROI, click "Start Live Detection". The live feed with detections and metrics will appear. Click "Stop Live Stream" to end.

## Troubleshooting & Tips

*   **Model Not Loaded:** Ensure `best_bag_box.pt` is in the correct directory and the `MODEL_PATH` in the script is accurate.
*   **Camera Not Working:**
    *   Use the "Camera Test" utility in the sidebar.
    *   Ensure your camera is connected and not being used by another application.
    *   Try different camera indices (0, 1, 2, ...).
*   **Poor Detection Performance:**
    *   Adjust the "Detection Confidence Threshold". Lower values detect more but might include false positives. Higher values are more accurate but might miss items.
    *   Ensure good lighting conditions for the camera.
    *   The ROI should be placed strategically where items clearly enter and exit.
*   **`use_column_width` Deprecation Warning:** This has been addressed in the code by using `use_container_width` for `st.image`. If it reappears, it might be from a Streamlit version change or another component.
*   **Memory Issues for Long Videos:** Processing very long or high-resolution videos can be memory-intensive. Consider processing videos in chunks if necessary for very large files (this functionality is not built-in).
*   **Error Recovery:** The application includes basic error handling. If an unrecoverable error occurs, you might need to restart the Streamlit app. Check the terminal for detailed error messages.

## Future Enhancements (Potential)

*   Support for more complex ROI shapes (e.g., polygons).
*   Batch processing of multiple video files.
*   Integration with databases for storing count data.
*   More advanced analytics and reporting.
*   Ability to select different pre-trained models or upload custom models directly through the UI.
*   Performance optimizations for higher FPS processing.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE.txt) - see the `LICENSE.txt` file for details (or choose an appropriate license for your project).

## Acknowledgements

*   **Streamlit Team:** For the awesome framework.
*   **Ultralytics Team:** For the YOLO models and library.
*   **OpenCV Community:** For the comprehensive computer vision library.

---

**CamEdge by ElevateTrust.AI Solutions**
*Empowering Vision with Intelligence*