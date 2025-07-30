# AI Agent Surveillance System
An advanced intelligent surveillance system that uses multiple AI models (YOLOv5 + Google Gemini) for automatic analysis of security videos.

## YouTube Link:
https://youtu.be/KTCd0Cu397o

---
## Main Features

- **🔍 Dual Operation Mode:**
  - **Real-Time Monitoring:** Continuous surveillance with automatic alerts.
  - **Specific Search:** Locating objects with particular characteristics.

- **🤖 Multi-Agent Architecture:**
  - **Dispatcher Agent:** Analyzes user intent and selects the appropriate pipeline.
  - **Guard Agent:** Describes scenes and evaluates detected situations.
  - **Head of Security Agent:** Makes final decisions and determines actions to take.

- **🚨 Specialized Detection:**
  - **Accidents:** Vehicular collisions and severity assessment.
  - **Fires:** Detection of fire and smoke.
  - **General Objects:** People, vehicles, domestic animals.

---
## Technologies Used

- **Frontend:** Streamlit
- **Object Detection:** YOLOv5 (Ultralytics)
- **Visual Analysis:** Google Gemini AI
- **Video Processing:** OpenCV
- **Image Processing:** PIL/Pillow
- **ML Framework:** PyTorch

---
## System Requirements

- Python 3.8+
- GPU recommended for better performance
- Google Gemini API Key
- At least 4GB of RAM
- Sufficient disk space for videos and generated clips

---
## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AI-agentic-surveillance
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure environment variables:**
    Create a `.env` file in the project root:
    ```env
    GEMINI_API_KEY=your_api_key_here
    VISION_MODEL=gemini-1.5-flash
    TEXT_MODEL=gemini-1.5-flash
    ```

4.  **Verify file structure:**
    ```
    AI-agentic-surveillance/
    ├── app.py
    ├── requirements.txt
    ├── .env
    ├── weights/
    │   ├── best_accident.pt
    │   ├── best_fire.pt
    │   └── best_general.pt
    ├── videos/
    ├── output/
    └── output_captures/
    ```

---
## Usage

1.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

2.  **Access the web interface:**
    - Open your browser to `http://localhost:8501`

3.  **Upload video and configure:**
    - Upload a video file (MP4, AVI, MOV).
    - Describe the task in natural language.
    - Adjust detection parameters.
    - Click "Process Video."

---
## Usage Examples

### Monitoring Mode
- *"Watch for any car accidents"*
- *"Alert me if you detect fire or smoke"*
- *"Monitor for the presence of intruders"*
- *"Let me know when animals appear"*

### Search Mode
- *"Find red cars"*
- *"Look for people in dark clothing"*
- *"Locate parked trucks"*
- *"Identify large dogs"*

---
## System Architecture
