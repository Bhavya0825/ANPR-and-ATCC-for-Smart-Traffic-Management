<img width="1895" height="987" alt="Screenshot 2025-11-18 213528" src="https://github.com/user-attachments/assets/99e37369-dd8b-45cd-b415-f6c92545a836" />ANPR & ATCC for Smart Traffic Management

An advanced AI-powered traffic monitoring system integrating Automatic Number Plate Recognition (ANPR) and Automatic Traffic Counting & Classification (ATCC). The system detects traffic violations, recognizes number plates, identifies accidents, counts vehicles, and generates heatmaps — all in real-time using deep learning and computer vision.

⭐ Features 🔹 ANPR (Automatic Number Plate Recognition)

Detects license plates

Extracts plate text via OCR

Supports multiple plate formats

🔹 ATCC (Traffic Counting & Classification)

Counts vehicles in real-time

Classifies cars, bikes, trucks, buses, etc.

Works on live and recorded videos

🔹 Traffic Violation Detection

Helmet detection

Triple riding detection

Wrong lane or rule violation detection

🔹 Accident Detection

Identifies collision events

Generates instant alerts

🔹 Heatmap Visualization

Tracks vehicle movement

Generates traffic density heatmaps

🔹 Flask Web Dashboard

Upload and process videos

View logs, detections, heatmaps

Simple and interactive UI

🧠 Tech Stack

Python 3.9 (Recommended for best compatibility)

OpenCV

YOLOv8

Tesseract OCR / Custom OCR

Flask

NumPy, Pandas, Matplotlib

MySQL (optional for logging)

Development and testing were done on Python 3.9, so using the same version is strongly recommended.

📁 Project Structure ANPR-and-ATCC-For-Smart-Traffic-Management │── app.py # Flask application │── anpr_video.py # ANPR detection script │── accident.py # Accident detection module │── triple_riding.py # Triple riding module │── traffic_violation.py # Violation detection │── atcc.py # Traffic counting & classification │── heatmap_visualization.py │── utils/ # Utility functions │── templates/ # HTML templates for Flask │── static/ # CSS, JS, Images │── uploads/ # Uploaded media │── best/ # YOLO model files │── requirements.txt # Dependencies └── ...

🛠 Installation & Setup ✔ Recommended Python Version

Use Python 3.9 for maximum compatibility and error-free execution.

1️⃣ Clone the repository https://github.com/Bhavya0825/ANPR-and-ATCC-for-Smart-Traffic-Management cd ANPR-and-ATCC-For-Smart-Traffic-Management

2️⃣ Create a virtual environment (optional but recommended) macOS / Linux: python3.9 -m venv venv source venv/bin/activate

Windows: python3.9 -m venv venv venv\Scripts\activate

3️⃣ Install dependencies pip install -r requirements.txt

4️⃣ Install Tesseract OCR macOS: brew install tesseract

Windows:

Download the EXE installer and install normally. (Ensure the path is added to system environment variables)

▶️ Usage Start the Flask Web App python app.py

Open in browser:

http://127.0.0.1:5000/

Run ANPR only python anpr_video.py

Run Traffic Counter python atcc.py

📊 Outputs

Real-time annotated video with detections

Extracted ANPR text

Violation alerts

Accident detection logs

Traffic heatmap visualizations

🤝 Contributing

Pull requests and suggestions are welcome!

📄 License

This project is released under the MIT License.

💡 Author
Bhavya 


<img width="799" height="594" alt="Screenshot 2025-11-18 233550" src="https://github.com/user-attachments/assets/a034ac4c-779a-43f3-bfb4-570f48179256" />
<img width="1339" height="875" alt="Screenshot 2025-11-16 172837" src="https://github.com/user-attachments/assets/46118cdf-1567-4d3f-9770-32d001ce7907" />



