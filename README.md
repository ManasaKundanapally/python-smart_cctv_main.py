# Smart CCTV Behaviour Analysis

A Python and OpenCV based smart surveillance project that detects people from a webcam feed and analyzes customer behaviour.

## Features
- Person detection
- Bounding box display
- People counting
- Time tracking
- Entry counting
- Line crossing detection
- CSV event logging

## Technologies Used
- Python
- OpenCV
- NumPy
- CSV

## Main File
Run the main project using:

```bash
python smart_cctv_main.py

## Future Improvements
- YOLO-based person detection
- Face detection
- Entry and exit tracking
- Analytics dashboard
## How to Run the Project

1. Install required libraries:
```bash
pip install -r requirements.txt
Added usage instructions to README
## Use Cases
- Retail store monitoring
- Customer behavior analysis
- Smart surveillance systems
## Output Screenshot

## Project Highlights
- Real-time person detection using OpenCV
- Customer entry tracking using line crossing logic
- Time tracking of individuals in frame
- CSV-based logging for analytics
- Modular Python scripts for scalability
![Project Output](screenshot.png)
## Example Output

When running the system, the console and screen display:

- People detected in real time
- Customers entered count
- Time spent in frame

## How It Works

1. Captures live video from webcam  
2. Detects people using OpenCV HOG detector  
3. Draws bounding boxes around detected persons  
4. Tracks movement and counts entries using line crossing  
5. Displays real-time data (People count, Entry count, Time)  
6. Logs data into CSV files for analysis

## Tech Stack

- Python
- OpenCV
- NumPy
- CSV (for data logging)

Example:
People: 1  
Entered: 3  
Time: 5s
## Author
Manasa Kundanapally
