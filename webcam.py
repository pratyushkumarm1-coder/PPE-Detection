import cv2, time
from ultralytics import YOLO

def main():
    print("Loading model...")
    # Load the trained model
    model = YOLO('best.pt')

    print("Opening webcam...")
    # Open the default camera (0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Warm up the camera so frames are not dark/blank on start
    print("Warming up camera...")
    for _ in range(10):
        cap.read()

    print("Live video started! Press 'q' in the video window to quit.")

    prev_time = time.time()
    fps = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break
        
        # Run inference filtering to only Helmet (class 0) and No-Helmet (class 1)
        results = model(frame, conf=0.10, classes=[0, 1], verbose=False)
        
        # Ask YOLO to draw the bounding boxes
        annotated_frame = results[0].plot()
        
        # Compute FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show detection summary
        n_helmet = sum(1 for b in (results[0].boxes or []) if int(b.cls[0]) == 0)
        n_nohelmet = sum(1 for b in (results[0].boxes or []) if int(b.cls[0]) == 1)
        cv2.putText(annotated_frame, f"Hardhat: {n_helmet}  NO-Hardhat: {n_nohelmet}",
                    (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Check if anyone without a helmet is in the frame
        no_helmet_detected = False
        if results[0].boxes:
            for box in results[0].boxes:
                # Class 1 is 'No-Helmet'
                if int(box.cls[0]) == 1:
                    no_helmet_detected = True
                    break
        
        # Display the red warning if No-Helmet is detected
        if no_helmet_detected:
            # Draw big red text at the top
            cv2.putText(annotated_frame, "WARNING: NO HELMET DETECTED!", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 3, cv2.LINE_AA)
            
            # Draw a thick red border around the entire video feed
            annotated_frame = cv2.copyMakeBorder(annotated_frame, 10, 10, 10, 10, 
                                                cv2.BORDER_CONSTANT, value=[0, 0, 255])
        else:
            # Draw a neutral black border so the video size doesn't jump around
            annotated_frame = cv2.copyMakeBorder(annotated_frame, 10, 10, 10, 10, 
                                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
        # Display the live annotated frame
        cv2.imshow("Live PPE Detection - Press 'q' to quit", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
