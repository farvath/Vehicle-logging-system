import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import pandas as pd
from datetime import datetime, timedelta
import os
import pytesseract
import re


COUNT=0

classNames = ['Car/Jeep/Van/MV', 'LCV/LGV/MiniBus', '2 axle', '3-axle', '4 to 6 axle', 'oversized']
detected_categories = ['category 1', 'category 2', 'category 3', 'category 4', 'category 5', 'category 6']

category_counts = {'category 1': 0,
                   'category 2': 0,
                   'category 3': 0,
                   'category 4': 0,
                   'category 5': 0,
                   'category 6': 0
                   }
class_mapping = {
    "Car/Jeep/Van/MV": "category 1",
    "LCV/LGV/MiniBus": "category 2",
    "2 axle": "category 3",
    "3-axle": "category 4",
    "4 to 6 axle": "category 5",
    "oversized": "category 6"
}

def map_class_name(original_class_name):
    return class_mapping.get(original_class_name, original_class_name)

model = YOLO('best.pt')

# Set tesseract path 
# find the tesseract.exe in local path and pass it below
pytesseract.pytesseract.tesseract_cmd = r'F:/cctv/tesseract-ocr/tesseract.exe'
prev_log_time = None


# Tracking
tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.6)


# Define the limits for angle 1
#limits = [0,214, 567,214]
#limits1 = [0,57,570,57]

# Define the limits for angle 2
limits = [0,234, 738,228]
limits1 = [0,105,739,107]

# Define the limits for angle 3
#limits = [390,0, 390,196]
#limits1 = [20,0,20,196]

# Define the limits for angle 4
#limits = [0,437, 775,437]
#limits1 = [0,85,779,85]


tracked_object_ids = set()

# Excel file name (initialize it to None)
excel_file_name = None

# Create a DataFrame to store counts at each x  interval
count_df = pd.DataFrame(columns=['Interval'] + list(category_counts.keys()))
count_data = []

frame_count = 0
frame_skip = 3



# Path to the folder containing video files
folder_path = "test_folder"

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    
    
   
        video_path = os.path.join(folder_path, filename)
        
        tracked_object_ids = set()
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if the video is opened successfully
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue
        
        while True:
            
            success, imag = cap.read()
            if not success:
                break

            img = cv2.resize(imag, (1080, 720))
            
           

            # roi for angle_1
            #roi = img[50:350, 380:950]
            
            # roi for angle_2
            roi = img[10:350, 200:940]  
            
            #roi for angle_3            
            #roi = img[120:350 ,  90:580]
            
            
            # roi for angle_4
            #roi = img[200:720, 300:]
            
           
            
            
            frame_count += 1
            
            if frame_count % (frame_skip + 1) == 0:
                
                print("CURRENT VIDEO FILE  :"+video_path)
                print("COUNT",COUNT)
                
                # To view the detection process to can pass a attribute " show=True " i.e : results = model(roi, show=True ,conf=0.5)
                
                results = model(roi, conf=0.40)

                detection = np.empty((0, 5))

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        conf = math.ceil((box.conf[0] * 100)) / 100  # confidence
                        cls = int(box.cls[0])  # class name
                        classfound = classNames[cls]

                        if classfound == "Car/Jeep/Van/MV" or classfound == "LCV/LGV/MiniBus" or classfound == "2 axle" or \
                                classfound == "3 axle" or classfound == "4 to 6 axle" or classfound == "oversized"  :
                            new_class_name = map_class_name(classfound)
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            detection = np.vstack((detection, currentArray))

                tracker_results = tracker.update(detection)

                # Drawing the lines
                cv2.line(roi, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 2)
                cv2.line(roi, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (255, 0, 0), 2)
                
                
                
                elapsed_time=False
                
                for res in tracker_results:
                    x1, y1, x2, y2, id = res
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(roi, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(roi,f'{int(id)}',(max(0,x1),max(35,y1)),scale=1,thickness=1,offset=10)
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(roi, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

                    # Check if the object crosses any of the lines
                    if (limits[0] < cx < limits[2] and limits[1]-10 < cy < limits[1] + 10) or (limits1[0] < cx < limits1[2] and limits1[1] -10 < cy < limits1[1] + 10) or (0 < cx < 739 and 107 +10 < cy < 228 - 10):
                        
                        if id not in tracked_object_ids:
                            tracked_object_ids.add(id)
                                
                            # detecting the time
                            #time_frame = img[:85, 945:]
                            time_frame = img[:85, 935:]

                           
                            gray_image = cv2.cvtColor(time_frame, cv2.COLOR_BGR2GRAY)
                            

                            #angle1 and angle_2 its same
                            _, threshold_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
                            
                            # angle3
                            #_, threshold_image = cv2.threshold(gray_image, 215, 255, cv2.THRESH_BINARY_INV)

                            # Angle4
                            #_, threshold_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
                            
                           

                            inverted_image = cv2.bitwise_not(threshold_image)
                            cv2.imshow("dsd",inverted_image)
                            
                            text = pytesseract.image_to_string(inverted_image)
                            print("text : ",text) 
                            match = re.search(r'\d{2}:\d{2}:\d{2}', text)
                            
                            print("match : ", match)
                           
                        
                            if match:
                                time_str = match.group()
                                print("time_str : ", time_str)
                                
                                if prev_log_time is None:
                                    prev_log_time = datetime.strptime(time_str, '%H:%M:%S')
                                        
                                # Check if 15 minutes have passed since the PREVIOUS logging
                                try:
                                    latest_time = datetime.strptime(time_str, '%H:%M:%S')
                                except ValueError as e:
                                    latest_time=prev_log_time+timedelta(seconds=60)
                                    
                                time_difference = latest_time - prev_log_time
                                
                                # Check if the time difference is greater than or equal to 15 minutes
                                # Change you preffered  time difference (x )
                                if time_difference >= timedelta(seconds=20):
                                    elapsed_time=True
                                    
                                    prev_log_time=latest_time
                            else :
                                COUNT=+1
                                print("-----------------------match not  found ----------------")       
                            if new_class_name in detected_categories:
                                category_counts[new_class_name] += 1
                                    
                                
                        # Drawing the lines
                        cv2.line(roi, (limits[0] , limits[1] ), (limits[2] , limits[3]), (255, 0, 255), 2)
                        cv2.line(roi, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 2)
                            

                

                if elapsed_time == True:  #  the flag is made True after x minutes have passed
                    print('updating the excel sheet...')
                    count_row = {'Interval':prev_log_time.strftime('%H:%M:%S')}
                    count_row.update(category_counts)
                    count_data.append(count_row)
                    category_counts = dict.fromkeys(category_counts, 0)  # reset category counts
                
                    

                    if excel_file_name is None:
                        # Create a new Excel file for the first 15 minutes
                        excel_file_name = os.path.abspath(f'counts_{folder_path}.xlsx')
                        # Create the directory if it doesn't exist
                        os.makedirs(os.path.dirname(excel_file_name), exist_ok=True)
                        pd.DataFrame(count_data).to_excel(excel_file_name, index=False)
                        print("-----------------------updating the excel  ----------------")
                        print(count_data)
                    else:
                        # Append to the existing Excel file for subsequent x intervals
                        with pd.ExcelWriter(excel_file_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                            if 'Sheet1' not in writer.book.sheetnames:
                                pd.DataFrame(count_data).to_excel(writer, index=False, sheet_name='Sheet1')
                            else:
                                # Existing sheet found, append data without writing header
                                pd.DataFrame(count_data).to_excel(writer, index=False, header=False, sheet_name='Sheet1')

                    print(f'Counts saved to {excel_file_name}.')
            

            
                #you can remove the below lines to make it faster , but you will not be able to see the frames
                cv2.imshow("Car Counter", img)
                

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("THE DATA IS SAVED TO : "+excel_file_name)
        
        # Release the video capture object
        cap.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()

