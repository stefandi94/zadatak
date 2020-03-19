1. Run ms_clustering.py script for the first part. It will return two images,
saved in images folder. Source code for this is in source/main_shift.py

2. Run docker-compose up and send post request {"data": image_file}
to the http://localhost:8000/app/predict/. 
It returns predicted class and confidence.