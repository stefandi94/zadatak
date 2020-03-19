### First task:
Run ms_clustering.py script for the first part. It will return two images,
saved in images folder. 

Source code for this is in *source/main_shift.py*


### Second task:

Run docker-compose up and send post request **{"data": image_file}**
to the **http://localhost:8000/app/predict/**. 
It returns predicted class and confidence.d

Source code for this is in models.

For the full pipeline, one should first run *annotation_preprocessing.py*, 
then *data_preprocessing.py* scripts in *source/preprocessing*. 
After this, data is available for deep learning training. 
For this is used *train.py* script.