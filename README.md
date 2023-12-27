# mma-videos-parser
CV parser of MMA translation records with extraction of all information from popup windows

## Pre-requirements
Python version 3.9+  
easyocr==1.7.1  
  
Video 1280 × 720  
  
## How to run on Linux  
1. `pip3 install -r requirements.txt`
2. `python3 main.py --path_video <path to video folder>/mma_video_name.mp4 --path_result ./output.json`  

## How to run in Docker  
1. Make sure that you have free space at least 10 GB (easyocr has very heavy dependencies) and ~40 minutes time for building this container   
2. `sudo docker build -t mma-parser -f Dockerfile .`  
3. `sudo docker run -v $(pwd)/<path to your folder with video>:/home/mma-parser/iofolder mma-parser --path_video ./iofolder/mma_video_name.mp4 --path_result ./iofolder/output.json`

You will see in your $(pwd)/your_folder_with_video folder the json file with parsing results.  
