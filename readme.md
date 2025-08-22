# The results are in output folder
## Also whenever you run the code the output folder that would be created in the same folder the where you ran the code

# Setup the environment:

## First download the files locally

## Then install dependencies
pip install -r requirements.txt

## If you just want see the results on the video that you provided run this:
python cover_drive_analysis_realtime.py --download --advanced

## If you have the video installed locally then run this:
python cover_drive_analysis_realtime.py --advanced --input 'your video path'

## If you have any other url of youtube video and want to analyze it then run this:
python cover_drive_analysis_realtime.py --advanced --url 'your link'



