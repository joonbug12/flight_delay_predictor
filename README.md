# Flight Delay Prediction & Airport Scorecard System

## pushing lfs files to git repos
1. git lfs install
2. git lfs track "data/flights.csv"
3. git add .gitattributes
4. git commit -m "message"


## clone git
To download the large files on your own local machines so you dont run into issues (i tested and if you do a normal clone, it just adds the small files)
1. You need to download git lfs before trying to clone 
    For Mac: brew install git-lfs
    For Windows go to https://git-lfs.github.com
2. Once you install you can run this in your terminal before you clone: git lfs install
3. Then you can type this: git lfs clone https://github.com/joonbug12/flight_delay_predictor.git
4. Now, you should be able to see all the files including each large data file

## Setup Environment
1. python3 -m venv flight_env
2. source flight_env/bin/activate

## Install dependencies
1. pip install pandas
2. pip install numpy
3. pip install scikit-learn
4. pip install matplotlib
5. pip install tensorflow
6. pip install flask
7. pip install joblib
8. pip install seaborn


## run the main program before the ui:
1. python3 main.py

## run the ui:
2. python3 app.py

