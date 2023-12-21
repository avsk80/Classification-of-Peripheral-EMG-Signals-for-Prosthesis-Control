## EMG Feature Extraction

This code repository has been adapted from https://github.com/SebastianRestrepoA/EMG-pattern-recognition.git, where signal preprocessing is done. We have added our data specific steps in the driver program suguru_feature_extraction.py 


### Prerequisites
You must have NumPy, Pandas, Matplotlib, Scipy, and  Pyyawt installed.

### Project Structure

1. digital_processing.py - It contains the digital filters (notch and band pass) configuration to eliminate signal noise and artifacts.
2. feature_extraction.py - It allows to compute time and frequency features above an EMG signal.
3. feature_extraction_scheme.py - It contains the feature extraction scheme: load data, biomedical signal processing and feature extraction.

### Running the project

1. Clone emg pattern recognition project in a local directory.
```
git clone https://github.com/SebastianRestrepoA/EMG-pattern-recognition.git
```

2. Create enviroment for run EMG pattern recognition project. In your cloned folder run the following commands:
```
virtualenv env
env\Scripts\activate
pip install pandas
pip install matplotlib
pip install numpy
pip install pyyawt 
```

4. Run feature_extraction_scheme.py using below command to see EMG characterization.
```
python feature_extraction_scheme.py
```

