This project implements an automated classroom attendance system using face recognition.
It captures faces through a webcam, identifies known students using machine learning models, and automatically marks attendance in a SQLite database.

The project demonstrates:

Data storage & manipulation using NumPy and Pandas

Database operations (read, write, modify)

Exploratory Data Analysis (EDA) on facial embeddings

Feature reduction using PCA

Machine learning models (SVM, Decision Tree, KNN)

Performance evaluation using classification metrics

Version control and sharing via GitHub

To test this app, kindly clone this repo to your local machine. Make sure that you have python 3.13 installed. Add a folder of your name in data\raw\ and add 15-20 pictures of your face in it. 
Create a virtual enviornment and inside the virtual environment run the following :
```python
pip install -r requirements.txt
```
This will install the required libraries need for the app to run.
After this run : 
```python
python main.py --stage prep
python main.py --stage train
python main.py --stage attend
```
You must see a box of your name around your face now.
To get the evaluations run :
```python
python main.py --stage eval
```
And those are our test results.
That's it!
Thank you!


