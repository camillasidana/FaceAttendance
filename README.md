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
```
This will train the model and will also give you perfomace results of each of those models.
```python
python main.py --stage attend
```
The app automatically chooses the best model based on the previous results.
You must see a box of your name around your face now. Your attend has been marked and it gets saved to attendance.db file in .\data\ once you close the cam. To close the cam, press "q". You can now view the attendance using simple queries on sqlite3. Following is how you can do it :
Make sure you already have sqlite3. On the terminal write,
```bash
sqlite3 data/attendance.db
```
Now in sqlite3 write the following query,
```sql
.tables
SELECT * FROM attendance;
```
That's it!
Thank you!




