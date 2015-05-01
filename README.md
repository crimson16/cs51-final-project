#Recognizing Handwritten Digits Using K-means Clustering
### CS 51 Final Project 2015
By: Olivia Angiuli, Martin Reindl, Ty Rocca, Wilder Wohns

### Setup Instructions :

#####Required Packages
+ python image library (PIL)
    - ` pip install PIL `
+ numpy, scipy, scikit-learn:
    - `pip install -U numpy scipy scikit-learn `
+ django version 1.7:
    - ` pip install django `
+ iPython Notebook
    - `pip install "ipython[notebook]"`
+ CVXOPT
    - `pip install cvxopt --user”` (or follow http://cvxopt.org/install/)
    
##### Running the Code from the Command Line

To run our code you need to first be in the /code folder of our project. Onece there, our code takes up to 4 command line arguments. The format for this is:

* `python main_cluster.py {k} {method} {init_type} {prop} `
    - *k* is the number of clusters
    - *method* is “means”, “medians”, or “medoids”
    - *init_type* is “random” or “kplusplus”
    - *prop* is a number from 0 to 100 which is the percentage of training data to train on
   
* Running the code in **nohup**
    - Running the code as a nohuped process is nice becasue when using the full dataset clustering can take up to 4 hours. This no hup format will still give you all of the images and the printed out put will be saved in the \*.out folder
    - `nohup python main_cluster.py {k} {method} {init_type} {prop} > {method}_{init_type}_clusters{k}.out&`

##### Running the Code in the iPython Notebook

You also have the option to explore our code in an iPython Notebook. The iPython Notebook is a great top level interface for interacting with our code, as well as seeing all of our plots, and a bit of how we process the image in the django app. You can view a static version of our notebook [here](http://nbviewer.ipython.org/github/crimson16/cs51-final-project/blob/master/code/cs51_notebook.ipynb) or you can run it from the code folder with the command: `ipython notebook`. Within the notebook the **main** function will run the code much like the one above. Main take 3 arguments `(k, method, init_type)`. Method is by default "means" and initialization type is by default "kplusplus". You can change "k" at the top of the page to change how much data is loaded.

##### Running the Django App

Our Django app is another fun bonus part of the our project. The django app allows us to run a front end interface where our users can see our clusters get put to the test. The bulk of the code is in *code/recognize_digits/views.py*. This is where all of the backend image processing happens. The font end javascript is stored in *code/static/js/homepage.js*. This makes the interactive canvas.

* To run the Django app:
    - cd into \code
    - run ` python manage.py runserver 8000 ` to run the app on your computer
        + *If it is your first time running the app you might need to run python manage.py make migrations*
    - Then go to [http://localhost:8000/](http://localhost:8000/) where the app should now be running
    - Draw a digit and click predict!
    
##### Optimizing function
* To run the optimization function (which will take a very long time to run since it clusters 102 different ways type ``python optimize.py`` and it will run the optimization of the code.
        
    

