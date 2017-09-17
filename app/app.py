# -*- coding: utf-8 -*- 
#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#future for python 2 and 3
from __future__ import print_function

#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images

#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
#mongo
from flask_pymongo import PyMongo
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
#initalize our flask app
app = Flask(__name__)
import pandas as pd
from googleapiclient.errors import HttpError
from googleapiclient import sample_tools
from oauth2client.client import AccessTokenRefreshError
import argparse
import sys
import collections
import csv
import numpy as np
import pandas as pd 
        
#The DB connection will assume that the database has the same name as the Flask Appliction which is "app"
app = Flask(__name__)
mongo = PyMongo(app)
UPLOAD_FOLDER = os.getcwd()+'/uploads' #'gs://neemfs/'
ALLOWED_EXTENSIONS = set(['raw','flac','mp3','wav'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# URI scheme for Cloud Storage.
GOOGLE_STORAGE = 'gs'
# URI scheme for accessing local files.
LOCAL_FILE = 'file'
brand = "Messiac"
url = "http://messiac.com"
customer = "Boomfix.es"
	
#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(imgstr.decode('base64'))

#######################
#        VIEWS        #
#######################

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("home.html", brand = brand)
	
@app.route('/prediktor')
def product():
	#initModel()
	#render out pre-built HTML file right on the index page
	brand = "Messiac"
	url = "http://messiac.com"
	os.system("python ../session_recorder/real_time.py;")
	return render_template("demo.html", brand = brand, url=url, customer=customer)

#######################
#                     #
#######################
@app.route('/real_time')
def real_time():
  return render_template("real_time.html", brand = brand)
    
#######################
#                     #
#######################
	
#Store the CVS file with the data matrix
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#View lilst of 
@app.route('/call_list', methods=['GET', 'POST'])
def upload_file():
    gs = "gs://neem-fs.appspot.com/"
    files = os.listdir(UPLOAD_FOLDER)
    # upload a new file to the view
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path_name, file_extension = os.path.splitext('./uploads/'+filename)
            if file_extension != '.raw':
                tfm = sox.Transformer()
                tfm.build(app.config['UPLOAD_FOLDER']+'/'+filename, os.path.splitext(file_path_name)[0]+'.raw')
                filename=os.path.splitext(filename)[0]+'.raw'
                os.system("gsutil cp ./uploads/"+filename+" "+gs+filename)
                print("does it?")
            #upload the file to gs    
            print(filename) 
            print(app.config['UPLOAD_FOLDER'])
            # Once the file has been stored in GS, we generate the transcript
            from google.cloud import speech
            client = speech.Client()
            hints = ['pantalla', 'iphone', '119', '69','bateria']
            sample = client.sample(content=None,source_uri=gs+filename,encoding='LINEAR16',sample_rate_hertz=8000)
            operation = sample.long_running_recognize(language_code='es-CL',max_alternatives=1, speech_contexts=hints)
            retry_count = 100
            while retry_count > 0 and not operation.complete:
                retry_count -= 1
                time.sleep(10)
                operation.poll()  # API call
            operation.complete
            for result in operation.results:
                for alternative in result.alternatives:
                    print('=' * 20)
                    print(alternative.transcript)
                    print(alternative.confidence)
                    save = mongo.db.transcripts.insert({'filename':filename, 
                        'content': {'text':alternative.transcript, 'confidence':alternative.confidence,
                        'verified':0}})
                    transcript = mongo.db.transcripts.find({'filename':filename})
            return render_template('transcript.html',user=user, transcript=transcript) 
    return render_template('call_list.html', files=files, brand = brand, url=url, customer=customer)
            
# retrieve the .CVS file
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

#2 - SHOW ONE PARTICULAR INTERACTION (FILENAME)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS    

# This show the trasnscript for one given audio file
@app.route('/transcript/<filename>')
def transcript(filename):
    transcript = mongo.db.transcripts.find({'filename':filename}) 
    return render_template('transcript.html', transcript=transcript, brand = brand, url=url, customer=customer)

# This show the trasnscript for one given audio file
@app.route('/analysis/<filename>')
def analysis(filename):
    #before anything, groupby phone number
    telephone = filename[0:11]
    #first, find the transcript in mongo
    transcript = mongo.db.transcripts.find({'filename':filename})
    #create a list of words for that transcript
    transcript_content = transcript[0]['content']['text']
    #separate each word with one space
    transcript_content_ready = str(transcript_content).split(" ")
    #secondly, iterate each case to verify which one fits the transcript better
    case = mongo.db.cases.find()
    for i in case:
        case = str(i['name'])
    # 1 - store the matching words from the transcript
        keywords = str(i['keywords']['identify']).split(" ")
        matching_keywords = Counter(set(keywords).intersection(transcript_content_ready)).keys()
        count_matching_keywords = int(len(set(keywords).intersection(transcript_content_ready)))
    # 2 - verify if the sale has been completed
        successful_keywords = str(i['keywords']['successful']).split(" ")
        successful_matching_keywords = Counter(set(successful_keywords).intersection(transcript_content_ready)).keys()
        count_successful_matching_keywords =  int(len(set(successful_keywords).intersection(transcript_content_ready)))
    # store these results on the transcript in the database
        inserter = mongo.db.matches.save({'filename':filename, 'telephone': telephone, 'case': case, 'matching_keywords': matching_keywords,
         'count_matching_keywords': count_matching_keywords, 'successful_keywords':successful_matching_keywords,
         'count_successful_matching_keywords':count_successful_matching_keywords})
    # Analize the results
    match = mongo.db.matches.find({'telephone':telephone}, sort=[("count_matching_keywords", -1)]).limit(1)
    # Store the sale case for that customer
    identify_case = match[0]['case']
    # Store sale case for that customer
    case_keywords = match[0]['count_matching_keywords']
    # Identify the if the sale was successful
    success = mongo.db.matches.find({'telephone':telephone}, sort=[("count_successful_matching_keywords", -1)]).limit(1)
    return render_template('analysis.html', transcript=transcript, match=match, success=success, brand = brand, url=url, customer=customer)


@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	imgData = request.get_data()
	#encode it into a suitable format
	convertImage(imgData)
	print("debug")
	#read the image into memory
	x = imread('output.png',mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x)
	#make it the right size
	x = imresize(x,(28,28))
	#imshow(x)
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,28,28,1)
	print("debug2")
	#in our computation graph
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		print("debug3")
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	
	

#if __name__ == "__main__":
	#decide what port to run the app in
	#port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	#app.run(host='localhost', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
