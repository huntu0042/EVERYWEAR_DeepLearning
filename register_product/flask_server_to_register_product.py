#-*- coding: utf-8 -*-


import threading

from flask import Flask, render_template, request, url_for
import time
import socket
import os

from werkzeug import secure_filename

mall_name = 'test3'
PROD_FOLDER = '../data/' + mall_name

################flask################
host = socket.gethostname()
port = 12226



app = Flask(__name__)
app.config.from_object(__name__)

def SockSend(userid,imageid):
	s = socket.socket()
	host = socket.gethostname()
	port = 12226

	s.connect((host, port))
	print( 'Connected to', host)


  
	send_str = str(userid) + " " + str(imageid)
	print(send_str)
	print('##\n\n')
	print(s.send(send_str.encode('utf-8')))


#Define a route for url
@app.route('/')
def form():
	#imgplus()
	return "HI BRO"

#upload user picture
@app.route('/userupload', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    prodid = request.form['userid']
    print(prodid)
    category = request.form['imageid']
    print(category)
    if category == "1001":
      upload_path = PROD_FOLDER + "/men_tshirts/images/"
    elif category == "1002":
      upload_path = PROD_FOLDER + "/men_nambang/images/"
    elif category == "1003":
      upload_path = PROD_FOLDER + "/men_long/images/"
    elif category == "1101":
      upload_path = PROD_FOLDER + "/men_pants/images/"
    try:
      file = request.files['userpicture']
      filename = secure_filename(file.filename)
      file.save(os.path.join(upload_path, prodid + "_1.jpg"))
    except Exception as ex:
      print("ERROR: ", ex)
    try:
      SockSend(prodid, category)
      return "OK"
    except Exception as ex:
      print("ERROR: ", ex)

  return '''
  <!doctype html>
  <title>Upload new File</title>
  <h1>Upload new File</h1>
  <form action="" method=post enctype=multipart/form-data>
  <p><input type=file name=userpicture>
    <input type=submit value=Upload><br><br>
    prodid : <input type=text name = userid>
    category : <input type=text name = imageid>
  </form>'''


#Run the app
if __name__ == '__main__':
	#th = threading.Thread(target=imgplus,args=())
	app.run(host='127.0.0.1', debug=True, port =10008)
