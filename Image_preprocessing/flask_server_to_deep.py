#-*- coding: utf-8 -*-


import threading

from flask import Flask, render_template, request, url_for
import time
import socket
import os

from werkzeug import secure_filename
UPLOAD_FOLDER = '../testdata/'



################flask################
host = socket.gethostname()
port = 12222
preprocess_port = 12224


app = Flask(__name__)
app.config.from_object(__name__)


def SockSend(userid,imageid,upperid,lowerid,isupper,category):
	s = socket.socket()
	host = socket.gethostname()
	port = 12222

	s.connect((host, port))
	print( 'Connected to', host)


  
	send_str = str(userid) + " " + str(imageid) + " " + str(upperid) + " " + str(lowerid) + " " + str(isupper) + " " + str(category)
	print(send_str)
	print('##\n\n')
	print(s.send(send_str.encode('utf-8')))

	#s.close()


def SockSend_PreProcess(userid,imageid):
	s = socket.socket()
	host = socket.gethostname()
	port = 12224

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

#form action
@app.route('/submit', methods=['POST'] )
def action():
	try:
		#userid = request.args.get('userid',"")
		userid = request.form['userid']
		print("USER ID    : " + userid)
		#productid = request.args.get('productid',"")
		imageid = request.form['imageid']
		print("image id : " + imageid)
		upperid = request.form['upperid']
		print("Upper PRODUCT ID : " + upperid)
		lowerid = request.form['lowerid']
		print("Lower PRODUCT ID : " + lowerid)
		isupper = request.form['isupper']
		print("isupper : " + isupper)
		try:
			category = request.form['category']
		except:
			category = "ERRORGORY"
		print("CATEGORY   : " + category)
		if len(userid) > 3 and len(upperid) > 3 and len(lowerid) > 3:
			send_str = str(userid) + " " + str(imageid) + " " + str(upperid) + " " + str(lowerid) + " " + str(isupper) + " " + str(category)
			print(send_str)
			#try:
			SockSend(userid,imageid,upperid,lowerid,isupper,category)
			#catch:
			#print("연결 실패")
		else:
			print("wrong data")
		print("\n\n")
	except:
		"submit ERROR"
	'''
	s = socket.socket()
	host = socket.gethostname()
	port = 12222

	s.connect((host, port))
	print( 'Connected to', host)

	#z = input("Enter something for the server: ")
	#s.send(z.encode('utf-8'))


	print(s)
	send_str = userid + " " + productid
	print(s.send(send_str.encode('utf-8')))
	s.close()
	'''
	#time.sleep(3)
	return "http://img.ezmember.co.kr/cache/board/2017/03/13/33760d6d6ddb60a5fbb390869a219d90.jpg"

#form action
'''
@app.route('/upload',upload methods=['POST'] )
def uploadaction():
	productid = request.form['productid']
	print(productid)
	#time.sleep(3)
	return "Success"
'''

	
#upload user picture
@app.route('/userupload', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		try: 
			userid = request.form['userid']
			print("USER ID    : " + userid)
			#productid = request.args.get('productid',"")
			imageid = request.form['imageid'] 
			print("image id : " + imageid)

			upload_path = UPLOAD_FOLDER + "/" + userid + "/input/body_images/"

			file = request.files['userpicture']
			filename = secure_filename(file.filename)
			#file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			file.save(os.path.join(upload_path, imageid+ "_0.jpg"))

			SockSend_PreProcess(userid,imageid)
			#print(url_for('uploaded_file',filename=filename))
			return "OK"#redirect(url_for('uploaded_file',filename=filename))3

		except Exception as ex:
			print("ERRROR : ",ex)


	return '''
  <!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form action="" method=post enctype=multipart/form-data>
	<p><input type=file name=userpicture>
	 <input type=submit value=Upload><br><br>
	 userid : <input type=text name = userid>
	 imageid : <input type=text name = imageid>
	</form>'''


#Run the app
if __name__ == '__main__':
	#th = threading.Thread(target=imgplus,args=())
	app.run(host='127.0.0.1', debug=True, port =10008)
