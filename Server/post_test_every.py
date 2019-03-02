
import urllib.parse
import urllib.request



URL = "http://172.16.101.36:10008/submit"

F_tshirts = open("tshirts_list.txt",'r').read().split('\n')
F_pants = open("pants_list.txt",'r').read().split('\n')

print(F_pants)

def sendData(data):
	data = data.encode('ascii')
	with urllib.request.urlopen(URL, data) as f:
		print(f.read().decode('utf-8'))



"""
    Args:
    - isupper
    0 : No composed upper image
    1 : composed upper image
    - category
    1001 : men_tshirts - 0001 ~
    1002 : men_nambang - 1001 ~
    1003 : men_long	
	url = "http://172.16.101.36:10008/submit"
	#url = "http://13.209.7.130:10008/submit"

	sendData(data,u
    1101 : men_pants - 5001 ~
"""

for tshirts in F_tshirts:

	data = urllib.parse.urlencode({
		'userid': 'test',
		'imageid' : '12345667',
		'upperid': tshirts,
		'lowerid' : '000000',
		'isupper' : '1', 
		'category': '1001'
	})

	sendData(data)

for pants in F_pants:

	data = urllib.parse.urlencode({
		'userid': 'test',
		'imageid' : '12345667',
		'upperid': '000000',
		'lowerid' : pants,
		'isupper' : '0', 
		'category': '1101'
	})


	url = "http://172.16.101.36:10008/submit"

	sendData(data)

	
	
	

	
	
	