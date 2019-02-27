
import urllib.parse
import urllib.request

"""
    Args:
    - isupper
    
    - category
    1001 : men_tshirts
    1002 : men_nambang
    1003 : men_long
    1101 : men_pants


data = urllib.parse.urlencode({
    'userid': 'cherry',
    'imageid' : '000001',
    'upperid': '102006',
	  'lowerid' : '000000',
    'isupper' : '1', 
    'category': '1001'

})
"""
data = urllib.parse.urlencode({
    'userid': 'cherry',
    'imageid' : '000001',
    'upperid': '102006',
	  'lowerid' : '000007',
    'isupper' : '1', 
    'category': '1101'

})

url = "http://127.0.0.1:10008/submit"

data = data.encode('ascii')
with urllib.request.urlopen(url, data) as f:
    print(f.read().decode('utf-8'))
