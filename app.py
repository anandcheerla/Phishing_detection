from flask import Flask
import pandas as pd
from flask import request
from logic import *

app=Flask(__name__)


def helper_fun(SFH,popUpWindow,SSLfinal_State,Request_URL,URL_of_Anchor,web_traffic,URL_Length,age_of_domain,having_IP_Address):
	inp=[[SFH],[popUpWindow],[SSLfinal_State],[Request_URL],[URL_of_Anchor],[web_traffic],[URL_Length],[age_of_domain],[having_IP_Address]]
	columns_inp=[]
	data_inp={}
	j=0
	for itr in X_test:
		data_inp[itr]=inp[j]
		columns_inp.append(itr)
		j+=1

	df_inp = pd.DataFrame(data_inp, columns=columns_inp)


	res=best_clf.predict(df_inp)
	print(res)

	if(res[0]==-1):
		return "Phishy"
	elif(res[0]==0):
		return "Suspicious"
	else:
		return "Legitimate"


@app.route('/',methods=['GET', 'POST'])
def main_form():
	if request.method == 'POST':  #this block is only entered when the form is submitted
		SFH = request.form.get('SFH')
		popUpWindow = request.form.get('popUpWindow')
		SSLfinal_State = request.form.get('SSLfinal_State')
		Request_URL = request.form.get('Request_URL')
		URL_of_Anchor = request.form.get('URL_of_Anchor')
		web_traffic = request.form.get('web_traffic')
		URL_Length = request.form.get('URL_Length')
		age_of_domain = request.form.get('age_of_domain')
		having_IP_Address = request.form.get('having_IP_Address')

		res='''<h1>Website is {}</h1>'''.format(helper_fun(SFH,popUpWindow,SSLfinal_State,Request_URL,URL_of_Anchor,web_traffic,URL_Length,age_of_domain,having_IP_Address))
		return res



	return '''<form method="POST">
                  SFH: <input type="text" name="SFH"><br>
                  popUpWindow: <input type="text" name="popUpWindow"><br>
                  SSLfinal_State: <input type="text" name="SSLfinal_State"><br>
                  Request_URL: <input type="text" name="Request_URL"><br>
                  URL_of_Anchor: <input type="text" name="URL_of_Anchor"><br>
                  web_traffic: <input type="text" name="web_traffic"><br>
                  URL_Length: <input type="text" name="URL_Length"><br>
                  age_of_domain: <input type="text" name="age_of_domain"><br>
                  having_IP_Address: <input type="text" name="having_IP_Address"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''



app.run(debug=True)