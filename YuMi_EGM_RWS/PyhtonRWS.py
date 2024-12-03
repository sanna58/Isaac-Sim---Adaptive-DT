# Web socket client using https://ws4py.readthedocs.org/en/latest/
import sys, argparse
import xml.etree.ElementTree as ET
from ws4py.client.threadedclient import WebSocketClient
import requests
from requests.auth import HTTPDigestAuth
 
namespace = '{http://www.w3.org/1999/xhtml}'
 
def print_event(evt):
    root = ET.fromstring(evt)        
    if root.findall(".//{0}li[@class='pnl-ctrlstate-ev']".format(namespace)):
        print("\tController State : " + root.find(".//{0}li[@class='pnl-ctrlstate-ev']/{0}span".format(namespace)).text)
    if root.findall(".//{0}li[@class='pnl-opmode-ev']".format(namespace)):
        print("\tOperation Mode : " + root.find(".//{0}li[@class='pnl-opmode-ev']/{0}span".format(namespace)).text)
    if root.findall(".//{0}li[@class='pnl-speedratio-ev']".format(namespace)):
        print("\tSpeed Ratio : " + root.find(".//{0}li[@class='pnl-speedratio-ev']/{0}span".format(namespace)).text)
 
 
 
 # This class encapsulates the Web Socket Callbacks functions.
class RobWebSocketClient(WebSocketClient):
    def opened(self):
        print ("Web Sockect connection established")

    def closed(self, code, reason=None):
        print ("Closed down", code, reason)
 
    def received_message(self, event_xml):        
        if event_xml.is_text:            
            print ("Events : ")
            print_event(event_xml.data.decode("utf-8"))
        else:
            print ("Received Illegal Event " + str(event_xml))
        
    
# The main RobotWare Panel class
class RWPanel:
 
    def __init__(self, host, username, password):
        self.host = host
        self.username = username
        self.password = password
        self.digest_auth = HTTPDigestAuth(self.username,self.password)
        self.subscription_url = 'http://{0}/subscription'.format(self.host)
        self.session = requests.Session()
        
    
    def subscribe(self):       
       # Create a payload to subscribe on RobotWare Panel Resources with high priority     
        payload = {'resources':['1','2','3'],
            '1':'/rw/panel/speedratio',
            '1-p':'1',
            '2':'/rw/panel/ctrlstate',
            '2-p':'1',
            '3':'/rw/panel/opmode',
            '3-p':'1'}        

        resp = self.session.post(self.subscription_url , auth=self.digest_auth, data=payload)
        print ("Initial Events : ")
        print_event(resp.text)           
        if resp.status_code == 201:
            self.location = resp.headers['Location']
            self.cookie = '-http-session-={0}; ABBCX={1}'.format(resp.cookies['-http-session-'], resp.cookies['ABBCX'])
            return True
        else:
           print ('Error subscribing ' + str(resp.status_code))
           return False  

    def set_RUN_SG_ROUTINE_DI(self, payload_value):
        url_SG = "http://192.168.125.1/rw/iosystem/signals/RUN_SG_ROUTINE?action=set"
        # payload = {'lvalue': '1'}
        payload = {'lvalue': payload_value}
        response = requests.post(url_SG, data=payload, auth=self.digest_auth)
        if response.status_code == 204:
            print("RUN_SG_ROUTINE_DI Activated")
        else:
            print("RUN_SG_ROUTINE_DI Deactivated")

    def set_EGM_START_JOINT(self):
        url_SG = "http://192.168.125.1/rw/iosystem/signals/EGM_START_JOINT?action=set"
        # payload = {'lvalue': '1'}
        payload = {'lvalue': '1'}
        response = requests.post(url_SG, data=payload, auth=self.digest_auth)
        if response.status_code == 204:
            print("EGM_START_JOINT Activated")
        else:
            print("EGM_START_JOINT Deactivated")

    def COMMAND_GRIP_IN_OUT(self, payload_value):
        url_COMMAND_GRIP_IN_OUT = "http://192.168.125.1/rw/rapid/symbol/data/RAPID/T_ROB_R/TRobSG/command_input?action=set"
        payload_COMMAND_GRIP_IN_OUT = {'value': payload_value} # 4 is for COMMAND_GRIP_IN and 5 is for COMMAND_GRIP_OUT
        response_COMMAND_GRIP_IN_OUT = requests.post(url_COMMAND_GRIP_IN_OUT, data=payload_COMMAND_GRIP_IN_OUT, auth=self.digest_auth)
        if response_COMMAND_GRIP_IN_OUT.status_code == 204:
            if payload_value == '4':
                print("Gripper Closing")
            elif payload_value == '5':
                print("Gripper Opening")
            else:
                print("Wrong gripper motion command")
        else:
            print("COMMAND_GRIP_IN_OUT Request Unsuccessful", response_COMMAND_GRIP_IN_OUT.status_code)

    # def COMMAND_GRIP_OUT(self):
    #     url_COMMAND_GRIP_OUT = "http://localhost/rw/rapid/symbol/data/RAPID/T_ROB_R/TRobSG/command_input?action=set"
    #     payload_COMMAND_GRIP_OUT = {'value': '5'} # 5 is according to the number assigned for this command input in the RAPID code
    #     response_COMMAND_GRIP_OUT = requests.post(url_COMMAND_GRIP_OUT, data=payload_COMMAND_GRIP_OUT, auth=self.digest_auth)
    #     if response_COMMAND_GRIP_OUT.status_code == 204:
    #         print("COMMAND_GRIP_OUT Request successful")
    #     else:
    #         print("COMMAND_GRIP_OUT Request Unsuccessful")
 
    def start_recv_events(self):
        self.header = [('Cookie',self.cookie)]
        self.ws = RobWebSocketClient(self.location, 
                                     protocols=['robapi2_subscription'], 
                                     headers=self.header)
        self.ws.connect()
        self.ws.run_forever()

    def close(self):
        self.ws.close()  

 
def enable_http_debug():
    import logging
    import httplib
    httplib.HTTPConnection.debuglevel = 1
    logging.basicConfig() # Initialize logging
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True



def main(argv):
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-host",help="The host to connect. Defaults to localhost on port 80", default='192.168.125.1:80')
        parser.add_argument("-user",help="The login user name. Defaults to default user name", default='Default User')
        parser.add_argument("-passcode",help="The login password. Defaults to default password", default='robotics')  
        parser.add_argument("-debug",help="Enable HTTP level debugging.", action='store_true')  
        args = parser.parse_args()       
      
        if args.debug:
            enable_http_debug()
        
        rwpanel = RWPanel(args.host, args.user, args.passcode)
        if rwpanel.subscribe():
            rwpanel.set_RUN_SG_ROUTINE_DI()
            rwpanel.start_recv_events()
    except KeyboardInterrupt:
        rwpanel.close()
 
 
if __name__ == "__main__":
     main(sys.argv[1:])