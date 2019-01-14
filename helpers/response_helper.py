__author__ = "Emir Ombasic"

import json

class ResponseHelper:

    response : False

    def setResponse(self, response):
        self.response = response

    def flushError(self, msg):
        self.response['error'] = msg
        self.flush()
        exit()

    def flush(self):
        json_response = json.dumps(self.response)
        print(json_response)