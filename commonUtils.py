import json


class configParams:

    def __init__(self):
        #configFile = "..\\config.json"
        configFile = "config.json"
        with open(configFile, "r") as config:
            self.data = json.load(config)

    def loadConfig(self):
        return self.data
