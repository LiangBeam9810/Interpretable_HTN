import sys
import os
 
class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

if __name__ == '__main__':
    sys.stdout = Logger()
    print(1233211234567)
