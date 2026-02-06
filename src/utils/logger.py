import logging
import os

def get_logger(name: str) -> logging.Logger:


 logger = logging.getLogger(name)
 logger.setLevel(logging.INFO)
 
 formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
 
 #create console handler and set level to Info
 ch = logging.StreamHandler()
 ch.setFormatter(formatter)
 logger.addHandler(ch)
 
#create file handler and set level to debug
 os.makedirs('logs', exist_ok = True)
 fh = logging.FileHandler('logs/app.log')
 fh.setLevel(logging.DEBUG)
 fh.setFormatter(formatter)
 logger.addHandler(fh)
 
 return logger