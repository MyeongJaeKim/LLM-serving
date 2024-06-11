import logging
import logging.handlers

import time

class FileLogger():
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # log format
        formatter = logging.Formatter('%(message)s')

        #streamHandler = logging.StreamHandler()
        fileHandler = logging.FileHandler('logs/log.out', encoding='utf-8')

        #streamHandler.setFormatter(formatter)
        fileHandler.setFormatter(formatter)

        #self.logger.addHandler(streamHandler)
        self.logger.addHandler(fileHandler)

        self.logger.propagate = False

    def log(self, msg: str) -> None:
        self.logger.info(msg)

file_logger = FileLogger()

def log(user_id, msg):
    current_epoch_milliesecond = int(time.time() * 1000)

   # Format :
   # 111111 || Asked  || Prompt
   # 111112 || Token1                      --> diff with first one to get TTFT (time to first token)
   # 111113 || Token2
   # ...
   # 111120 || Fully received || Prompt
   #  TPOT : average of time differences of each token
   # Latency : time difference between the first and the lastest token

    log_msg: str = f'{current_epoch_milliesecond} || {user_id} || {msg}'

    file_logger.log(log_msg)
    