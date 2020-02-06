# -*- coding:utf-8 -*-
"""
This is whole project logging functionality that could be used.

author: Guangqiang.lu
"""

import logging
import tempfile
import shutil
import os
from datetime import datetime


class Logger(object):
    def __init__(self, logger_name=None):
        if logger_name is None:
            logger_name = "logging"   # we could change this.
        self.logger_name = logger_name.split('/')[-1] if logger_name is not None else logger_name
        self._logger_path = os.path.join(tempfile.gettempdir(), 'logging')
        if not os.path.exists(self._logger_path):
            try:
                os.mkdir(self._logger_path)
            except IOError as e:
                raise IOError("When try to create logging folder with error:%s" % e)

        self.logger = logging.getLogger(self.logger_name) if logger_name is not None else __file__
        self.logger.setLevel(logging.INFO)
        # this is to write logging info to disk
        # I just want to ensure there is only one logging file in current folder,
        # after the whole process finish, I just upload the file to HDFS for later use case
        self.logger_file_name = 'logging_%s.log' % datetime.now().strftime('%Y%m%d')
        self.logger_file_path = os.path.join(self._logger_path, self.logger_file_name)

        file_handler = logging.FileHandler(self.logger_file_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        # this is to write the logging info to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.addHandler(file_handler)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def exception(self, msg):
        self.logger.exception(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def delete(self):
        logger_file_path = os.path.join(self._logger_path, self.logger_name)
        if os.path.exists(logger_file_path):
            try:
                os.remove(logger_file_path)
            except:
                pass


def create_logger(logger_name=None):
    logger = Logger(logger_name)

    return logger


if __name__ == '__main__':
    logger = create_logger(__file__)
    logger.info("test")
    logger.delete()

