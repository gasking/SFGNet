#--------------------------#
# 日志文件
#--------------------------#
import logging
import time
class Logger():
    def __init__(self,**kwargs):
        if 'file_path' not in kwargs.keys():
           kwargs['file_path'] = 'logs/' + time.strftime("%Y-%m-%d-%H",time.localtime()) + '.log'
        if 'level'not in kwargs.keys():
           kwargs[ 'level' ] = logging.INFO
        if 'format' not in kwargs.keys():
           kwargs[ 'format' ] = '%(asctime)s - %(filename)s - %(lineno)d - %(funcName)s - %(message)s'
        self.logger = logging.getLogger(name = kwargs['file_path'] + '.log')
        self.logger.setLevel(kwargs['level'])

        # 设置写入文件格式
        handler = logging.FileHandler(kwargs['file_path'])
        formatter = logging.Formatter(kwargs['format'])
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def info(self,msg):
        self.logger.info(msg)

    def warning(self,msg):
        self.logger.warning(msg)

    def debug(self,msg):
        self.logger.debug(msg)

    def critical(self,msg):
        self.logger.critical(msg)

    def error(self,msg):
        self.logger.error(msg)



if __name__ == '__main__':
    parameters = {
        'file_path':'record.log',
        'name':'test',
        'level':logging.INFO,
        'format':'%(asctime)s - %(filename)s - %(lineno)d - %(funcName)s - %(message)s'
    }
    logger = Logger(**parameters)

    logger.info("这是一个测试文件")

    logger.warning("发生警告")

    logger.error('这是一个错误')

    logger.critical("这是一个严重问题")

    logger.debug('这是一个调试')




