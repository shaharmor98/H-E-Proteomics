import logging


class Logger:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        # Create a logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        formatter.datefmt = '%Y-%m-%d %H:%M:%S'

        # Create a stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def info(self, msg):
        self.logger.info(msg)
