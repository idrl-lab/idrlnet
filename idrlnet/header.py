"""Initialize public objects"""

import logging
import functools

DIFF_SYMBOL = "__"


class TestFun:
    registered = []

    def __init__(self, fun):
        self.fun = fun
        self.registered.append(self)

    def __call__(self, *args, **kwargs):
        print(str(self.fun.__name__).center(50, "*"))
        self.fun()

    @staticmethod
    def run():
        for fun in TestFun.registered:
            fun()


def testmemo(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        if id(fun) not in testmemo.memo:
            logger.info(f"'{fun}' needs tests")
            testmemo.memo.add(id(fun))
        fun(*args, **kwargs)

    return wrapper


testmemo.memo = set()

log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
handlers = [logging.FileHandler("train.log", mode="a"), logging.StreamHandler()]
logging.basicConfig(
    format=log_format,
    level=logging.INFO,
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=handlers,
)
logger = logging.getLogger(__name__)
