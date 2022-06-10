import yaml
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys
import os
import gzip
import errno
import tarfile


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def mkdir_p(path):
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise


class CompressedRotatingFileHandler(RotatingFileHandler):
    def __init__(
        self, filename, mode="a", maxBytes=0, backupCount=0, encoding=None, delay=0
    ):
        mkdir_p(os.path.dirname(filename))
        RotatingFileHandler.__init__(
            self, filename, mode, maxBytes, backupCount, encoding, delay
        )

    def doRollover(self):
        self.stream.close()
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = "%s.%d.gz" % (self.baseFilename, i)
                dfn = "%s.%d.gz" % (self.baseFilename, i + 1)
                if os.path.exists(sfn):
                    # print "%s -> %s" % (sfn, dfn)
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = self.baseFilename + ".1.gz"
            if os.path.exists(dfn):
                os.remove(dfn)

            try:
                make_tarfile(dfn, self.baseFilename)
            except:
                if not os.path.exists(dfn):
                    if os.path.exists(self.baseFilename):
                        os.rename(self.baseFilename, dfn)
            finally:
                if os.path.exists(self.baseFilename):
                    os.remove(self.baseFilename)
        self.mode = "w"
        self.stream = self._open()


class ErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno > logging.INFO


class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno <= logging.INFO


if __name__ == "__main__":
    import logging.config

    with open("./logging_http.yaml", "r") as f:
        log_cfg = yaml.safe_load(f.read())
        logging.config.dictConfig(log_cfg)
    logging.info("info")
    logging.warn("warn")
    logging.error("error")

    make_tarfile("./logger/http/error.log.1.gz", "./logger/http/error.log")
