import unittest

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover('tests')
    ret = not unittest.TextTestRunner().run(suite).wasSuccessful()
    sys.exit(ret)
