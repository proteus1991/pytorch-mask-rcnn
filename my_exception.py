class myerr(BaseException):
    def __init__(self,err):
        Exception.__init__(self)
        self.err = err