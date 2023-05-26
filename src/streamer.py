class CallbackStreamer:
    def __init__(self, token_callback, end_callback):
        self.tokens = []
        self._token_callback = token_callback
        self._end_callback = end_callback
    def put(self, value):
        val = value.tolist()[0]
        if not isinstance(val, list):
            val = [val]
        self.tokens.extend(val)
        self._token_callback(self.tokens)
    def end(self):
        self._end_callback(self.tokens)