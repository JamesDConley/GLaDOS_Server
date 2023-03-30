class StopOnStr:
    def __init__(self, stop_str, tokenizer):
        """Class to create a custom stop condition for transformers generation

        Args:
            stop_str (str): String to halt generation on
            tokenizer (galactica tokenizer): Tokenizer to convert string to tokens for generation checking
        """
        self.stop_ids = tokenizer(stop_str, add_special_tokens=False).input_ids
    def __call__(self, input_ids, score=None):
        return self.stop_on_str(input_ids, score=score)

    def stop_on_str(self, input_ids, score=None):
        """Return True if the generated sequence ends with the stop string

        Args:
            input_ids (list): List of integers to check
            score (float): Not used- requried for huggingface Interface though. Defaults to None

        Returns:
            Bool: True if it ends with the loaded string
        """
        for i in range(1, len(self.stop_ids)+1):
            idx = -i
            if int(input_ids[0][idx]) != int(self.stop_ids[idx]):
                return False
        return True
            