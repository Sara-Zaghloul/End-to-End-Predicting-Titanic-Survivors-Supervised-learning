import os

class InvalidInputError(Exception):
    def __init__(self, message = "The input provided is invalid."):
        self.message = message
        super().__init__(self.message)