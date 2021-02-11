class Character:
    def __init__(self, firstGuess, firstGuessConfidentLvl, secondGuess, secondGuessConfidentLvl):
        self._firstGuess = firstGuess
        self._firstGuessConfidentLvl = firstGuessConfidentLvl
        self._secondGuess = secondGuess
        self._secondGuessConfidentLvl = secondGuessConfidentLvl

    @property
    def firstGuess(self):
        return self._firstGuess

    @firstGuess.setter
    def firstGuess(self, firstGuess):
        self._firstGuess = firstGuess

    @property
    def firstGuessConfidentLvl(self):
        return self._firstGuessConfidentLvl

    @firstGuessConfidentLvl.setter
    def firstGuessConfidentLvl(self, firstGuessConfidentLvl):
        self._firstGuessConfidentLvl = firstGuessConfidentLvl

    @property
    def secondGuess(self):
        return self._secondGuess

    @secondGuess.setter
    def secondGuess(self, secondGuess):
        self._secondGuess = secondGuess

    @property
    def secondGuessConfidentLvl(self):
        return self._secondGuessConfidentLvl

    @secondGuessConfidentLvl.setter
    def secondGuessConfidentLvl(self, secondGuessConfidentLvl):
        self._secondGuessConfidentLvl = secondGuessConfidentLvl
