class Character:
    def __init__(self, firstGuessClass, firstGuess, firstGuessConfidentLvl,
                 secondGuessClass, secondGuess, secondGuessConfidentLvl):
        self._firstGuessClass = firstGuessClass
        self._firstGuess = firstGuess
        self._firstGuessConfidentLvl = firstGuessConfidentLvl
        self._secondGuessClass = secondGuessClass
        self._secondGuess = secondGuess
        self._secondGuessConfidentLvl = secondGuessConfidentLvl

    @property
    def firstGuessClass(self):
        return self._firstGuessClass

    @firstGuessClass.setter
    def firstGuessClass(self, firstGuessClass):
        self._firstGuessClass = firstGuessClass

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
    def secondGuessClass(self):
        return self._secondGuessClass

    @secondGuessClass.setter
    def secondGuessClass(self, secondGuessClass):
        self._secondGuessClass = secondGuessClass

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
