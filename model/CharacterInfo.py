class CharacterInfo:
    def __init__(self, character, name, unicode, phonetic, group, description, audio):
        self._character = character
        self._name = name
        self._unicode = unicode
        self._phonetic = phonetic
        self._group = group
        self._description = description
        self._audio = audio

    @property
    def character(self):
        return self._character

    @character.setter
    def character(self, character):
        self._character = character

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def unicode(self):
        return self._unicode

    @unicode.setter
    def unicode(self, unicode):
        self._unicodes = unicode

    @property
    def phonetic(self):
        return self._phonetic

    @phonetic.setter
    def phonetic(self, phonetic):
        self._phonetic = phonetic

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, group):
        self._group = group

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @property
    def audio(self):
        return self._audio

    @audio.setter
    def audio(self, audio):
        self._audio = audio
