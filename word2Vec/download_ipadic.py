import fugashi

tagger = fugashi.Tagger()
print([word.surface for word in tagger("私は猫です")])
