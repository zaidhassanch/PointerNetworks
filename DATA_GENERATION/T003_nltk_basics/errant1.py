import errant
#
# annotator = errant.load('en')
# orig = annotator.parse('This are gramamtical sentence .')
# cor = annotator.parse('This is a grammatical sentence .')
# edit = [1, 2, 1, 2, 'SVA'] # are -> is
# edit = annotator.import_edit(orig, cor, edit)
# print(edit.to_m2())


annotator = errant.load('en')
orig = annotator.parse('This are gramamtical sentence .')
cor = annotator.parse('This is a grammatical sentence .')
alignment = annotator.align(orig, cor)
edits = annotator.merge(alignment)
for e in edits:
    e = annotator.classify(e)
    print(e)