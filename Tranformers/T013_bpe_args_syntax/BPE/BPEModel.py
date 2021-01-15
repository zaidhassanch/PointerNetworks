import torch
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.data.functional import load_sp_model
import pickle

def printSample(text, trn, example_id):
	print(text)
	ex = trn.examples[example_id].src
	dec_ex = sp_gec.decode(ex)
	print(ex)
	print(dec_ex)

	print("Correction1:")
	ex = trn.examples[example_id].trg
	dec_ex = sp_gec.decode(ex)
	print(ex)
	print(dec_ex)

	# print("Correction2:")
	# ex = trn.examples[example_id].correction2
	# dec_ex = sp_gec.decode(ex)
	# print(ex)
	# print(dec_ex)
	# exit()


#tokenize = lambda x: x.split()
# sp_gec_orig = load_sp_model("GCEBPE30k.model")
#
# data1 = {'sp_gec_orig': sp_gec_orig}
# output = open('data.pkl', 'wb')
# pickle.dump(data1, output)

pkl_file = open('data.pkl', 'rb')
data1 = pickle.load(pkl_file)
sp_gec = data1["sp_gec_orig"]

print("print(len(sp_gec)) 1", len(sp_gec))
print("\n=====================")
# print(vars(sp_gec))
# print(dir(sp_gec))
print("\n===============")
print(sp_gec.bos_id(), sp_gec.eos_id(), sp_gec.pad_id())
# exit()
SRC = Field(use_vocab = False, tokenize = sp_gec.encode, 
	init_token = sp_gec.bos_id(), eos_token = sp_gec.eos_id(), pad_token = sp_gec.pad_id(), batch_first = True)
#noSW = Field(use_vocab = True, tokenize = tokenize,  init_token='<sos>', eos_token='<eos>', lower = True)

# tv_datafields = [("orig", SRC), ("correction1", SRC), ("correction2", SRC)]
tv_datafields = [("src", SRC), ("trg", SRC)]

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("before")
#trn, tst = TabularDataset.splits(path = ".", train = "train300k.tsv", test = "test10k.tsv", format='tsv', skip_header=False, fields = tv_datafields)
trn, tst = TabularDataset.splits(path = ".", train = "test2016.tsv", test = "test2016.tsv", format='tsv', skip_header=False, fields = tv_datafields)

#trn, tst = TabularDataset.splits(path = "../2017-data/", train = 'trainBPE.tsv', test = "testBPE.tsv", format='tsv', skip_header=False, fields = tv_datafields)
print("tablular loaded")
#len(SRC.vocab)
BATCH_SIZE = 2
# train_iterator, test_iterator = BucketIterator.splits((trn, tst), batch_size = BATCH_SIZE, sort_within_batch = True, sort_key = lambda x : len(x.orig), device = device)
# train_iterator, test_iterator = BucketIterator.splits((tst, tst), batch_size = BATCH_SIZE, sort_within_batch = True, sort_key = lambda x : len(x.orig), device = device)
train_iterator, test_iterator = BucketIterator.splits((trn, tst), batch_size = BATCH_SIZE, device = device)
example_id = 0

printSample("Orig", trn, example_id)

print("GEC-BPE Len:")
print("print(len(sp_gec)) 2", len(sp_gec)) #, len(SRC.vocab)
# print("noSW Vocab Len:")
# print(len(noSW.vocab))

count = 0
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
for idx, batch in enumerate(train_iterator):
	print(batch.src[0])
	v = sp_gec.decode(batch.src[0].tolist())
	print(v)
	print(batch.src[1])
	v = sp_gec.decode(batch.src[1].tolist())
	print(v)
	print(">>>>", sp_gec.encode(v))
	count += 1
	break





# {'orig': [807, 4538, 6400, 880, 65, 20, 24507, 439, 1707, 4, 7669, 3963, 4, 4877, 4042, 4, 8, 23344, 6128, 6150, 4, 13, 7595, 8194, 2579, 327, 164, 20837, 2084, 4, 2094, 941, 5, 192, 463, 240, 6450, 3234, 9, 787, 3800, 424, 963, 10, 11, 1358, 1656, 18198, 40, 4326, 11, 10683, 5699, 454, 1667, 6], 
# 'correction1': [27588, 37, 6400, 725, 65, 20, 24507, 439, 1707, 4, 11, 7669, 3963, 4, 11, 4877, 4042, 8, 23344, 6128, 6150, 4, 2024, 8, 8194, 2579, 327, 1768, 941, 5, 192, 463, 240, 6450, 3234, 176, 40, 11, 2135, 9, 787, 3800, 424, 963, 10, 11, 1358, 1656, 18198, 6, 56, 192, 1925, 21, 9, 5370, 11, 10683, 5699, 454, 1667, 6], 
# 'correction2': ['utilising', 'hardware', 'tool', 'moisture', 'sensor', 'solar', 'module', 'lee', 'created', 'project', 'aimed', 'reduce', 'agricultural', 'water', 'consumption', 'vineyard', 'project', 'went', 'win', 'competition']}



# 	{'this': <Swig Object of type 'sentencepiece::SentencePieceProcessor *' at 0x7f0843b9bf30>, 
# 	'_out_type': <class 'int'>, 
# 	'_add_bos': False, 
# 	'_add_eos': False, 
# 	'_reverse': False, 
# 	'_enable_sampling': False, 
# 	'_nbest_size': -1, 
# 	'_alpha': 0.1}
# ['Decode', 
# 'DecodeIds', 
# 'DecodeIdsAsSerializedProto', 
# 'DecodeIdsAsSerializedProtoWithCheck', 
# 'DecodeIdsWithCheck', 
# 'DecodePieces', 
# 'DecodePiecesAsSerializedProto', 
# 'Detokenize', 
# 'Encode', 
# 'EncodeAsIds', 
# 'EncodeAsPieces', 
# 'EncodeAsSerializedProto', 
# 'GetEncoderVersion', 
# 'GetPieceSize', 
# 'GetScore', 
# 'IdToPiece', 
# 'Init', 
# 'IsByte', 'IsControl', 'IsUnknown', 'IsUnused', 'Load', 'LoadFromFile', 'LoadFromSerializedProto', 
# 'LoadVocabulary', 'NBestEncodeAsIds', 'NBestEncodeAsPieces', 'NBestEncodeAsSerializedProto', 'PieceToId', 
# 'ResetVocabulary', 'SampleEncodeAsIds', 'SampleEncodeAsPieces', 'SampleEncodeAsSerializedProto', 'SetDecodeExtraOptions', 
# 'SetEncodeExtraOptions', 'SetEncoderVersion', 'SetVocabulary', 'Tokenize', '__class__', '__delattr__', 
# '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__',
#  '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', 
#  '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', 
#  '__swig_destroy__', '__weakref__', '_add_bos', '_add_eos', '_alpha', '_enable_sampling', '_nbest_size', '_out_type', '_reverse', 
#  'bos_id', 'decode', 'decode_ids', 'decode_ids_as_serialized_proto', 'decode_ids_as_serialized_proto_with_check', 'decode_ids_with_check', 
#  'decode_pieces', 'decode_pieces_as_serialized_proto', 'detokenize', 'encode', 'encode_as_ids', 'encode_as_pieces',
#   'encode_as_serialized_proto', 'eos_id', 'get_encoder_version', 'get_piece_size', 'get_score', 'id_to_piece', 'init', 'is_byte', 
#   'is_control', 'is_unknown', 'is_unused', 'load', 'load_from_file', 'load_from_serialized_proto', 'load_vocabulary', 'nbest_encode_as_ids', 
#   'nbest_encode_as_pieces', 'nbest_encode_as_serialized_proto', 'pad_id', 'piece_size', 'piece_to_id', 'reset_vocabulary', 'sample_encode_as_ids', 
#   'sample_encode_as_pieces', 'sample_encode_as_serialized_proto', 'serialized_model_proto', 'set_decode_extra_options', 'set_encode_extra_options', 
#   'set_encoder_version', 'set_vocabulary', 'this', 'thisown', 'tokenize', 'unk_id', 'vocab_size']















