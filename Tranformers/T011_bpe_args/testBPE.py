import pickle

pkl_file = open('BPE/data.pkl', 'rb')
data1 = pickle.load(pkl_file)
sp_gec = data1["sp_gec_orig"]

encoded_text = sp_gec.encode("Hello world.")

decoded_text = sp_gec.decode(encoded_text)