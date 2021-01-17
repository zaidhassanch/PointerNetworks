
import pickle
pkl_file = open('BPE/data.pkl', 'rb')
data1 = pickle.load(pkl_file)
sp_gec = data1["sp_gec_orig"]

print(sp_gec)

sentence = "A trendy girl talking on her cellphone while gliding slowly down the street"
s_enc =  sp_gec.encode("dy")
s_arr = sentence.split()
print(s_enc)
print(s_arr)
for word in s_arr:
    print(word, sp_gec.encode(word))

print(sp_gec.decode(149))