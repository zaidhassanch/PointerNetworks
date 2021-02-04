import spacy
import torch
from torchtext.data.metrics import bleu_score
#
# def translate_sentencex(model, sentence, src_field, trg_field, device, max_len=50):
#     model.eval()
#
#     if isinstance(sentence, str):
#         nlp = spacy.load('de')
#         tokens = [token.text.lower() for token in nlp(sentence)]
#     else:
#         tokens = [token.lower() for token in sentence]
#
#     tokens = [src_field.init_token] + tokens + [src_field.eos_token]
#
#     src_indexes = [src_field.stoi[token] for token in tokens]
#
#     src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
#
#     with torch.no_grad():
#         encoder_outputs, hidden = model.encoder(src_tensor)
#
#     mask = model.create_mask(src_tensor)
#
#     trg_indexes = [trg_field.stoi[trg_field.init_token]]
#
#     attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
#
#     for i in range(max_len):
#
#         trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
#
#         with torch.no_grad():
#             output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
#
#         attentions[i] = attention
#
#         pred_token = output.argmax(1).item()
#         trg_indexes.append(pred_token)
#
#         if pred_token == trg_field.stoi[trg_field.eos_token]:
#             break
#
#     trg_tokens = [trg_field.itos[i] for i in trg_indexes]
#
#     return trg_tokens[1:], attentions[:len(trg_tokens) - 1]
#
#
# def translate_sentencex1(model, sentence, src_field, trg_field, device, max_len=50):
#     model.eval()
#
#     if isinstance(sentence, str):
#         nlp = spacy.load('de')
#         tokens = [token.text.lower() for token in nlp(sentence)]
#     else:
#         tokens = [token.lower() for token in sentence]
#
#     tokens = [src_field.init_token] + tokens + [src_field.eos_token]
#
#     src_indexes = [src_field.stoi[token] for token in tokens]
#
#     src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
#
#     with torch.no_grad():
#         encoder_outputs, hidden = model.encoder(src_tensor)
#
#     mask = model.create_mask(src_tensor)
#
#     trg_indexes = [trg_field.stoi[trg_field.init_token]]
#
#     attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
#
#     for i in range(max_len):
#
#         trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
#
#         with torch.no_grad():
#             output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
#
#         attentions[i] = attention
#
#         pred_token = output.argmax(1).item()
#         trg_indexes.append(pred_token)
#
#         if pred_token == trg_field.stoi[trg_field.eos_token]:
#             break
#
#     trg_tokens = [trg_field.itos[i] for i in trg_indexes]
#
#     return trg_tokens[1:], attentions[:len(trg_tokens) - 1]
#
#
#
# def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
#
#     trgs = []
#     pred_trgs = []
#
#     for datum in data:
#
#         src = datum.src
#         trg = datum.trg
#         pred_trg, _ = translate_sentence(model, src, src_field, trg_field, device, max_len)
#         pred_trg = pred_trg[:-1]        #cut off <eos> token
#         pred_trgs.append(pred_trg)
#         trgs.append([trg])
#
#     return bleu_score(pred_trgs, trgs)
#
