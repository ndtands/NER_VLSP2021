from pyvi import ViTokenizer, ViPosTagger
import re

def preprocess_email_url(datas):
  datas_trained = []
  for i in range(len(datas)):
    data = datas[i]

    if data[1] == 'EMAIL':
      check = is_Email(data[0])
      if len(check) == 0:
        data = (data[0], 'O')
    
    if data[1] != 'EMAIL' and  data[1] != 'URL': #(url, org, loc, o,.....)
      check = is_Email(data[0])
      if len(check) > 0:
        data = (data[0], 'EMAIL')


    if data[1] == "URL":
      check = is_URL(data[0])
      if len(check) > 0 and  check[0][1] - check[0][0] == len(data[0]):
        data = (data[0], 'URL')
      else: 
        data = (data[0], 'O')
      
    try:
      if data[1] != 'URL' and data[1] != 'EMAIL':
        check = is_URL(data[0])
        if len(check) > 0 and  check[0][1] - check[0][0] == len(data[0]):
          data = (data[0], 'URL')
    except:
      print(check)
    datas_trained.append(data)
  return datas_trained
    
def merge_word(sent, pred_out):
  '''
    :sent: is input sentences (hanlded pre-processing). example: 'pham van manh have email ( pvm26042000@gmail.com ) ....'
    :out : is input of predict, is list tuple. example: [('pham', 'O'), ('van', 'O'), ('manh', 'O'), ('have', 'O'), ('email', 'O'), ('(', 'O'),  ('pvm26042000', 'EMAIL'), ('@', 'EMAIL'),('gmail', 'EMAIL'), ('.', 'EMAIL'),('com', 'EMAIL'),(')', 'O'),('....', 'O')]
  '''
  out_merged = []
  parts = sent.split()

  for index in range(0, len(parts)):
    word = parts[index]

    
    for jndex in range(1, len(pred_out) + 1):
      token = pred_out[0:jndex]
      ws_token, ls_token = list(zip(*token))
      word_token = "".join(ws_token)
      if word_token == word:
        if len(token) == 1:
          out_merged.append(token[0])
        elif len(token) > 1:
          a, b = list(zip(*token))
          word_merged = "".join(a)
          l_merged = decide_label((word_merged, b))
          out_merged.append(l_merged)
        pred_out = pred_out[jndex:]
        break
  return out_merged

def post_processing(origin_sentence, out_predict):

  out_merged = merge_word(origin_sentence, out_predict)
  datas_trained = post_process_email_url(out_merged)

  gr_indexs = cluster(datas_trained, 3)
  if len(gr_indexs) > 0:
    for index in gr_indexs:
      string, label = list(zip(*datas_trained[index[0]: index[-1] + 1]))
      if is_ADDRESS(string, label) == True:
        for i in range(index[0], index[-1] + 1):
          datas_trained[i] =(datas_trained[i][0], "ADDRESS")
      else:
        for i in range(index[0], index[-1] + 1):
          if datas_trained[i][0] in ',':
            datas_trained[i] = (datas_trained[i][0], "O")
          else:
            datas_trained[i] =(datas_trained[i][0], "LOCATION")
  return datas_trained

def decide_label(part):
  word = part[0]
  labels = part[1]
  return (word, max(labels))


import re
def constain_alpha(token):
  for character in token:
    is_letter = character.isalpha()
    if is_letter == True:
      return True
  return False

def is_URL(token):
    black_list = [".exe",".txt", ".jpg", ".png", ".mp3 "]
    token = token.lower()
    index = 0
    indexs = []

    for tk in black_list:
      if tk in token:
        return indexs

    if constain_alpha(token) == True:
      domain = re.findall(r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b', token)
      
      if len(domain) != 0:
          index_start_domain = token.find(domain[0]) + index
          if token.find(domain[0]) == 0:
              index_end_domain = index_start_domain + len(token)
          else:
              index_end_domain = index_start_domain + len(domain[0])
          indexs.append((index_start_domain, index_end_domain))
      index += len(token) + 1
    return indexs

def is_Email(token):
    index = 0
    indexs = []
    for word in token.split(" "):
        emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", word)
        if len(emails) != 0:
            index_start_email = word.find(emails[0]) + index
            
            index_end_email = index_start_email + len(emails[0])
            
            indexs.append((index_start_email, index_end_email))
        index += len(word) + 1
    return indexs

def is_IP(token):
  ipv4 = r"[0-9*a-z]{1,3}\.[0-9*a-z]{1,3}\.[0-9*a-z]{1,3}\.[0-9*a-z]{1,3}\:*[\d]*"
  ipv4Bi = r"[0-1*a-z]{8}\.[0-1*a-z]{8}\.[0-1*a-z]{8}\.[0-1*a-z]{8}\:*[\d]*"
  ipv6 = r'[A-Fa-f0-9:]+\:+[A-Fa-f0-9]+'
  index = 0
  indexs = []
  for word in token.split(" "):
      emails = re.findall(ipv4 + '|' + ipv4Bi + '|' + ipv6, word)
      if len(emails) != 0:
          index_start_email = word.find(emails[0]) + index
          
          index_end_email = index_start_email + len(emails[0])
          
          indexs.append((index_start_email, index_end_email))
      index += len(word) + 1
  return indexs

def post_process_email_url(datas):
  black_word = ["tp.hcm"]
  datas_trained = []
  flag = False
  for i in range(len(datas)):
    data = datas[i]

      # check predict email
    if data[1] == 'EMAIL':
        check = is_Email(data[0])
        if len(check) == 0:
          check = is_URL(data[0])
          if len(check) == 0 or  check[0][1] - check[0][0]!= len(data[0]):
            data = (data[0], 'O')
          else:
            data = (data[0], 'URL')
    
    elif data[1] == 'URL':
        check = is_URL(data[0])
        if len(check) == 0 or  check[0][1] - check[0][0]!= len(data[0]):
          data = (data[0], 'O')
    
    elif data[1] == 'IP':
        check = is_IP(data[0])
        if len(check) == 0 or  check[0][1] - check[0][0]!= len(data[0]):
          if data[0].isalnum():
            data = (data[0], 'QUANTITY')
          else:
            data = (data[0], 'O')
    datas_trained.append(data)
  return datas_trained

def cluster(data, maxgap):
  '''Arrange data into groups where successive elements
      differ by no more than *maxgap*
      >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
      [[1, 6, 9], [100, 102, 105, 109], [134, 139]]
      >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
      [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]
  '''
  black_list = [":", "(", ";", "{", "[", "và", "tại", "ở", "của"]

  indexs = []
  for index in range(len(data)):
    token = data[index]
    if token[1] == "LOCATION" or token[1] == "ADDRESS" :
      indexs.append(index)

  if len(indexs) == 0:
    return list()

  indexs.sort()
  groups = [[indexs[0]]]

  for jndex in range(1,len(indexs)):
    x  = indexs[jndex]
    w, labels = list(zip(*data[indexs[jndex-1]:x]))
    if abs(x - groups[-1][-1]) <= maxgap and any(character in w for character in black_list) == False:
        groups[-1].append(x)
    elif any(character in data[indexs[jndex-1]:x] for character in black_list):
        groups.append([x])
    else:
        groups.append([x])
  return groups
  

def has_numbers(inputString):
  parts = inputString.split()

  for i in range(len(parts)):
    part = parts[i]
    for char in part:
      if char.isdigit():
        if i > 0 and parts[i-1].lower() in ['cn', "tổ", "quận", "q.", 'p.', 'phường']:
          return False
        else:
          return True
  return False

def is_ADDRESS(string, label):
  index_not_dau_phay = [i for i, e in enumerate(label) if e == "O"]

  uy_tin = 0
  string_loc = " ".join(string)
  if 'ADDRESS' in label:
    uy_tin += 0.15
    #print('label address', uy_tin)

  if has_numbers(string_loc):
    uy_tin += 0.15
    #print('has numbers', uy_tin)

  for i in index_not_dau_phay:
      if string[i] not in [",", "-"]:
        uy_tin -= 0.05
        #print('not in ,-', uy_tin)
      else:
        if string[i] == ",":
          uy_tin += 0.02
          #print('==,', uy_tin)
        if string[i] == "-":
          uy_tin += 0.05
          #print('== -', uy_tin)
  level = ["toà_nhà", "nhà", "lầu", "tầng", "căn_hộ", "số", "lô", "km","quốc_lộ","đại_lộ","kcn", "đường","tổ", "ngõ", "toà", "ngách", "hẻm","kiệt", "chung_cư", "ấp" ,"thôn", "khu","phố" , "quận", "phường", "xã", "thị_xã","huyện", "thành_phố", "tp", "tỉnh" ]
  level_0 ={'status': True,'keywords': ["toà", "toà_nhà", "nhà", "lầu", "tầng", "căn_hộ", "chung_cư", "số", "lô", "kcn", "km", "quốc_lộ", "đại_lộ"] }
  level_1 = {'status': True, 'keywords': [ "ngõ", "ngách", "hẻm","kiệt",]}
  level_2 = {'status': True, 'keywords':["ấp" ,"thôn", "khu","phố" , "quận", "phường", "xã", "tổ", "dân_phố", "đường"]}
  level_3 = {'status': True,'keywords':["thị","huyện"]}
  level_4 = {'status': True,'keywords':["thành_phố", "tp", "tỉnh"]}

  punct = ',;.-'

  parts =  ViPosTagger.postagging(ViTokenizer.tokenize(string_loc))[0]

  for i, seg_word in enumerate(parts):
    if seg_word.lower() in level:
      if seg_word.lower() in level_0['keywords'] and level_0['status'] == True and i < len(parts) - 1 and parts[i+1] not in punct:
        uy_tin += 0.25
        level_0['status'] = False
        #print('level0', uy_tin)

      elif seg_word.lower() in level_1['keywords'] and level_1['status'] == True and i < len(parts) - 1 and parts[i+1] not in punct:
        uy_tin += 0.075
        level_1['status'] = False
        #print('level1', uy_tin)

      elif seg_word.lower()  in level_2['keywords'] and level_2['status'] == True and i < len(parts) - 1 and parts[i+1] not in punct:
        uy_tin += 0.025
        level_2['status'] = False
        #print('level2', uy_tin)
      
      elif seg_word.lower() in  level_3['keywords'] and level_3['status'] == True and i < len(parts) - 1 and parts[i+1] not in punct:
        uy_tin += 0.015
        level_3['status'] = False
        #print('level3', uy_tin)
      
      elif seg_word.lower() in level_4['keywords'] and level_4['status'] == True and i < len(parts) - 1 and parts[i+1] not in punct:
        uy_tin += 0.01
        level_4['status'] = False
        #print('level4', uy_tin)

  #print(uy_tin)
  return uy_tin >= 0.3

def post_processing(origin_sentence, out_predict):

  out_merged = merge_word(origin_sentence, out_predict)
  datas_trained = post_process_email_url(out_merged)

  gr_indexs = cluster(datas_trained, 3)
  if len(gr_indexs) > 0:
    for index in gr_indexs:
      if len(index) > 1:
        string, label = list(zip(*datas_trained[index[0]: index[-1] + 1]))
        set_label = set(label)
        if len(set_label) >= 2 and (set_label != {'O', 'LOCATION'} and set_label != {'PERSONTYPE', 'LOCATION'}): 
          #print(string)
          if is_ADDRESS(string, label) == True:
            for i in range(index[0], index[-1] + 1):
              datas_trained[i] =(datas_trained[i][0], "ADDRESS")
          else:
            for i in range(index[0], index[-1] + 1):
              #print(datas_trained[i])
              if datas_trained[i][0] in ',':
                datas_trained[i] = (datas_trained[i][0], "O")
              elif datas_trained[i] not in ['-/']:
                datas_trained[i] =(datas_trained[i][0], "LOCATION")
  return datas_trained

def span_cluster(dts, pred=1):
    sent = list(dts)
    sent[0] = (sent[0][0], sent[0][pred])
    sent[-1] = (sent[-1][0], sent[-1][pred])
    for i in range(1, len(sent)-1):
        if sent[i-1][pred] in ['SKILL'] and (sent[i-1][pred] == sent[i+1][pred] or i == len(sent)-2 or sent[i-1][pred] == sent[i+2][pred]) and sent[i][pred] != sent[i-1][pred] and sent[i][-1][2] >= 0.1:
            sent[i] = (sent[i][0], sent[i-1][pred])
        else:
            sent[i] = (sent[i][0], sent[i][pred])
    return sent