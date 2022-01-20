LABEL_O = 'O'
LABEL_PAD = 'PAD'
TOKEN_START = '<s>'
TOKEN_END = '</s>'
TOKEN_BREAK = '</x>'
PUNCT_END = '.!?'
PUNCT_SPEC = '.,\\|'
FLAG_STRICT ={
    'MAX': 2,
    'MEDIUM': 1,
    'MIN': 0
}

COLORS ={
    'EMAIL':'#FDEE00',
    'ADDRESS':'#C32148',
    'PERSON':'#FE6F5E',
    'PHONENUMBER': '#9F8170',
    'MISCELLANEOUS':'#007BA7',
    'QUANTITY':'#D891EF',
    'PERSONTYPE':'#FF91AF',
    'ORGANIZATION':'#3DDC84',
    'PRODUCT':'#FBCEB1',
    'SKILL':'#B0BF1A',
    'IP':'#703642',
    'LOCATION':'#C0E8D5',
    'DATETIME':'aqua',
    'EVENT':'darkorange',
    'URL':'#BD33A4'
}

NER_COLOR = list(COLORS.keys())
OPTIONS = {'ents': NER_COLOR, 'colors': COLORS}

PATH_CONFIG = 'config.json'
PATH_MODEL = 'model/xlmr_span2_10t01_pool_nocat.pt'