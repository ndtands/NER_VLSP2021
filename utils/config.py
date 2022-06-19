import os

LABELS = ["PAD", "ADDRESS","SKILL","EMAIL","PERSON","PHONENUMBER","MISCELLANEOUS","QUANTITY","PERSONTYPE",
        "ORGANIZATION","PRODUCT","IP","LOCATION","O","DATETIME","EVENT", "URL"]

host =  os.getenv("HOST", default= "0.0.0.0")
port =  os.getenv("PORT", default= "5053")
is_prod =  os.getenv("IS_PROD", default= True)

sub = os.getenv("SUB", default= "_")
device = os.getenv("DEVICE", default= "cpu")
max_len = os.getenv("MAX_LEN", default= 256)
batch_size = os.getenv("BATCH_SIZE", default= 32)
label_classes = os.getenv("LABEL_CLASSES", default= LABELS)
model_config =  os.getenv("MODEL_PATH", default= "model/xlmr/config.json")

