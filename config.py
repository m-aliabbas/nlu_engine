import torch
config = dict()

#---------------------- general config ---------------------------------------
config["cuda_device"] = torch.device("cuda:0" if torch.cuda.is_available() 
                                     else "cpu")
config["local_flag"] = True 
# either MobileBertSQ2 or DistilBertCaseSquad
config["qa_engine_type"] = "DistilBertCaseSquad" #Mobile Bert Trained on Squad Q/A Dataset v2
# either MobileBertZS or  DistilRobertaBaseZS
config["classifier_type"] = "DistilRobertaBaseZS" # Mobile Bert trained on Zeroshot learning fashion

#-------------------  NLI Config ---------------------------------------------
config["nli"] = dict()

#------------------------ QA Engine Config -----------------------------------
#------------------------- MobileBertSquad2 ----------------------------------
if config["qa_engine_type"] == "MobileBertSQ2":
    config["qa_engine"] = dict()
    config["qa_engine"]["model_path"] = "aware-ai/mobilebert-squadv2"
#---------------------- DistilBertCaseSquad ----------------------------------
if config["qa_engine_type"] == "DistilBertCaseSquad":
    config["qa_engine"] = dict()
    config["qa_engine"]["model_path"] = "distilbert-base-cased-distilled-squad"

#-------------------  Classifer Config ---------------------------------------
#------------------------- Mobile Bert ---------------------------------------
if config["classifier_type"] == "MobileBertZS":
    config["classifier"] = dict()
    config["classifier"]["model_path"] = "typeform/mobilebert-uncased-mnli"


#---------------------- Distil Robert ----------------------------------------
if config["classifier_type"] == "DistilRobertaBaseZS":
    config["classifier"] = dict()
    config["classifier"]["model_path"] = "cross-encoder/nli-distilroberta-base"
