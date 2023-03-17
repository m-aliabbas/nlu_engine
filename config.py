import torch
config = dict()

#---------------------- general config ---------------------------------------
config["cuda_device"] = torch.device("cuda:0" if torch.cuda.is_available() 
                                     else "cpu")
config["local_flag"] = True 
config["qa_engine_type"] = "MobileBertSQ2" #Mobile Bert Trained on Squad Q/A Dataset v2
config["classifier_type"] = "MobileBertZS" # Mobile Bert trained on Zeroshot learning fashion

#-------------------  NLI Config ---------------------------------------------
config["nli"] = dict()

#------------------------ QA Engine Config -----------------------------------

if config["qa_engine_type"] == "MobileBertSQ2":
    config["qa_engine"] = dict()
    config["qa_engine"]["model_path"] = "aware-ai/mobilebert-squadv2"

#-------------------  Classifer Config ---------------------------------------

if config["classifier_type"] == "MobileBertZS":
    config["classifier"] = dict()
    config["classifier"]["model_path"] = "typeform/mobilebert-uncased-mnli"




