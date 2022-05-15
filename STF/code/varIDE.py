#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re
from itertools import product
import os
import sys
import ast
import itertools
from itertools import permutations
import time
import shutil
from shutil import copyfile
from cupt2typo_POS_dep_morpho import * 
from distSyn import * 
from tqdm import tqdm 
import pickle
from STEP1_extractInfos_Train import *
from STEP2_extractCandidats import *
from STEP3_Classif import *
import configparser


#==========================================
#        MAIN
#==========================================

#Parameters in configFile
config = configparser.ConfigParser()
config.read('config.cfg')
langues = ast.literal_eval(config.get('Parameters', 'langues'))
languesFiltre = ast.literal_eval(config.get('Parameters', 'languesFiltre'))
filterMaxInsert = int(config.get('Parameters', 'filterMaxInsert'))
importClassif = config.get('Parameters', 'importClassifier')
for langue in langues :
    #==============================================================
    # STEP 1: Caractéristiques des MWE annotées 
    #==============================================================
    if not os.path.exists(str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue)):
        os.makedirs(str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue))  
        os.makedirs(str(repParent) + "/OUTPUT/EVAL/" + str(langue)) 
    else :
        shutil.rmtree(str(repParent) + "/OUTPUT")
        os.makedirs(str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue))  
        os.makedirs(str(repParent) + "/OUTPUT/EVAL/" + str(langue))         
    Train_cupt = str(repParent) + "/INPUT/DATA_LANG/" + str(langue) + "/train.cupt"
    NFallPatterns = trainCupt2tableur(Train_cupt,langue) 
    #==============================================================
    # STEP 2 : Extract candidates
    #==============================================================
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    t_TRAIN = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/train_MWEs.tsv"
    cuptTRAIN = str(repParent) + "/INPUT/DATA_LANG/" + str(langue) + "/train.cupt"
    cuptTEST = str(repParent) + "/INPUT/DATA_LANG/" + str(langue) + "/test.blind.cupt"
    cuptDEV = str(repParent) + "/INPUT/DATA_LANG/" + str(langue) + "/dev.cupt"  
    # 1/  POSnorm du TOP 10 pour chaque langue 
    #listePOSnormTopX = extractPOSnormTopX(langue)  
    listePOSnormTopX,dicoPOSnorm2Vars = cupt2patternsTopX(NFallPatterns,langue)    
    dicoNF2POSnorm,dicoLemmePOSNF = NF2POSnorm_POS_lemma(t_TRAIN,langue)   
    # 2/  Génération des lemmes possibles   
    dicoGenerLemmes,variantesPOS_NF = NF2Vars(t_TRAIN,dicoPOSnorm2Vars,dicoNF2POSnorm,dicoLemmePOSNF,listePOSnormTopX)
    # 3/ Extraction des MWEs dans TRAIN  + catégories:
    MWEannoteesTrain,nbMWEinTrain,dicoIDIOMAT = idPhrase2MWE(t_TRAIN,listePOSnormTopX)            #'fr-ud-train_04716': ['41|42|43', '54|55']
    dicoNFcategories = NF2Categorie(t_TRAIN)   
    # 4/ Recherche des infos morphoSyntaxiques existantes
    dicoMorpho,listePOS,listeRelDep = cupt2POSdepMorpho(langue) 
    #======== Choix d'un filtre du nombre max d'insertions =================
    t_Idiomat = t_TRAIN     
    # 4/ Recherche des candidats => fichier temporaire avec idphrase/idToks/NF
    for corpus in ["train", "dev","test.blind"] :
        rechercheCandidats(corpus,langue,variantesPOS_NF,dicoNF2POSnorm,dicoGenerLemmes,filterMaxInsert,MWEannoteesTrain)  
    if langue in languesFiltre :
        filtre = True
    else :
        filtre = False
    # TRAIN
    if not filtre :      
        t_Candidats = str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue) + "/train_Candidats.tsv"
        t_CandidatsCorrige = str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue) + "/train_Candidats_CORRIGE.tsv"
    else :
        t_Candidats = str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue) + "/filter" + str(filterMaxInsert) + "_train_Candidats.tsv"
        t_CandidatsCorrige = str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue) + "/filter" + str(filterMaxInsert) +  "_train_Candidats_CORRIGE.tsv"            
    # postraitement TRAIN: si candidats annotés non extraits => ajoutés au Tableur_candidats
    nbIDIOMATextr,listeCand = EcritCandidatsExtr(cuptTRAIN,langue,"train",dicoMorpho,listePOS,listeRelDep,dicoNFcategories,filtre,filterMaxInsert)         
    copyfile(t_Candidats, t_CandidatsCorrige)
    if nbIDIOMATextr != nbMWEinTrain :
        NEW_postTrait_IdiomatManquants(langue,dicoIDIOMAT,listeCand,t_CandidatsCorrige)
    # 6/ Recherche de candidats dans DEV/TEST    
    for corpus in ["dev","test.blind"] :
        if corpus == "test.blind":
            cupt = cuptTEST
        else :
            cupt = cuptDEV
        EcritCandidatsExtr(cupt,langue,corpus,dicoMorpho,listePOS,listeRelDep,dicoNFcategories,filtre,filterMaxInsert)           
    #==============================================================
    # STEP 3 : Naive Bayes Classifier
    #==============================================================   
    print("Naive Bayes Classification...")  
    trainABS, dicoTraitsNF_MWE,dicoTraitsNF_noMWE = candidatTrain2ABS(langue,filtre,filterMaxInsert) 
    trainCOMP = candidatTrain2COMP(dicoTraitsNF_MWE,dicoTraitsNF_noMWE,langue,trainABS,filtre,filterMaxInsert)     
    trainInput = fusionABSCOMP(trainABS,trainCOMP)          
    for corpus2annot in ["dev","test"] :
        if corpus2annot == "test" :            
            corpus2annot = "test.blind"
            corpusBlind = "new." + corpus2annot 
            devtestABS,traitsABSparNF_test,idCandidats_test = candidatDevTest2ABS("test.blind",langue,filtre,filterMaxInsert) 
            devtestCOMP = candidatDevTest2COMP(idCandidats_test,"test",langue,dicoTraitsNF_MWE,traitsABSparNF_test,filtre,filterMaxInsert) 
        else :
            devtestABS,traitsABSparNF_dev,idCandidats_dev = candidatDevTest2ABS("dev",langue,filtre,filterMaxInsert) 
            devtestCOMP = candidatDevTest2COMP(idCandidats_dev,"dev",langue,dicoTraitsNF_MWE,traitsABSparNF_dev,filtre,filterMaxInsert)
            corpusBlind = "new." + corpus2annot + ".blind"
        devTestInput = fusionABSCOMP(devtestABS,devtestCOMP)
        topX = 300
        labelPred = NBClassif(trainInput,devTestInput,topX,langue,importClassif)
        # Phrases d'origine :
        idPhrases,dicoPhrase = cupt2idPhrases(corpusBlind,langue)
        # candidats : filtrage des candidats étiquettés IDIOMAT
        dicoCandidats,dicoTypeCand = candidats_toId_Categ(labelPred,corpus2annot,langue,filtre,filterMaxInsert)  #{'fr-ud-dev_01265_MWE6': ['26', '59'],
        ecritAnnot(langue,corpus2annot,idPhrases,dicoPhrase,dicoCandidats,dicoTypeCand,filtre,filterMaxInsert)  
    copie2EVAL(langue)
    print("\n" + str(langue) + ": TEST file annotated")
print("\n\nThe end")
