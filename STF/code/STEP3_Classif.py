#!/usr/bin/python3
# -*- coding: utf-8 -*-

#================== Imports =====================================
import pickle
import sys
import ast
import nltk
import random
import time
import string
import re
import os
from random import shuffle
from random import randint
from nltk.classify import SklearnClassifier
from collections import defaultdict
import collections
import numpy as np
from shutil import copyfile
from distutils.dir_util import copy_tree
from tqdm import tqdm
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.svm import SVC
#from nltk.classify import maxent
#from sklearn import svm
#=== Frequence NF in Train ====
def train2FreqNF(langue) :
    """
    OUTPUT = dico : clé = NF, valeur = fréquence de la NF dans Train
    => pour identification des hapax dans Train
    """
    freqNFinTrain = {}
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    tableurTRAIN = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/train_MWEs.tsv" 
    with open(tableurTRAIN,"r") as f_tab:
        lignes = f_tab.readlines()
        for ligne in lignes[1:]:
            infos = ligne.split("\t")
            if len(infos) > 2 :
                NF = infos[6].lower()
                if NF not in freqNFinTrain :
                    freqNFinTrain[NF] = 1
                else : 
                    freqNFinTrain[NF] += 1
    return freqNFinTrain
#=== Traits ABS et COMP ====
def candidatTrain2ABS(langue,filtre,filterMaxInsert) :
    """
    OUTPUT = 
    -  features ABS pour candidats du train avec label IDIOMAT/LITERAL pour training classif,
    -  2 dicos séparés IDIOMAT/LITERAL pour avoir ces traits aussi par NF_idCandidat 
    """
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    if not filtre : 
        tableur = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/train_Candidats_CORRIGE.tsv" 
    else :
        tableur = str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue) + "/filter" + str(filterMaxInsert) + "_train_Candidats_CORRIGE.tsv"        
    donneesTRAIN_pourClassif = []
    dicoTraitsNF_MWE = {} # tous les traits ABS par NF et candidat qui sont IDIOMAT dans Train
    dicoTraitsNF_noMWE = {}
    with open(tableur,"r") as f_tab :
        lignes = f_tab.readlines()
        for ligne in lignes[1:] :
            infos = ligne.split("\t")
            if len(infos) > 2 :
                # Pour chaque MWE : combinaison des traits observés
                dicoTraits = {}
                NF = infos[6].lower()
                lemmes = infos[8].split("|")
                positionElts = infos[5].split("|")
                dicoTraits["ABS_NF"] = infos[6]
                dicoTraits["ABS_typeMWE"] = infos[3]
                ordrePOS = infos[9].split("|")
                #Insertions
                dicoTraits["ABS_insert_Raw"] = infos[12]  #DET--NOUN--ADP--DET
                insertionsSansDoublons = ast.literal_eval(infos[16])
                for POS_valeur in insertionsSansDoublons :  #[('ADJ', 0), ('ADP', 1), ('ADV', 1), ('AUX', 0), ('CCONJ', 1), ('DET', 1), ('INTJ', 0), ('NOUN', 1),
                    POS = POS_valeur[0]
                    valeur = POS_valeur[1]
                    dicoTraits["ABS_insert_" + str(POS)] = valeur
                #Morpho et Dep : /!\ besoin de savoir de quel elt il s'agit
                # cas 1 : on précise la relation de dep de l'elt (si plusieurs POS avec meme relDep entrante, on n'en conserve qu'un)
                # cas 2 : plus général : si max_1POS : on met ABS_morpho_VERB = ...ABS_morpho_NOUN =..., si max1 non possible => val = -1                # ------ Morpho -------
                # {'avoir_a_19': {'Poss': '_', 'VerbForm': 'Fin', ..}, 'intérêt_intérêts_24': {'Poss': '_', 'VerbForm': '_', 'Degree': '_', 'Case': '_', 'Polarity': '_', 'Voice': '_', 'NumType': '_', 'unknown': '_', 'Mood': '_', 'Tense': '_', 'Definite': '_', 'Number': 'Plur', 'Gender': 'Masc', 'PronType': '_', 'Person': '_'}}
                #================== Morpho =====================
                syntheseMorpho = ast.literal_eval(infos[17])
                depEntrElts = ast.literal_eval(infos[23])
                #---- cas 1/ -------
                # compte la freq de chaque POS dans l'expression
                dicoFreqPOS = {}
                for elt in ordrePOS :
                    if elt not in dicoFreqPOS :
                        dicoFreqPOS[elt] = 1
                    else :
                        dicoFreqPOS[elt] += 1
                # si la POS n'apparait qu'une fois => valeur ABS_morpho_max1_NOUN_number =  'plur', sinon val = -1   
                for elt in range(0,len(syntheseMorpho)) : 
                    POSElt = ordrePOS[elt]
                    traitsElt = syntheseMorpho[elt]
                    # pour chaque elt :
                    for trait in traitsElt:
                        valeur = traitsElt[trait]                        
                        if dicoFreqPOS[POSElt] == 1 :
                            if trait != "" :
                                dicoTraits["ABS_morpho_max1_" + str(POSElt) + "_" + str(trait)] = valeur
                                dicoTraits["ABS_lemme_max1_" + str(POSElt)] = lemmes[elt]
                        else :
                            if trait != "" :
                                dicoTraits["ABS_morpho_max1_" + str(POSElt) + "_" + str(trait)] = -1   
                                dicoTraits["ABS_lemme_max1_" + str(POSElt)] = -1
                #---- cas 2/ -------
                #print(depEntrElts)
                for elt in range(0,len(syntheseMorpho)) : 
                    depEntrElt = depEntrElts[elt]
                    POSElt = ordrePOS[elt]
                    traitsElt = syntheseMorpho[elt]
                    # pour chaque elt :
                    for trait in traitsElt :
                        if trait != "" :
                            valeur = traitsElt[trait]  
                            dicoTraits["ABS_morpho_" + str(POSElt) + "_" + str(depEntrElt) + "_" +  str(trait)] = valeur
                            dicoTraits["ABS_lemme_" + str(POSElt) + "_" + str(depEntrElt) ] = lemmes[elt]    
                #================= dep syntaxiques sortantes ==============
                syntheseDeps = ast.literal_eval(infos[18])                
                #------ cas 1/ :
                # si la POS n'apparait qu'une fois , sinon val = -1   
                for elt in range(0,len(syntheseDeps)) : 
                    allTraits = syntheseDeps[elt]
                    # pour chaque elt :
                    for Tuple_trait in allTraits :
                        trait = Tuple_trait[0]
                        valeur = Tuple_trait[1]
                        if dicoFreqPOS[POSElt] == 1 :
                            if trait != "" :
                                dicoTraits["ABS_depSyn_max1_" + str(POSElt) + "_" + str(trait)] = valeur
                        else :
                            if trait != "" :
                                dicoTraits["ABS_depSyn_max1_" + str(POSElt) + "_" + str(trait)] = -1  
                #---- cas 2/ -------
                for elt in range(0,len(syntheseDeps)) : 
                    allTraits = syntheseDeps[elt]
                    depEntrElt = depEntrElts[elt]
                    POSElt = ordrePOS[elt]
                    # pour chaque elt :
                    for Tuple_trait in allTraits :
                        trait = Tuple_trait[0]  
                        valeur = Tuple_trait[1]  
                        if trait != "" :                            
                            dicoTraits["ABS_depSyn_" + str(POSElt) + "_" + str(depEntrElt) + "_" +  str(trait)] = valeur                       
                # distSyn  V-N(ssi VB/AUX -NOUN)
                distSyn = infos[19]
                typeDistSyn = infos[20]
                dicoTraits["ABS_distSyn_VerbNoun"] = distSyn
                dicoTraits["ABS_TypedistSyn_VerbNoun"] = typeDistSyn
                # distSyn2 elts  
                distSyn = infos[21]
                typeDistSyn = infos[22]
                dicoTraits["ABS_distSyn_2elts"] = distSyn
                dicoTraits["ABS_TypedistSyn_2elts"] = typeDistSyn  
                # TRAIN => 2 labels : Idiomat vs  literal
                label = infos[24].split("\n")[0]
                # Donnees sortie :
                donneesTRAIN_pourClassif.append((dicoTraits,label))   
                #============  Traits par NF et candidat ===
                if label == "IDIOMAT" :
                    if NF not in dicoTraitsNF_MWE :
                        dicoTraitsNF_MWE[NF] = {}
                        dicoTraitsNF_MWE[NF][infos[0]] = dicoTraits
                    else :
                        dicoTraitsNF_MWE[NF][infos[0]] = dicoTraits
                else :
                    if NF not in dicoTraitsNF_noMWE :
                        dicoTraitsNF_noMWE[NF] = {}
                        dicoTraitsNF_noMWE[NF][infos[0]] = dicoTraits
                    else :
                        dicoTraitsNF_noMWE[NF][infos[0]] = dicoTraits 
    return donneesTRAIN_pourClassif,dicoTraitsNF_MWE,dicoTraitsNF_noMWE

def candidatTrain2COMP(dicoTraitsNF_MWE,dicoTraitsNF_noMWE,langue,newtrainABS,filtre,filterMaxInsert) :
    """
    Traits COMP obtenus apr comapraison des traits ABS
    """    
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    if not filtre : 
        tableur = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/train_Candidats_CORRIGE.tsv" 
    else :
        tableur = str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue) + "/filter" + str(filterMaxInsert) + "_train_Candidats_CORRIGE.tsv"        
    with open(tableur,"r") as f_tab :
        lignes = f_tab.readlines()   
        featuresCOMP = []
        allfreqNFinTRAIN = train2FreqNF(langue)
        message = str(langue) + "-train: Candidate Features"
        pbar = tqdm(lignes[1:],desc=message)         
        for ligne in pbar :            
            infos = ligne.split("\t")
            if len(infos) > 2 :    
                idCandidat = infos[0]
                NF = infos[6].lower()     
                POS_elts = infos[9].split("|")
                label = infos[24].split("\n")[0]
                dicoTraitsCOMP_extr = {}  #comparaison avec les occ extraites ET annotées  pour une même NF
                #dicoTraitsNF _> {'hommage;rendre': {'annodis.er_00353_MWE2': {'ABS_insert_PRON': 0, 'ABS_depSyn_obl:agent':
                freqNFinTRAIN = allfreqNFinTRAIN[NF]    # frequence de la NF annotée               
                if freqNFinTRAIN == 1 and label == "IDIOMAT":
                    # Si une seule occurrence annotée :
                    # - soit le candidat est cette ocurrence => no comapraison possible => SUPPR du corpus d'entrainement
                    # - soit le candidat n'est pas annoté => comparaison possible                    
                    allTraits = dicoTraitsNF_MWE[NF][idCandidat]
                    for trait in allTraits :                                                
                        if "ABS_NF" not in trait and "ABS_typeMWE" not in trait and "ABS_lemme" not in trait :
                            traitCOMP = trait.split("ABS_")[1]
                            dicoTraitsCOMP_extr["COMP_" + str(traitCOMP)] = "noCOMP"
                    featuresCOMP.append((dicoTraitsCOMP_extr,label))  
                else :                   
                    for idMWE in dicoTraitsNF_MWE[NF] : 
                        allTraits1 = dicoTraitsNF_MWE[NF][idMWE]
                        #1/ comparaison d'une MWE IDIOMAT avec une autre MWE IDIOMAT :
                        if idCandidat in dicoTraitsNF_MWE[NF] and idMWE != idCandidat :
                            allTraits2 = dicoTraitsNF_MWE[NF][idCandidat]
                            for trait in allTraits1 :
                                if "ABS_NF" not in trait and "ABS_typeMWE" not in trait and "ABS_lemme" not in trait :
                                    traitCOMP = "COMP_" + str(trait.split("ABS_")[1])
                                    dicoTraitsCOMP_extr[str(traitCOMP)] = "false"
                                    if trait in allTraits1 and trait in allTraits2 :
                                        if allTraits1[trait] == allTraits2[trait]  :
                                            dicoTraitsCOMP_extr[str(traitCOMP)] = "true"                            
                        # 2/ comparaison d'une MWE LITERAL avec une MWE IDIOMAT
                        elif NF in dicoTraitsNF_noMWE:
                            if idCandidat in dicoTraitsNF_noMWE[NF] :
                                allTraits2 = dicoTraitsNF_noMWE[NF][idCandidat]
                                for trait in allTraits1 :
                                    if "ABS_NF" not in trait and "ABS_typeMWE" not in trait and "ABS_lemme" not in trait :
                                            traitCOMP = "COMP_" + str(trait.split("ABS_")[1])
                                            dicoTraitsCOMP_extr[str(traitCOMP)] = "false"
                                            if trait in allTraits1 and trait in allTraits2 :
                                                if allTraits1[trait] == allTraits2[trait]  :
                                                    dicoTraitsCOMP_extr[str(traitCOMP)] = "true"    
                    featuresCOMP.append((dicoTraitsCOMP_extr,label))    
    return featuresCOMP

def candidatDevTest2ABS(corpus,langue,filtre,filterMaxInsert) :
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    if not filtre : 
        tableur = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/" + str(corpus) + "_Candidats.tsv"  
    else :
        tableur = str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue) + "/filter" + str(filterMaxInsert) + "_" + str(corpus) + "_Candidats.tsv"    
    donnees_pourClassif = []
    traitsdevABStest = {}
    idCandidats = []
    with open(tableur,"r") as f_tab :
        lignes = f_tab.readlines()
        for ligne in lignes[1:] :
            infos = ligne.split("\t")
            if len(infos) > 2 :
                # Pour chaque MWE : combinaison des traits observés
                dicoTraits = {}
                idCandidat = infos[0]
                idCandidats.append(idCandidat)
                NF = infos[6]
                lemmes = infos[8].split("|")
                ordrePOS = infos[9].split("|")
                dicoTraits["ABS_NF"] = NF
                dicoTraits["ABS_typeMWE"] = infos[3]
                depEntrElts = ast.literal_eval(infos[23])
                #------------------------------------------------------------
                dicoTraits["ABS_insert_Raw"] = infos[12]  #DET--NOUN--ADP--DET
                insertionsSansDoublons = ast.literal_eval(infos[16])
                for POS_valeur in insertionsSansDoublons :  #[('ADJ', 0), ('ADP', 1), ('ADV', 1), ('AUX', 0), ('CCONJ', 1), ('DET', 1), ('INTJ', 0), ('NOUN', 1),
                    POS = POS_valeur[0]
                    valeur = POS_valeur[1]
                    dicoTraits["ABS_insert_" + str(POS)] = valeur                             
                #------------------------------------------------------------
                # Morpho 
                # {'avoir_a_19': {'Poss': '_', 'VerbForm': 'Fin', ..}, 'intérêt_intérêts_24': {'Poss': '_', 'VerbForm': '_', 'Degree': '_', 'Case': '_', 'Polarity': '_', 'Voice': '_', 'NumType': '_', 'unknown': '_', 'Mood': '_', 'Tense': '_', 'Definite': '_', 'Number': 'Plur', 'Gender': 'Masc', 'PronType': '_', 'Person': '_'}}
                syntheseMorpho = ast.literal_eval(infos[17])
                # compte la freq de chaque POS dans l'expression
                dicoFreqPOS = {}
                for elt in ordrePOS :
                    if elt not in dicoFreqPOS :
                        dicoFreqPOS[elt] = 1
                    else :
                        dicoFreqPOS[elt] += 1
                # si la POS n'apparait qu'une fois => valeur ABS_morpho_max1_NOUN_number =  'plur', sinon val = -1  
                for elt in range(0,len(syntheseMorpho)) : 
                    POSElt = ordrePOS[elt]
                    traitsElt = syntheseMorpho[elt]
                    # pour chaque elt :
                    for trait in traitsElt:
                        valeur = traitsElt[trait]                        
                        if dicoFreqPOS[POSElt] == 1 :
                            if trait != "" :
                                dicoTraits["ABS_morpho_max1_" + str(POSElt) + "_" + str(trait)] = valeur
                                dicoTraits["ABS_lemme_max1_" + str(POSElt)] = lemmes[elt]
                        else :
                            if trait != "" :
                                dicoTraits["ABS_morpho_max1_" + str(POSElt) + "_" + str(trait)] = -1                        
                                dicoTraits["ABS_lemme_max1_" + str(POSElt)] = -1
                #---- cas 2/ -------
                #print(depEntrElts)
                for elt in range(0,len(syntheseMorpho)) : 
                    depEntrElt = depEntrElts[elt]
                    POSElt = ordrePOS[elt]
                    traitsElt = syntheseMorpho[elt]
                    # pour chaque elt :
                    for trait in traitsElt :
                        if trait != "" :
                            valeur = traitsElt[trait]  
                            dicoTraits["ABS_morpho_" + str(POSElt) + "_" + str(depEntrElt) + "_" +  str(trait)] = valeur
                            dicoTraits["ABS_lemme_" + str(POSElt) + "_" + str(depEntrElt) ] = lemmes[elt]
                #------------------------------------------------------------
                #dep syntaxiques sortantes
                syntheseDeps = ast.literal_eval(infos[18])
                #------ cas 1/ :
                # si la POS n'apparait qu'une fois => valeur ABS_depSyn_max1_NOUN_number =  'plur', sinon val = -1   
                for elt in range(0,len(syntheseDeps)) : 
                    allTraits = syntheseDeps[elt]
                    # pour chaque elt :
                    for Tuple_trait in allTraits :
                        trait = Tuple_trait[0]
                        valeur = Tuple_trait[1]
                        if dicoFreqPOS[POSElt] == 1 :
                            if trait != "" :
                                dicoTraits["ABS_depSyn_max1_" + str(POSElt) + "_" + str(trait)] = valeur
                        else :
                            if trait != "" :
                                dicoTraits["ABS_depSyn_max1_" + str(POSElt) + "_" + str(trait)] = -1      
                #---- cas 2/ -------
                for elt in range(0,len(syntheseDeps)) : 
                    allTraits = syntheseDeps[elt]
                    depEntrElt = depEntrElts[elt]
                    POSElt = ordrePOS[elt]
                    # pour chaque elt :
                    for Tuple_trait in allTraits :
                        trait = Tuple_trait[0]  
                        valeur = Tuple_trait[1]  
                        if trait != "" :                            
                            dicoTraits["ABS_depSyn_" + str(POSElt) + "_" + str(depEntrElt) + "_" +  str(trait)] = valeur 
                #------------------------------------------------------------
                # distSyn  V-N(ssi VB/AUX -NOUN)
                distSyn = infos[19]
                typeDistSyn = infos[20]
                dicoTraits["ABS_distSyn_VerbNoun"] = distSyn
                dicoTraits["ABS_TypedistSyn_VerbNoun"] = typeDistSyn
                #------------------------------------------------------------
                # distSyn2 elts  
                distSyn = infos[21]
                typeDistSyn = infos[22]
                dicoTraits["ABS_distSyn_2elts"] = distSyn
                dicoTraits["ABS_TypedistSyn_2elts"] = typeDistSyn  
                #------------------------------------------------------------
                if NF not in traitsdevABStest :
                    traitsdevABStest[NF] = {}
                    traitsdevABStest[NF][idCandidat] = dicoTraits
                else :
                    traitsdevABStest[NF][idCandidat] = dicoTraits
                    
                donnees_pourClassif.append((dicoTraits))  
    return donnees_pourClassif,traitsdevABStest,idCandidats

def candidatDevTest2COMP(idCandidats,corpus,langue,traitstrainABS,traitsdevABStest,filtre,filterMaxInsert) :
    """
    Comparaison avec NF annotées du Train
    """
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    if not filtre : 
        tableurTrain = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/train_Candidats_CORRIGE.tsv" 
        tableurDevTest = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/" + str(corpus) + "_Candidats.tsv" 
    else :
        tableurTrain = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/filter" + str(filterMaxInsert) +  "_train_Candidats_CORRIGE.tsv" 
        tableurDevTest = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/filter" + str(filterMaxInsert) + "_" + str(corpus) + "_Candidats.tsv"         
             
    # traitstrainABS  : toutes les propriétés observées dans Train pour une NF annotée
    allTraitsCOMP = []
    for idcandidat in idCandidats :
        traitsCOMP = {}
        for NF1 in traitsdevABStest :
            if idcandidat in traitsdevABStest[NF1] :
                traitsCandidat = traitsdevABStest[NF1][idcandidat]
                #init
                for trait in traitsCandidat :
                    if "ABS_NF" not in trait and "ABS_typeMWE" not in trait and "ABS_lemme" not in trait :
                        traitCOMP = "COMP_" + str(trait.split("ABS_")[1])
                        traitsCOMP[traitCOMP] = "false"
                # recherche de la comparaison de ce candidat avec les MWE de meme NF dans Train
                for NF2 in traitstrainABS :
                    if NF1 == NF2 : 
                        traitsAnnots_alloccs = traitstrainABS[NF2]
                        for occ in traitsAnnots_alloccs :
                            traitsAnnots_occ = traitsAnnots_alloccs[occ] 
                            for trait in traitsAnnots_occ :
                                if "ABS_NF" not in trait and "ABS_typeMWE" not in trait and "ABS_lemme" not in trait :
                                    traitCOMP = "COMP_" + str(trait.split("ABS_")[1])                                   
                                    if trait in traitsAnnots_occ and trait in traitsCandidat :
                                        if traitsAnnots_occ[trait] == traitsCandidat[trait] :
                                                traitsCOMP[traitCOMP] = "true"
                                    else : 
                                        traitsCOMP[traitCOMP] = -1 
        
        allTraitsCOMP.append(traitsCOMP)
    return allTraitsCOMP

def fusionABSCOMP(liste_traitsABS,liste_traitsCOMP) :
    all_traitsCandidats_ABSCOMP = []
    for candidat in range(0,len(liste_traitsABS)) :
        dico_traitsCandidat_ABSCOMP = {}
        if len(liste_traitsABS[candidat]) == 2 :
            traitsCandidatABS = liste_traitsABS[candidat][0]
            traitsCandidatCOMP = liste_traitsCOMP[candidat][0]            
            labelCandidat = liste_traitsABS[candidat][1]
            for traitABS in traitsCandidatABS :
                dico_traitsCandidat_ABSCOMP[traitABS] = traitsCandidatABS[traitABS]
            for traitCOMP in traitsCandidatCOMP :
                dico_traitsCandidat_ABSCOMP[traitCOMP] = traitsCandidatCOMP[traitCOMP] 
            all_traitsCandidats_ABSCOMP.append((dico_traitsCandidat_ABSCOMP,labelCandidat))
        else :
            traitsCandidatABS = liste_traitsABS[candidat]
            traitsCandidatCOMP = liste_traitsCOMP[candidat]
            for traitABS in traitsCandidatABS :
                dico_traitsCandidat_ABSCOMP[traitABS] = traitsCandidatABS[traitABS]
            for traitCOMP in traitsCandidatCOMP :
                dico_traitsCandidat_ABSCOMP[traitCOMP] = traitsCandidatCOMP[traitCOMP] 
            all_traitsCandidats_ABSCOMP.append((dico_traitsCandidat_ABSCOMP))        
    return all_traitsCandidats_ABSCOMP
#=== Naive Bayes ================
def NBClassif(train,corpusEval,topX,langue,importClassif) :
    labelPred,listeFeatures = extractInformativeFeatures_NB(train,corpusEval,topX,langue,importClassif)
    return labelPred
def showInformativeFeatures(classifier, topX):     
    """
    Return a nested list of the "most informative" features for both labels IDIOMAT/LITERAL
    used by the classifier along with it's predominant labels + ratio
    """
    n = int(topX)
    cpdist = classifier._feature_probdist       # probability distribution for feature values given labels
    feature_list = []
    dicoFeature = {}         
    for (fname, fval) in classifier.most_informative_features(n):
        def labelprob(l):
            return cpdist[l, fname].prob(fval)
        labels = sorted([l for l in classifier._labels if fval in cpdist[l, fname].samples()], 
                        key=labelprob)            
        if len(labels) == 1:
            continue
        l0 = labels[0]
        l1 = labels[-1]        
        if cpdist[l0, fname].prob(fval) == 0:
            ratio = 'INF'
        else:
            ratio = '%8.1f' % (cpdist[l1, fname].prob(fval) /
                               cpdist[l0, fname].prob(fval))
        """print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
               (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)) )  """     
        if l1 not in dicoFeature or dicoFeature[l1] == None:
            dicoFeature[l1] = [str(fname) + "=" +str(fval),float(str(ratio[2:]).replace(" ", ""))]  
        else :
            dicoFeature[l1] = dicoFeature[l1] + [str(fname) + "=" +str(fval),float(str(ratio[2:]).replace(" ", ""))]    
    newDico = {}            
    for cle in dicoFeature:
        compteur = 0
        newListe = []
        for elt in range(0,int(len(dicoFeature[cle]))):  
            if elt%2 ==0 :
                newListe.append([dicoFeature[cle][elt],dicoFeature[cle][elt+1]])
        newDico[cle] = newListe    
    newDicoTri = collections.OrderedDict(sorted(newDico.items(), key=lambda t: t[0]))
    return newDicoTri

def save_classifier(classifier,langue):
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    path2classifier = str(repParent)  + "/classifier/" +  str(langue) + "_NBclassifier.pickle"   
    with open(path2classifier, 'wb') as f :
        pickle.dump(classifier, f, -1)

def load_classifier(langue):
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    path2classifier = str(repParent)  + "/classifier/" +  str(langue) + "_NBclassifier.pickle"  
    with open(path2classifier, 'rb') as f :
        classifier = pickle.load(f)
    return classifier

def extractInformativeFeatures_NB(train,test,topX,langue,importClassif):     
    if importClassif == "False" : 
        # Parameter = importClassif set to FALSE
        classifier = nltk.classify.NaiveBayesClassifier.train(train)
    else :
        # Parameter = importClassif set to TRUE
        repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
        path2classifier = str(repParent)  + "/classifier/" +  str(langue) + "_NBclassifier.pickle"        
        if os.path.exists(path2classifier) :    
            # load existing classifier
            classifier = load_classifier(langue)           
        else :
            # 1st time => save classifier as pickle
            classifier = nltk.classify.NaiveBayesClassifier.train(train)
            save_classifier(classifier,langue)
    resultClassif = classifier.classify_many(test)
    listeFeatures = [] # [['MorphoNOUN_sing', 'LITERAL'], ['MorphoNOUN_plur', 'LITERAL'],
    listeFeatures = showInformativeFeatures(classifier, topX)
    return resultClassif,listeFeatures

#=== Ecriture resultats classif =========
def cupt2idPhrases(cupt,langue) :
    """
    renvoie la liste des idPhrases dans leur ordre d'apparition dans le fichier cupt
    """
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    fichierCupt = str(repParent) + "/INPUT/DATA_LANG/" + str(langue) + "/" + str(cupt) + ".cupt" 
    listeIdPhrases = []
    dicoPhrase = {}
    with open (fichierCupt, "r") as f_cupt : 	
        infosCupt =  f_cupt.readlines()
        corpusCupt = "".join(infosCupt[1:])					
        phrases = corpusCupt.split("\n\n")[:-1]
        for phrase in phrases :
            lignes = phrase.split("\n")                        
            for ligne in lignes :
                if "# source_sent_id" in ligne :
                    sourcePhrase = ligne.split("# source_sent_id = ")[1]
                    idPhrase = sourcePhrase.split(" ")[2]     
                    dicoPhrase[idPhrase] = lignes
                    if idPhrase not in listeIdPhrases : 
                        listeIdPhrases.append(idPhrase)
                    else :
                        print("ERREUR : idPhrase déjà présent")
                        sys.exit()
    return listeIdPhrases,dicoPhrase


def candidats_toId_Categ(labelPred,corpus,langue,filtre,filterMaxInsert) :
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    if not filtre :
        tableur = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/" + corpus + "_Candidats.tsv" 
    else :
        tableur = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/filter" + str(filterMaxInsert) + "_" + str(corpus) + "_Candidats.tsv"     
    dicoCandidats = {}
    dicoTypeCand = {}
    with open(tableur,"r") as f_tab :
        lignes = f_tab.readlines()    
        comptCandidat = 0        
        for ligne in lignes[1:] :
            infos = ligne.split("\t")
            if len(infos) > 2 :
                if labelPred[comptCandidat] == "IDIOMAT" :
                    idMWE = infos[0]
                    typeMWE = infos[3]                
                    numToks = infos[5].split("|")
                    dicoCandidats[idMWE] = numToks                
                    dicoTypeCand[idMWE] = typeMWE
                comptCandidat += 1
    return dicoCandidats,dicoTypeCand

def ecritAnnot(langue,corpus2annot,idPhrases,dicoPhrase,dicoCandidats,dicoTypeCand,filtre,filterMaxInsert):
    """
    Ajout des anotations du système dans dev/test
    """
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    if corpus2annot == "test.blind" :
        corpus2annot = "test"    
    if not filtre : 
        if not os.path.exists(str(repParent)  + "/OUTPUT/varIDE_ANNOT/" + str(langue) + "/noFILTER/"):
            os.makedirs(str(repParent)  + "/OUTPUT/varIDE_ANNOT/" + str(langue) + "/noFILTER/")              
        fichierSyst = str(repParent) + "/OUTPUT/varIDE_ANNOT/" + str(langue) + "/noFILTER/" + str(corpus2annot) + ".system.cupt" 
    else :
        if not os.path.exists(str(repParent)  + "/OUTPUT/varIDE_ANNOT/" + str(langue) + "/FILTER/"):
            os.makedirs(str(repParent)  + "/OUTPUT/varIDE_ANNOT/" + str(langue) + "/FILTER/")        
        fichierSyst = str(repParent) + "/OUTPUT/varIDE_ANNOT/" + str(langue) + "/FILTER/" + str(corpus2annot) + ".system.cupt"  
    with open(fichierSyst,"a") as f_out : 
        f_out.write("# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE\n")
        for idPhrase in idPhrases : 
            candidatInPhrase = False
            for idCandidat in dicoCandidats :                
                idPhraseCandidat = idCandidat.split("_MWE")[0]
                if idPhrase == idPhraseCandidat :
                    # au moins 1 candidat dans la phrase
                    candidatInPhrase = True 
            if not candidatInPhrase :
                # pas de condidat => recopie la phrase
                lignes = dicoPhrase[idPhrase]
                for ligne in lignes : 
                    f_out.write(str(ligne) + "\n")
                f_out.write("\n")
            else :              
                # si plusieurs candidats dans la même phrase :
                lignes = dicoPhrase[idPhrase]    
                for idCandidat in dicoCandidats :  
                    idPhraseCandidat = idCandidat.split("_MWE")[0]
                    if idPhraseCandidat == idPhrase :
                        # ajoute l'annotation du systeme en dernière colonne            
                        toksCandidat = dicoCandidats[idCandidat]
                        for i in range(0,len(lignes)) :
                            infos = lignes[i].split("\t")
                            if len(infos) > 2 : # pas besoin de changer l'entete
                                for tok in toksCandidat : 
                                    if not "-" in infos[0] and tok == infos[0] :
                                            numCandidat = idCandidat.split("_MWE")[1]
                                            # si pas d'annotation sur cette ligne
                                            infos = lignes[i].split("\t")
                                            if infos[10] == "*" : 
                                                if tok == toksCandidat[0] :
                                                    # 1ere apparition : num candidat + type MWE prédit
                                                    typeMWE = dicoTypeCand[idCandidat]
                                                    oldLigne = lignes[i]
                                                    oldInfos = oldLigne.split("\t")
                                                    newInfosSaufDernCol = "\t".join(oldInfos[:-1])
                                                    lignes[i] = str(newInfosSaufDernCol) + "\t" + str(numCandidat) + ":" + str(typeMWE)  
                                                else :
                                                    # apparitions suivantes
                                                    oldLigne = lignes[i]
                                                    oldInfos = oldLigne.split("\t")
                                                    newInfosSaufDernCol = "\t".join(oldInfos[:-1])                                                
                                                    lignes[i] = str(newInfosSaufDernCol) + "\t" + str(numCandidat)  
                                            else :
                                                # tok déjà annoté
                                                oldLigne = lignes[i]
                                                oldInfos = oldLigne.split("\t")
                                                newInfosSaufDernCol = "\t".join(oldInfos[:-1])   
                                                oldDernCol = oldInfos[len(oldInfos)-1]
                                                if tok == toksCandidat[0] :
                                                    # 1ere apparition : num candidat + type MWE prédit
                                                    typeMWE = dicoTypeCand[idCandidat]
                                                    lignes[i] = str(newInfosSaufDernCol) + "\t" + str(oldDernCol) + ";" + str(numCandidat) + ":" + str(typeMWE)                                                  
                                                else :
                                                    # apparitions suivantes
                                                    lignes[i] = str(newInfosSaufDernCol) +  "\t" + str(oldDernCol) + ";" + str(numCandidat)      
                for i in lignes :
                    f_out.write(str(i) + "\n")
                f_out.write("\n")
    # copie du fichier annote par le systeme pour evaluation
    fichierSyst_repEVAL = str(repParent) + "/OUTPUT/EVAL/" + str(langue) + "/" + str(corpus2annot) + ".system.cupt"
    copyfile(fichierSyst, fichierSyst_repEVAL)
    RepIN = str(repParent) + "/evaluation"
    RepOUT = str(repParent) + "/OUTPUT/EVAL/"   
    copy_tree(RepIN, RepOUT)    
    
def copie2EVAL(langue) :
    """
    copie les fichiers train.cupt et dev.cupt pour l'évaluation avec evaluate.py
    """
    repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
    repIN = str(repParent) + "/INPUT/DATA_LANG/" + str(langue) + "/"
    repEVAL = str(repParent) + "/OUTPUT/EVAL/" + str(langue) + "/"
    repCAND = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/"
    trainIN =  str(repIN) + "train.cupt"
    trainOUT = str(repEVAL) + "train.cupt"
    devIN =  str(repIN) + "dev.cupt"
    devOUT = str(repEVAL) + "dev.cupt"        
    copyfile(trainIN, trainOUT)
    copyfile(devIN, devOUT)
    for fichier in os.listdir(repEVAL):       
        if fichier.startswith("test.blind"):
            os.rename(os.path.join(repIN, fichier), os.path.join(repEVAL, fichier.replace("test.blind", "test")))  
    for fichier in os.listdir(repCAND):       
        if fichier.endswith("_temp.tsv"):
            os.remove(str(repCAND) + str(fichier))  
    