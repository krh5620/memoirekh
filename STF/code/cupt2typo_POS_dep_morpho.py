#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import dirname	# remonter arborescence fichiers
import os.path
repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])

def cupt2POSdepMorpho(langue) :
  """
  INPUT : language
  OUTPUT : for a given language, 3 text files called morpho.txt, POS.txt, relationDep.txt 
  """  
  dicoMorpho = {}
  listeMorpho = []	# pour éviter doublons
  listePOS = []
  listeRelDep = []      
  Train_Cupt = repParent + "/INPUT/DATA_LANG/" + str(langue) + "/train.cupt"
  Dev_Cupt = repParent + "/INPUT/DATA_LANG/" + str(langue) + "/dev.cupt"
  Test_Cupt = repParent + "/INPUT/DATA_LANG/" + str(langue) + "/test.blind.cupt"      
  for corpus in [Train_Cupt,Dev_Cupt,Test_Cupt] :
    with open (corpus, "r") as f_corpus : 
      if langue in ["IT","SL"] :
        colonnePOS = 4
      else :
        colonnePOS = 3      
      lignesCorpus = "".join(f_corpus.readlines()[1:])					
      phrases = lignesCorpus.split("\n\n")[:-1]
      for phrase in phrases :
        lignes = phrase.split("\n")
        for ligne in lignes :
          if "# source_sent_id" in ligne :
            sourcePhrase = ligne.split("# source_sent_id = ")[1]
            idPhrase = sourcePhrase.split(" ")[2]     #e.g. email-enronsent34_01-0024           
          elif "# text = " in ligne :
            textPhrase =  ligne.split("# text = ")[1] #e.g. "A little birdie told me..."                
          elif ligne[0] != "#" :              
            infos = ligne.split("\t")		
            if len(infos) != 11 :
              print("WARNING: missing column in Sentence=" + str(idPhrase))
            else :                  
              POS = infos[colonnePOS]                
              morphos = infos[5].split("|")	#e.g. PronType=Art, PronType=Dem
              dep = infos[7]             
              if POS not in listePOS :
                listePOS.append(POS)                      
              for morpho in morphos :                        
                if "=" in morpho :                                         
                  morphoType = morpho.split("=")[0]
                  morphoValeur = morpho.split("=")[1]
                  if morphoValeur not in listeMorpho : 
                    listeMorpho.append(morphoValeur)                      
                  if  morphoType not in dicoMorpho :
                    dicoMorpho[morphoType] = morphoValeur
                  else :
                    old = dicoMorpho[morphoType].split("|")
                    if morphoValeur not in old : 
                      dicoMorpho[morphoType] = dicoMorpho[morphoType] + "|" + morphoValeur
                else:  
                  # morpho = ["_"] ou comme en turc ["A3sg","Loc","P3sg"]                         
                  if morpho == ['_'] or morpho == "_":                       
                    dicoMorpho["_"] = morpho                                             
                  else :
                    if morpho not in listeMorpho : 
                      listeMorpho.append(morpho)                        
                      if "unknown" not in dicoMorpho :
                        dicoMorpho["unknown"] = morpho
                      else :
                        dicoMorpho["unknown"] = dicoMorpho["unknown"] + "|" + morpho                                                                                                                                        
              if dep not in listeRelDep:
                listeRelDep.append(dep)                
    #====== synthese sur les 3 corpus ====
    listePOS = sorted(listePOS)
    listeRelDep = sorted(listeRelDep)
  return dicoMorpho,listePOS,listeRelDep