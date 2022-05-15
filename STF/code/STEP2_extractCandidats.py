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
from shutil import copyfile
from cupt2typo_POS_dep_morpho import * 
from distSyn import * 
from tqdm import tqdm 
#-----------------------
def cupt2phrases(cupt):
  dicoPhrasesSansEntete = {}
  dicoPhrasesAvecEntete = {}
  with open (cupt, "r") as f_cupt : 	
    infosCupt =  f_cupt.readlines()
    corpusCupt = "".join(infosCupt[1:])					
    phrases = corpusCupt.split("\n\n")[:-1]
    textPhrases = []
    indicesDoublons = []
    compteurPhrase = 1
    for phrase in phrases :
      lignes = phrase.split("\n")                        
      for ligne in lignes :
        if "# source_sent_id" in ligne :
          sourcePhrase = ligne.split("# source_sent_id = ")[1]
          idPhrase = sourcePhrase.split(" ")[2]     #email-enronsent34_01-0024           
        elif "# text = " in ligne :
          textPhrase =  ligne.split("# text = ")[1] #"A little birdie told me..."  
      dicoPhrasesSansEntete[idPhrase] = lignes[2:]
      dicoPhrasesAvecEntete[compteurPhrase] = (idPhrase,lignes)
      compteurPhrase += 1
  return dicoPhrasesSansEntete,dicoPhrasesAvecEntete

def tupleOrdreCroissant (liste):
  #liste = [(['36', '37'], 'IDIOMAT'), (['2', '19'], 'LITERAL'), (['1', '19'], 'LITERAL'), (['12', '36'], 'LITERAL'), (['12', '19'], 'LITERAL'), (['19', '24'], 'LITERAL')]
  newListe = []
  allpositionMWE = []
  # input = [(['36', '37'], 'IDIOMAT'), (['2', '19'], 'LITERAL'), (['1', '19'], 'LITERAL'), 
  # output = [(['1', '19'], 'LITERAL'), (['2', '19'], 'LITERAL') (['36', '37'], 'IDIOMAT'), , , 
  for MWE in liste :
    positionMWE = list(map(int, MWE[0]))
    allpositionMWE.append(positionMWE)
  allpositionMWE = sorted(allpositionMWE)
  for p in allpositionMWE :
    for MWE in liste :
      if list(map(int, MWE[0])) == p :
        newListe.append((MWE[0],MWE[1],MWE[2]))
  return newListe

def isTupleCroissant(tupleString):    
  tupleInt = list(map(int, tupleString))
  tupleOrdon = sorted(tupleInt)
  if tupleOrdon == tupleInt :
    croissant = True
  else :
    croissant = False
  return croissant
# ---- Initialisations
def initInsertions(listePOS):
  statsInsertions = {}
  for POS in listePOS :
    statsInsertions[POS] = 0
  return statsInsertions
# ---- Recherche candidats
def rechercheCandidats(corpus,langue,variantesPOS_NF,dicoNF2POSnorm,dicoGenerLemmes,filterInsert,MWEannoteesTrain):
  candidatTrouve = {} # comparaison avec annot manuelle
  # Obtention de toutes les phrases du fichier cupt => pour recherche des candidats
  repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
  cupt =  str(repParent)  + "/INPUT/DATA_LANG/" + str(langue) + "/" + str(corpus) + ".cupt"
  dicoPhrases,dicoPhrasesAvecEntete = cupt2phrases(cupt)
  message = str(langue) + "-" + str(corpus) + ": Search Candidates"
  pbar = tqdm(dicoPhrases,desc=message)     
  for phrase in pbar:       
    numCandidat = 1
    lignes = dicoPhrases[phrase]
    phraseLemm,phraseNumToks = infosPhrase_to_lemmes_numTok(lignes,langue)
    for NF in dicoNF2POSnorm :
      NF_in_phrase = False    
      # 1/ Recherche des différentes occurrences des lemmes dans la phrase :
      # {'avoir': [2, 4], 'y': [1, 5, 6], 'il': [0, 3, 7]}
      dicoElts = {}
      listeEltsNFAtrouver = NF.split(";")
      for elt in listeEltsNFAtrouver :
        for elt2 in range(0,len(phraseLemm)) :
          if phraseLemm[elt2] == elt :           
            if elt not in dicoElts :
              dicoElts[elt] = [phraseNumToks[elt2]] 
            else :
              dicoElts[elt] = dicoElts[elt] + [phraseNumToks[elt2]]  
      if len(dicoElts) == len(listeEltsNFAtrouver) :
        NF_in_phrase = True
      if NF_in_phrase and NF in dicoGenerLemmes : # NF in dicoGenerLemmes => ie si la NF est associée à un patron dans le topX                                    
        # 2/ Liste toutes les combinaisons possibles pour chaque variante dans cette phrase
        allCombinaisons = {}   
        for v in dicoGenerLemmes[NF] :
          listeLemm = v.split("|")
          combinaisons = []
          for elt in listeLemm :
            combinaisons.append(dicoElts[elt])
            allCombinaisons[v] = list(product(*combinaisons))	
        #3/ Ne conserve que les lemmes trouvés sont dans un ordre compatible avec les patrons autorisés
        # VERIF LEMMES : 1er test si tuples par ordre croissant => si croissant conservé
        combinaisonsOK1 = []
        for v in allCombinaisons :
          for tupleCombi in  allCombinaisons[v] :
            if isTupleCroissant(tupleCombi) :
              combinaisonsOK1.append(tupleCombi)  
        # VERIF POS : 2e test : verif POS correspondent aux POS possibles
        combinaisonsOK2 = []
        for numsCandidat in combinaisonsOK1 :	
          listeCandidatPOS = []
          for num in numsCandidat :		
            for ligne in lignes :
              infos = ligne.split("\t")
              if str(infos[0]) == str(num) :
                if langue not in ["SL","IT"]:
                  listeCandidatPOS.append(infos[3])	
                else :
                  listeCandidatPOS.append(infos[4])
          candidatPOS = "|".join(listeCandidatPOS)
          if str(candidatPOS) in variantesPOS_NF[NF] :
            combinaisonsOK2.append(numsCandidat)
        if combinaisonsOK2 != [] : 
          for combinaison in combinaisonsOK2 :
            toksArechercher = ""
            for tok in range(0,len(combinaison)) :                                        
              if toksArechercher == "" : 
                toksArechercher = str(combinaison[tok])  
              else :
                toksArechercher = str(toksArechercher) + "|" + str(combinaison[tok]) 
            #---- Recherche de correspondance parfaite => sinon LITERAL
            # SI phrase contient au moins un candidat ANNOTE (mais pas forcément celui-là) :
            if phrase in MWEannoteesTrain:
              # Si correspondance extraction-annotation => IDIOMAT 
              if toksArechercher in MWEannoteesTrain[phrase] :
                if phrase in candidatTrouve :
                  candidatTrouve[phrase].append((NF,toksArechercher,"IDIOMAT"))
                else :
                  candidatTrouve[phrase] = [(NF,toksArechercher,"IDIOMAT")]  
              else :
                # Si pas de correspondance extraction-annotation :    
                if phrase in candidatTrouve :
                  candidatTrouve[phrase].append((NF,toksArechercher,"LITERAL"))
                else :
                  candidatTrouve[phrase] = [(NF,toksArechercher,"LITERAL")]                                                   
            else :
              # Phrase ne contenant aucune MWE annotée :
              if phrase not in candidatTrouve :
                candidatTrouve[phrase]= [(NF,toksArechercher,"LITERAL")]                                        
              else :
                candidatTrouve[phrase].append((NF,toksArechercher,"LITERAL")) 
  if not os.path.exists(str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue)):
    os.makedirs(str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue)) 
  # Avec et sans filtre sur le nb max d'insertions
  fichierCand_TEMP = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/" + str(corpus) + "_Candidats_temp.tsv"
  fichierCand_TEMP_filt = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/" + "filter"+ str(filterInsert) + "_" + str(corpus) + "_Candidats_temp.tsv"
  with open(fichierCand_TEMP,"a") as f_candTEMP, open(fichierCand_TEMP_filt,"a") as f_candTEMP_filt :
    for phrase in candidatTrouve :
      for candidat in candidatTrouve[phrase] :
        f_candTEMP.write(str(phrase) + "\t" + str(candidat[0]) + "\t" + str(candidat[1]) + "\t" + str(candidat[2]) + "\n")  
        toks = candidat[1].split("|")
        tokDebut = int(toks[0])
        tokFin = int(toks[len(toks)-1])
        NbInserts = tokFin - tokDebut - 1
        if NbInserts <= filterInsert :# Shared task : filterX = 20
          f_candTEMP_filt.write(str(phrase) + "\t" + str(candidat[0]) + "\t" + str(candidat[1]) + "\t" + str(candidat[2]) + "\n")
         
# ---- Extraction infos du tableur Train
def infosTableur(tableur):
  with open (tableur,"r") as f_tab :
    lignes = f_tab.readlines()                
  return lignes   

def NF2Categorie(tableur):
  # OUTPUT = dico clé = NF de TRAIN, valeur = catégorie majoritairement attribuée pour chaque NF de TRAIN 
  """TODO : si même fréquence : Cf guide annotation, si doute entre LVC.full et cause, privilégier LVC.full"""
  lignes = infosTableur(tableur)
  dicoNFcategorie = {}
  # fréquence des catégories par NF
  for ligne in lignes :
    infos = ligne.split("\t")      
    if len(infos) > 9 :
      NF = infos[6].lower()
      categorie = infos[3]
      if NF not in dicoNFcategorie :
        dicoNFcategorie[NF] = {}
        dicoNFcategorie[NF][categorie] = 1 
      else :
        if categorie in dicoNFcategorie[NF] :
          dicoNFcategorie[NF][categorie] += 1
        else :
          dicoNFcategorie[NF][categorie] = 1
  # choix categorie majoritaire
  dicoNFcategorieMAX = {}
  for NF in dicoNFcategorie :
    freqMax = 0
    categorieMAX = ""
    for categorie in dicoNFcategorie[NF] :
      if dicoNFcategorie[NF][categorie] > freqMax:
        freqMax = dicoNFcategorie[NF][categorie] 
        categorieMAX = categorie
    dicoNFcategorieMAX[NF] = categorieMAX
  return dicoNFcategorieMAX

def NF2POSnorm_POS_lemma(tableur,langue):
  dicoNF_POSnorm = {}        
  dicoNF2POSnorm = {}
  dicoLemmePOSNF = {}        
  lignes = infosTableur(tableur)
  for ligne in lignes :
    infos = ligne.split("\t")      
    if len(infos) > 9 :
      NF = infos[6].lower()                                             
      POSnorm = ";".join(sorted(infos[9].split("|")))
      if NF not in dicoNF2POSnorm :
        dicoNF2POSnorm[NF] = [POSnorm]
      else :
        if POSnorm not in dicoNF2POSnorm[NF] :
          dicoNF2POSnorm[NF] += [POSnorm]
      lemmes = infos[8].lower()              
      listeLemmes = lemmes.split("|")  
      # ATTENTION : ajout pour FR : pronoms mal lemmatisés:
      if langue == "FR" :
        new_lemmesEltLex = []
        for l in range(0,len(listeLemmes)) :
          if infos[3] == "IRV" and listeLemmes[l] in ["me","te","le","lui","nous","vous"]:
            newl = "se"
            new_lemmesEltLex.append(newl)                            
          elif listeLemmes[l] in ["me","te","nous","vous"]:
            newl = "se"
            new_lemmesEltLex.append(newl)                            
          else :
            newl = listeLemmes[l].lower()
            new_lemmesEltLex.append(newl)
        listeLemmes = new_lemmesEltLex
      #-------------------------------------
      listePOS = infos[9].split("|")
      lemmePOS = list(zip(listeLemmes,listePOS))
      if NF not in dicoLemmePOSNF :
        dicoLemmePOSNF[NF] = {}
        for elt in lemmePOS :
          dicoLemmePOSNF[NF][elt[0]] = [elt[1]]
      else :
        old = dicoLemmePOSNF[NF]
        for lem in old :
          for elt in lemmePOS :
            if lem == elt[0] and elt[1] not in  dicoLemmePOSNF[NF][lem] :
              dicoLemmePOSNF[NF][lem] += [elt[1]]        
  return dicoNF2POSnorm,dicoLemmePOSNF

def NF2Vars(tableur,dicoPOSnorm2Vars,dicoNF2POSnorm,dicoNF2LemmePOS,listePOSnormTopX): 
  """
  pour toutes les NFs => variantes lemmes générées
  """
  dicoGenerLemmes = {}
  variantesPOS_NF = {}
  #Pour chaque NF annotée du tableur
  for NF in dicoNF2POSnorm :      
    generLemm = []
    listePOSnormObservees = dicoNF2POSnorm[NF]
    POSnormDejaTraitees = []
    for POSnorm in listePOSnormObservees :
      if POSnorm not in POSnormDejaTraitees :
        POSnormDejaTraitees.append(POSnorm)
        # si POSnorm de la NF est dans le top X 
        if POSnorm in listePOSnormTopX : 
          variantesPOSnorm = dicoPOSnorm2Vars[POSnorm]
          variantesPOS_NF[NF] = variantesPOSnorm
          # on génère les lemmes correspondants
          generLemms = var2lemme(variantesPOSnorm,dicoNF2LemmePOS,NF)
          if NF not in dicoGenerLemmes :
            dicoGenerLemmes[NF] =  generLemms  
          else :  
            newGenerLemms = []
            oldGenerLemms = dicoGenerLemmes[NF]
            for generLemm in generLemms :
              if generLemm not in oldGenerLemms :
                newGenerLemms = oldGenerLemms + [generLemm]  
              else :
                newGenerLemms = oldGenerLemms
            dicoGenerLemmes[NF] = newGenerLemms
  return dicoGenerLemmes,variantesPOS_NF

def idPhrase2MWE(tableur,listePOSnormTopX) : 
  """
  INPUT = tableur Train, liste des POSnorm sélectionnées 
  OUTPUT : dico clé = idPhrase Train, valeur = liste des MWE annotées respectant les POSnorm autorisées e.g. {fr-ud.... : ['41|43','50|52|53']
  Utilisé pour : Attribution du label IDIOMAT, postraitement des candidats Train éventuellement oubliés
  """
  dicoIdPhrase_ToksAnnots = {}
  dicoIDIOMAT = {}
  lignes = infosTableur(tableur)
  nbMWEinTrain = len(lignes) - 1
  for ligne in lignes :
    infos = ligne.split("\t")    
    if len(infos) > 9 :
      idPhrase = infos[0].split("_MWE")[0]
      tokAnnots = infos[5]
      POSnorm = ";".join(sorted(infos[9].split("|")))      
      if idPhrase not in dicoIdPhrase_ToksAnnots :
        dicoIdPhrase_ToksAnnots[idPhrase] = [tokAnnots]
      else :
        dicoIdPhrase_ToksAnnots[idPhrase] += [tokAnnots]      
      if POSnorm in listePOSnormTopX : 
        idMWE = str(infos[0]) + "_" + str(tokAnnots)
        dicoIDIOMAT[idMWE] = "\t".join(infos[1:])
  return dicoIdPhrase_ToksAnnots,nbMWEinTrain,dicoIDIOMAT

# ---- Génération des lemmes à partir des NF connues et des alternances de patrons observées SSI freq patron POSnorm dans TOPX
def var2lemme(listeVars,dicoNF2LemmePOS,NFrecherche):
  """
  Input = ["PRON|PRON|VERB","PRON|VERB|VERB"... ]
          dico {"avoir;il;y : {'avoir' : [AUX,VERB]}{'y' : PRON|ADV} {'il' : PRON}}
  Output =  ["il|y|avoir","y|il|avoir"..]       
  """
  #listeVars = ast.literal_eval(listeVars)
  allLemmesCandidat = []
  for NF in dicoNF2LemmePOS :
    if NF == NFrecherche :  
      POSelts = dicoNF2LemmePOS[NF]
      # Combinaison possibles :
      listeTuples = []
      for lemme in POSelts :
        POSpossibles = POSelts[lemme]
        for POSpossible in POSpossibles :   
          listeTuples.append((lemme,POSpossible))      
      listeTuplesPermutations = list(itertools.permutations(listeTuples))
      #Verif si combinaisons font partie des transfos déjà observées :                        
      for listeTuples in listeTuplesPermutations : 
        ALL_listeTuplesSansDoublon = permutations(listeTuples,len(NF.split(";")))        
        #/!\ si 1 meme elt a plusieurs POS possibles => choix POSnorm + fréquent
        for permut in list(ALL_listeTuplesSansDoublon) :
          POS_genere = ""
          lemme_genere = ""       
          for elt in permut :  
            if POS_genere == "" :
              POS_genere = elt[1]
              lemme_genere = elt[0]
            else :
              POS_genere = POS_genere + "|" + elt[1]
              lemme_genere = lemme_genere + "|" + elt[0]
          
          if POS_genere in listeVars and NF == ";".join(sorted(lemme_genere.split("|"))) and lemme_genere not in allLemmesCandidat  :
            allLemmesCandidat.append(lemme_genere)        
  return allLemmesCandidat


def infosPhrase_to_lemmes_numTok(phrase,langue):
  """
  lemmatise la phrase (minuscules)
  """
  lemmesPhrase = []
  numToksPhrase = []
  for ligne in phrase :
    infos = ligne.split("\t")
    if len(infos) > 2 :
      numToksPhrase.append(infos[0])
      if langue != "FR" : 
        lemmesPhrase.append(infos[2].lower())  
      else :
        # prise en compte de pb de lemmatisation UNIQUEMENT pour phrases annotées 
        # sinon il faudrait faire la différence entre "vous vous rappelez" vs "je vous rappele"
        if infos[2] in ["me","te","nous","vous"] and infos[9] != "*":
          # ie si expression annotee et pronom mal lemmatisé :
          lemmesPhrase.append("se") 
        else :
          lemmesPhrase.append(infos[2].lower())         
  return lemmesPhrase,numToksPhrase

# ---- Ecriture Tableur Candidats avec posttraitement
def NEW_postTrait_IdiomatManquants(langue,dicoIdiomat,listeCandidats,t_CandidatsCORR):
  """
  pour FR  : pb lié aux pronoms mal lemmatisés => pas détectés lors de recherche de candidats
  """  
  with open(t_CandidatsCORR,"a") as f_out :        
        message = str(langue) + "-train: postTrait missing Candidates"
        pbar = tqdm(dicoIdiomat,desc=message)           
        for idMWE in pbar :
          MWE_idPhrase_numToks = re.match(r'(.+?)_(MWE\d+?)_(.+?$)',idMWE)
          idPhrase = MWE_idPhrase_numToks.groups()[0]
          numMWE = MWE_idPhrase_numToks.groups()[1]
          numToks = MWE_idPhrase_numToks.groups()[2] 
          verifCand = str(idPhrase) + "_" + str(numToks)
          if verifCand not in listeCandidats :
            idMWE_postTrait = idPhrase + "_Check_" + str(numMWE)
            infos = dicoIdiomat[idMWE].split("\n")[0]
            f_out.write(str(idMWE_postTrait) + "\t" + str(infos) + "\tIDIOMAT\n")

def OLD_postTrait_IdiomatManquants(listePOSnormTopX,dicoPOSnorm2Vars,t_Idiomat,t_Candidats,t_CandidatsCORR):
  """
  pour FR  : pb lié aux pronoms mal lemmatisés => pas détectés lors de recherche de candidats
  """  
  with open(t_Idiomat,"r") as f_MWE, open(t_Candidats,"r") as f_cand,  open(t_CandidatsCORR,"a") as f_out :
        lignesMWE = f_MWE.readlines()
        lignesCand = f_cand.readlines()
        message = str(langue) + "-train: postTrait missing Candidates"
        pbar = tqdm(lignesMWE[1:],desc=message)           
        for ligne in pbar :
          POSnormAextraire = False
          isInCandidats = False
          isPhraseInCandidats = False          
          infosMWE = ligne.split("\t")
          idPhraseMWE = infosMWE[0].split("_MWE")[0] 
          numToksMWE =  infosMWE[5]
          POSnorm = ";".join(sorted(infosMWE[9].split("|")))  
          if POSnorm in listePOSnormTopX :
            POSnormAextraire = True   
          numMaxMWE = -1
          for ligneCand in lignesCand[1:] :
            infosCand = ligneCand.split("\t")
            idPhraseCand = infosCand[0].split("_MWE")[0]          
            numToksCand =  infosCand[5]
            if idPhraseCand == idPhraseMWE :
              isPhraseInCandidats = True
              numMWE = infosCand[0].split("_MWE")[1] 
              if int(numMWE) > numMaxMWE :
                numMaxMWE = int(numMWE)
              if numToksMWE == numToksCand :
                isInCandidats = True
          if not isInCandidats :
            # si MWE ne fait pas partie des candidats extraits :
            if not isPhraseInCandidats and POSnormAextraire:
              for i in range(0,(len(infosMWE)-1)) :
                f_out.write(infosMWE[i] + "\t")
              derniereCol = infosMWE[len(infosMWE)-1]
              derniereCol = derniereCol.split("\n")[0]
              f_out.write(str(derniereCol) + "\tIDIOMAT\n")
            elif isPhraseInCandidats and POSnormAextraire: 
              numMWE = str(idPhraseMWE) + "_MWE" + str(numMaxMWE+1)
              col1 = str(numMWE)
              colSuivantes = infosMWE[1:]
              f_out.write(str(numMWE) + "\t")
              for col in range(0,len(colSuivantes)-1) :
                f_out.write(str(colSuivantes[col]) + "\t")
              derniereCol = colSuivantes[len(colSuivantes)-1]
              derniereCol = derniereCol.split("\n")[0]
              f_out.write(str(derniereCol) + "\tIDIOMAT\n")     
    
def EcritCandidatsExtr(cupt,langue,corpus,dicoMorpho,listePOS,listeRelDep,dicoNFcategories,filtre,filterMaxInsert):
  """
  corpus = TRAIN ou DEV ou TEST
  cupt = le fichier cupt correspondant        
  """
  if not filtre : 
    fichierCandidatsExtr_TEMP = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/" + str(corpus) + "_Candidats_temp.tsv"
    nomTableur =  str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue) + "/" + str(corpus) + "_Candidats.tsv"
  else :
    fichierCandidatsExtr_TEMP = str(repParent) + "/OUTPUT/CANDIDATS/" + str(langue) + "/filter" + str(filterMaxInsert) + "_" + str(corpus) + "_Candidats_temp.tsv"
    nomTableur =  str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue)  + "/filter" + str(filterMaxInsert) + "_" + str(corpus) + "_Candidats.tsv"
  if langue in ["IT","SL"] :
    colonnePOS = 4
  else :
    colonnePOS = 3      
  dico_NFinCorpus = {}						
  dico_allInfosNFinCorpus = {}					# all infos utiles TrC	
  listeCand = [] # liste des candidats extraits avec idPhrase_numToks (e.g. [frud123_41|45, ...] utile pour postTrait)
  nbIDIOMATextr = 0
  # Candidats extraits :
  with open(fichierCandidatsExtr_TEMP,"r") as f_CandExtr :
    candidatsExtraits = {}
    lignes = f_CandExtr.readlines()
    message = str(langue) + "-" + str(corpus) + ": Extract Infos Candidates"
    pbar = tqdm(lignes,desc=message)      
    for ligne in pbar :   #fr-ud-train_06807	air;avoir	17|20	LITERAL
      infos = ligne.split("\t")
      idPhrase = infos[0]
      NF_cand = infos[1]
      positionToks = infos[2].split("|")
      label = infos[3].split("\n")[0]      
      if label == "IDIOMAT" :
        nbIDIOMATextr += 1
        idPhrase_NumToks = str(idPhrase) + "_" + str(infos[2])
        listeCand.append(idPhrase_NumToks)
      if idPhrase not in candidatsExtraits :
        candidatsExtraits[idPhrase] = [(positionToks,label,NF_cand)]
      else :
        candidatsExtraits[idPhrase] += [(positionToks,label,NF_cand)]

  #===================================================================
  #		extract infos 
  #===================================================================	

  # Creation Tableur & ecriture entete colonnes
  #nomTableur =  str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue) + "/" + str(corpus) + "_Candidats.tsv"
  with open  (nomTableur,"a") as f_Tableur : 
    f_Tableur.write("idphrase_NUMmwe\tphrase\tFenêtre annotation\t" + "type MWE\tnb Elts Lex\tPosition Elts Lex\t" +   "NF\tMWE : flexion\tMWE : lemmes\t" + "MWE : POS"  + "\t" +  "insertions : flexion\tinsertions : lemmes" +	"\tinsertions : POS" + "\t" +  "Nb insertions\tNb discontinuités\t" + "Synthese insertions" + "\t" +    "Synthese insertions sans Doublons" + "\t" + "Morpho Elts Lex" + "\t" +   "Synthèse dépendances Sans Doublons dans l'ordre d'apparition des elts lex\t" + "distance Syntaxiq V(...)N\tType Distance Syntaxique V(...)N\t" + "distance Syntaxiq 2Elts\tType Distance Syntaxique 2Elts\tRel Dependance Entrante\t" + "Label\n")        
    # Extract infos des MWE annotées du fichier cupt :        
    with open (cupt,"r") as f_cupt :
      infosCupt =  f_cupt.readlines()	
      corpusCupt = "".join(infosCupt[1:])					
      phrases = corpusCupt.split("\n\n")[:-1]
      # Init : Typologie des insertions que l'on souhaite comparer
      # Extraction POS, relDep et morpho spécifiques à la langue (d'après TrC car de taille supérieure) :
      """dicoMorpho,listePOS,listeRelDep = typo_POS_dep_Morpho (langue)   """
      # Infos Elts Lex :            
      for phrase in phrases :              
        col3_fenetre = {}  
        col4_typeMWE = {}
        col6_positionEltsMwe = {}
        col7_EltLexFlex = {}						
        col8_EltLexLemme = {}
        col9_EltLexPOS = {}
        #col10_NF = {}
        col11_insertFlex = {}
        col12_insertLemme =  {}
        col13_insertPOS =  {}
        col14_nbInsert =  {}
        col15_nbDiscont =  {}
        col15_Insert = {}
        col15_InsertSansDoublons = {}
        col16_Morpho =  {}	
        col16_AllMorpho = {}
        col17_dep_eltsLex_flechi =  {}
        col18_dep_eltsLex_lemme =  {}
        col18_ok = []
        col19_dep_eltsLex_pos =  {}		
        col20_RelDepSansDoublon =  {}   
        col25_RelDepEntrante = {}        
        col16_listeMorpho = []
        #col21_Label = {}
        lignes = phrase.split("\n")
        for ligne in lignes :
          if "# source_sent_id" in ligne :
            sourcePhrase = ligne.split("# source_sent_id = ")[1]
            idPhrase = sourcePhrase.split(" ")[2]     #email-enronsent34_01-0024  
          elif "# text = " in ligne :
            textPhrase =  ligne.split("# text = ")[1] #"A little birdie told me..."      
            col2_phrase = textPhrase
          elif ligne[0] != "#" :              
            infos = ligne.split("\t")           
            # caractéristiques des elements extraits  (potentiellement EltLex)
            if idPhrase in candidatsExtraits:   
              nbMWEs = len(candidatsExtraits[idPhrase]) 
              candidatsExtr = tupleOrdreCroissant(candidatsExtraits[idPhrase])
              for idMWE in range(1,nbMWEs+1):
                newId = "MWE" + str(idMWE) 
                positionToksAnnot = candidatsExtr[idMWE-1][0]
                for tok in positionToksAnnot :
                  if tok == str(infos[0]) : 
                    if idMWE not in col6_positionEltsMwe :
                      col6_positionEltsMwe[idMWE] = infos[0]
                      col7_EltLexFlex[idMWE] = infos[1]
                      col8_EltLexLemme[idMWE] = infos[2]
                      col9_EltLexPOS[idMWE] = infos[colonnePOS]  
                    else :
                      col6_positionEltsMwe[idMWE] = col6_positionEltsMwe[idMWE] + "|" + infos[0] 
                      col7_EltLexFlex[idMWE] = col7_EltLexFlex[idMWE] + "|" +infos[1]
                      col8_EltLexLemme[idMWE] =  col8_EltLexLemme[idMWE] + "|" +infos[2]
                      col9_EltLexPOS[idMWE] = col9_EltLexPOS[idMWE] + "|" + infos[colonnePOS] 
        # ====================================================================
        #	Recherche des insertions / discontinuités
        # ====================================================================        
        # ==== Pour chaque MWE de la phrase =======
        #-------------- Extraction fenêtre annotation :----------
        for MWE in col6_positionEltsMwe :           #{'1': '2|3', '2': '10|12'}                     
          #Initialisations : 
          dicostatsInsertions = initInsertions(listePOS)                     
          dicostatsInsertionsSansDoublons = initInsertions(listePOS)                      
          idMWE = "MWE" + str(MWE)
          positionEltsMwe = col6_positionEltsMwe[MWE].split("|")  #['1', '2', '3', '4']
          positionDebut = positionEltsMwe[0]
          positionFin = positionEltsMwe[len(positionEltsMwe)-1]
          insertions_flechi = ""
          insertions_lemme = ""
          insertions_POS = ""	
          newFenetre = ""
          newDicoFenetre = {}
          nbDiscont = 0
          nbInsertions = 0
          dicoStatsInsertions = {}      
          #Initialisation 
          MorphoMWE = []
          relDepMWE = []    
          relDepEntrante = []
          for ligne in lignes :                           
            infos = ligne.split("\t")   
            # NB : lignes avec amalgames (du = de+le) , eg 7-8	du
            #=> non pris en compte 
            if len(infos) > 10 and "-" not in infos[0]: 
              if int(infos[0]) >= int(positionDebut) and int(infos[0]) <= int(positionFin) :
                if newFenetre != "" :
                  if infos[0] not in positionEltsMwe : 
                    newFenetre = newFenetre + " " + infos[1]
                  else :
                    newFenetre = newFenetre + " [" + infos[1] + "]"
                else :
                  if infos[0] not in positionEltsMwe : 
                    newFenetre = infos[1]  
                  else:
                    newFenetre = "[" + infos[1] + "]"
              for elt in range(int(positionDebut),int(positionFin) + 1) : 
                elt_lex = False 
                dicoRelDepElt = {}
                for num_eltLex in positionEltsMwe :	#['1', '2', '3', '4']
                  if int(elt) == int(num_eltLex) :	
                    elt_lex = True     
                if int(elt) == int(infos[0]) : 
                  elt_POS = infos[colonnePOS]
                  if not elt_lex :  
                    if str(elt -1) not in positionEltsMwe : 
                      insertions_flechi = insertions_flechi + "--" + infos[1]
                      insertions_lemme = insertions_lemme + "--" +  infos[2]
                      insertions_POS = insertions_POS + "--" +  infos[colonnePOS] 
                    else :
                      if int(elt) == int(positionDebut) + 1 :
                        insertions_flechi = infos[1] 
                        insertions_lemme = infos[2]
                        insertions_POS = infos[colonnePOS]
                      else:
                        insertions_flechi = insertions_flechi + "|" + infos[1] 
                        insertions_lemme = insertions_lemme + "|"  +  infos[2]
                        insertions_POS = insertions_POS + "|" +  infos[colonnePOS] 
                    # Stats par type d'insertion 				
                    dicostatsInsertions[elt_POS] += 1  
                    if (dicostatsInsertionsSansDoublons[elt_POS]) == 0:                                                                                                                         
                      dicostatsInsertionsSansDoublons[elt_POS] = 1 
                  else :                       
                    #------------ infos Morpho --------------------
                    dicoMorphoElt = {}
                    infosMorpho = infos[5].split("|") #e.g. Case=Acc|Number=Sing|Person=1|PronType=Prs  
                    #initialisation                                        
                    for propriete in dicoMorpho :
                      dicoMorphoElt[propriete] = "_"                                         
                    for infoMorpho in infosMorpho :                 # Turc -> ['A3pl', 'Past', 'Pos']
                      if "=" in infoMorpho :
                        propriete = infoMorpho.split("=")[0]
                        valeur = infoMorpho.split("=")[1]
                        dicoMorphoElt[propriete] = valeur
                      elif infoMorpho == "_" :
                        dicoMorphoElt["_"] = "_"
                      else :                                              
                        dicoMorphoElt[infoMorpho] = infoMorpho          # pour le turc => {'A3pl' : 'A3pl'}
                    MorphoMWE.append(dicoMorphoElt)
                    #------------ Relations de Dépendance  ENTRANTE
                    relDepEntrante.append(infos[7])            
                    #------------ Relations de Dépendance  --------------------                                                                                
                    relDep = infos[8]
                    #initialisation
                    for propriete in listeRelDep :
                      dicoRelDepElt[propriete] = 0      
                    # --- éléments dépendant de l'elt lex  SANS DOUBLONS ---
                    for l in lignes :
                      infos2 = l.split("\t")
                      if len(infos2) == 11 and "-" not in infos2[0] and infos2[6] not in ["_","-"] and infos2[7] not in ["_","-"] :

                        if int(infos2[6]) == int(elt) and dicoRelDepElt[infos2[7]] == 0:  #PB 2018-04-30 : builtins.KeyError: 'obl:arg'
                          dicoRelDepElt[infos2[7]] = 1

                    dicoRelDepElt = sorted(dicoRelDepElt.items(), key=lambda t: t[0])
                    relDepMWE.append(dicoRelDepElt)   
          col3_fenetre[MWE] = newFenetre  
          col11_insertFlex[MWE] =  insertions_flechi
          col12_insertLemme[MWE] =  insertions_lemme
          col13_insertPOS[MWE] =  insertions_POS   
          col25_RelDepEntrante[MWE] = relDepEntrante

          nbDiscont = len(insertions_flechi.split("|")) - 1 
          for inserts in insertions_flechi.split("|"):
            if inserts != [''] :
              insert = inserts.split("--")
              if insert != [''] :
                nbInsertions += len(insert)
          col14_nbInsert[MWE] = nbInsertions
          col15_nbDiscont[MWE] = nbDiscont        
          # avec nombre exact d'insertions
          statsInsertions = sorted(dicostatsInsertions.items(), key=lambda t: t[0])                        
          col15_Insert[MWE] = statsInsertions
          # Sans doublons d'insertions
          statsInsertionsSansDoublons = sorted(dicostatsInsertionsSansDoublons.items(), key=lambda t: t[0])      
          col15_InsertSansDoublons[MWE] = statsInsertionsSansDoublons
          # Relations de dépendances sortantes :
          col20_RelDepSansDoublon[MWE] = relDepMWE  
          # morpho : 
          col16_Morpho[MWE] =  MorphoMWE  
          col16_listeMorpho.append(MorphoMWE)    
        #sorted Dicos par num MWE croissant :
        col3_fenetre = sorted(col3_fenetre.items(), key=lambda t: t[0])  #[('1', 'called in'), ('2', 'pick it up')]
        col4_typeMWE = sorted(col4_typeMWE.items(), key=lambda t: t[0])    
        col6_positionEltsMwe = sorted(col6_positionEltsMwe.items(), key=lambda t: t[0])     #{'1': '2|3', '2': '10|12'}
        col7_EltLexFlex = sorted(col7_EltLexFlex.items(), key=lambda t: t[0])         #{'1': 'called|in', '2': 'pick|up'} 
        col8_EltLexLemme = sorted(col8_EltLexLemme.items(), key=lambda t: t[0])             #{'1': 'call|in', '2': 'pick|up'}
        col9_EltLexPOS = sorted(col9_EltLexPOS.items(), key=lambda t: t[0])                 #{'1': 'VERB|ADP', '2': 'VERB|ADP'} 
        col11_insertFlex = sorted(col11_insertFlex.items(), key=lambda t: t[0])   #[('1', ''), ('2', 'it')]
        col12_insertLemme = sorted(col12_insertLemme.items(), key=lambda t: t[0])   #[('1', ''), ('2', 'it')]
        col13_insertPOS = sorted(col13_insertPOS.items(), key=lambda t: t[0])   #[('1', ''), ('2', 'it')]
        col15_nbDiscont = sorted(col15_nbDiscont.items(), key=lambda t: t[0])               #[('1', 0), ('2', 0)]
        col14_nbInsert = sorted(col14_nbInsert.items(), key=lambda t: t[0])                 #[('1', 2), ('2', 3)]
        col15_Insert = sorted(col15_Insert.items(), key=lambda t: t[0])   #[('1', [('ADJ', 0),...]), ('2', [... ('PRON', 1)..]
        col15_InsertSansDoublons = sorted(col15_InsertSansDoublons.items(), key=lambda t: t[0])   #[('1', [('ADJ', 0),...]), ('2', [... ('PRON', 1)..]
        col16_Morpho = sorted(col16_Morpho.items(), key=lambda t: t[0])  
        col20_RelDepSansDoublon = sorted(col20_RelDepSansDoublon.items(), key=lambda t: t[0])
        col25_RelDepEntrante = sorted(col25_RelDepEntrante.items(), key=lambda t: t[0])
        #============================================================
        #     distance syntaxique : uniquement si 1 VB/AUX et 1 NOUN
        #============================================================                
        col21_distSyn = {}
        col22_typeDistSyn = {}
        # Verif Distance syntaxique 2/3 elts : VERB (prep/det/..) NOUN
        for MWE in range(0,len(col9_EltLexPOS)) :
          #init
          col21_distSyn[MWE] = -1
          col22_typeDistSyn[MWE] = ""                   
          verifDistSyn = False
          eltLex_liste = col9_EltLexPOS[MWE][1].split("|")
          positionEltLex_liste = col6_positionEltsMwe[MWE][1].split("|")
          comptVERB = 0
          comptNOUN = 0 
          for elt in eltLex_liste :
            if elt == "VERB" or elt == "AUX" :
              comptVERB += 1
            elif elt == "NOUN" :
              comptNOUN += 1
          if comptNOUN == 1 and  comptVERB == 1 :
            verifDistSyn = True
          if verifDistSyn :
            for elt in range(0,len(eltLex_liste)) :
              if eltLex_liste[elt] == "VERB" or eltLex_liste[elt] == "AUX" :
                positionVERB = int(positionEltLex_liste[elt])
              elif eltLex_liste[elt] == "NOUN" :
                positionNOUN = int(positionEltLex_liste[elt]) 
            col21_distSyn[MWE],col22_typeDistSyn[MWE] = distanceSyntaxique(positionVERB,positionNOUN,phrase)
        # calcul distSyn 2 elts quels qu'ils soient
        col23_distSyn2elts = {}
        col24_typeDistSyn2elts = {}      
        for MWE in range(0,len(col9_EltLexPOS)) :   
          #init                    
          col23_distSyn2elts[MWE] = -1
          col24_typeDistSyn2elts[MWE] = ""                           
          positionEltLex_liste = col6_positionEltsMwe[MWE][1].split("|")
          if len(positionEltLex_liste) == 2 :                    
            positionElt1 = int(positionEltLex_liste[0])
            positionElt2 = int(positionEltLex_liste[1])  
            col23_distSyn2elts[MWE],col24_typeDistSyn2elts[MWE] = distanceSyntaxique(positionElt1,positionElt2,phrase)                
        #-------------------------------
        col21_distSyn = sorted(col21_distSyn.items(), key=lambda t: t[0])
        col22_typeDistSyn = sorted(col22_typeDistSyn.items(), key=lambda t: t[0])
        col23_distSyn2elts = sorted(col23_distSyn2elts.items(), key=lambda t: t[0]) 
        col24_typeDistSyn2elts = sorted(col24_typeDistSyn2elts.items(), key=lambda t: t[0]) 
        # =========================================
        #	Fichier TSV <=> Tableur
        # =========================================
        for MWE in range(0,len(col3_fenetre)): 
          dicoPatronMWE = {}                 
          col1 = str(idPhrase) + "_MWE" + str(MWE+1) #email-enronsent34_01-0024_MWE1  => ok
          col2 = col2_phrase                      #A little birdie told me that ... => ok
          col3 = col3_fenetre[MWE][1]             #A little birdie told      => ok                                  
          col6 = col6_positionEltsMwe[MWE][1]     #1|2|3|4      => ok  
          col8 = col7_EltLexFlex[MWE][1]          #A|little|birdie|told      => ok  
          col9 = col8_EltLexLemme[MWE][1]         #a|little|birdie|tell      => ok 
          col10 = col9_EltLexPOS[MWE][1]           #DET|ADJ|NOUN|VERB      => ok  
          lemmesEltLex = col8.split("|")
          col5 = len(lemmesEltLex)       
          col7 = candidatsExtr[int(MWE)][2]      #NF
          col4 = dicoNFcategories[col7]          #VID 
          col11 = col11_insertFlex[MWE][1]        #
          col12 = col12_insertLemme[MWE][1]       #
          col13 = col13_insertPOS[MWE][1]         #DET--NOUN--ADP
          col14 = col14_nbInsert[MWE][1]          #3
          col15 = col15_nbDiscont[MWE][1]
          col16 = col15_Insert[MWE][1]            #[('ADJ', 0), ('ADP', 0),
          col17 = col15_InsertSansDoublons[MWE][1] #[('ADJ', 0), ('ADP', 0),   
          col18_ok = col16_Morpho[MWE][1]
          col20 = col21_distSyn[MWE][1]
          col21 = col22_typeDistSyn[MWE][1]
          col23 = col23_distSyn2elts[MWE][1]
          col24 = col24_typeDistSyn2elts[MWE][1]
          col25 = col25_RelDepEntrante[MWE][1]
          eltLexPOS = col9.split("|")
          dicoMorphoParElt = {}
          for POS in range(0,len(eltLexPOS)) :
            lemmeElt = col8.split("|")[POS]
            positionElt = col6.split("|")[POS]
            dicoMorphoParElt[str(eltLexPOS[POS]) + "_" + str(lemmeElt) + "_" + str(positionElt)] = col16_Morpho[MWE][1][POS]                    
          col18 = dicoMorphoParElt        #{'ADP': {'Gender': '_', 'Degree': '_'
          col19 = col20_RelDepSansDoublon[MWE][1]  
          col22 = candidatsExtr[int(MWE)][1] #label 
          f_Tableur.write(str(col1) + "\t" + str(col2) + "\t" + \
                                        str(col3) + "\t" + str(col4) + "\t" + str(col5) + "\t" + \
                                        str(col6) + "\t" + str(col7) + "\t" + str(col8) + "\t" + \
                                   str(col9) + "\t" + str(col10) + "\t" + str(col11) + "\t" + \
                                   str(col12) + "\t" + str(col13) + "\t" + str(col14) + "\t" + \
                                   str(col15) + "\t" + str(col16) + "\t" + str(col17) + "\t" +\
                                   str(col18_ok) + "\t" + str(col19) + "\t" + str(col20) + "\t" + \
                                   str(col21) + "\t" + str(col23) + "\t" + str(col24) + "\t" + \
                                   str(col25) + "\t" + str(col22) + "\n")
    
  return nbIDIOMATextr,listeCand