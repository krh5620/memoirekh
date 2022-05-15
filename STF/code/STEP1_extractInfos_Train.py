#!/usr/bin/python3
# -*- coding: utf-8 -*-


#==========================================
#             IMPORTS
#==========================================
import os
import sys
import time
import progressbar
import subprocess
import numpy as np
from collections import OrderedDict
repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])
from cupt2typo_POS_dep_morpho import * 
from distSyn import * 
from tqdm import tqdm
#==========================================
#      FONCTIONS 
#==========================================
def cupt2patternsTopX(NFallPatterns,langue):         
       """
       INPUT = dico pour chaque langue : clé = NF ; valeur =  variantes séquences POS observées {'VERB|NOUN': 2, 'NOUN|VERB': 1}
       OUTPUT = liste du top X des POS norm
       """
       listePatterns = []
       patternsTopX = []
       allPOSnorm = {}   
       for NF in NFallPatterns :                        
              listePOSnormNF = []
              variantesNF = {}              
              variantesNF = NFallPatterns[NF]    
              for var in variantesNF :           
                     POSnorm = ";".join(sorted(var.split("|")))  
                     if POSnorm not in listePOSnormNF : 
                            listePOSnormNF.append(POSnorm)
                            if POSnorm not in allPOSnorm :                                   
                                   allPOSnorm[POSnorm] = variantesNF
                            else :
                                   variantesNF2 = NFallPatterns[NF]
                                   for var2 in variantesNF2 :                            
                                          if var2 in allPOSnorm[POSnorm] :                                             
                                                 allPOSnorm[POSnorm][var2] += variantesNF2[var2]
                                          else:
                                                 allPOSnorm[POSnorm][var2] = variantesNF2[var2] 
       allFreqs = []
       allPatterns = []
       dicoPOSnorm2Vars = {}
       for POSnorm in allPOSnorm: 
              total = 0
              allPatterns.append(POSnorm)                            
              variantes = []
              for var in allPOSnorm[POSnorm] :
                     variantes.append(var)
                     total += allPOSnorm[POSnorm][var]   
              allFreqs.append(total)  
              dicoPOSnorm2Vars[POSnorm] = variantes                 
       freq_pattern = sorted(zip(allFreqs,allPatterns),reverse=True) #[(1978, 'NOUN;VERB'), (1454, 'PRON;VERB'), (298, 'PRON;PRON;VERB'...
       # Estimation de couverture du top X
       allFreqs = sorted(allFreqs,reverse=True)
       seuil10 = allFreqs[9]
       compt = 0
       comptRang = 0
       sommeFreq10 = 0                     
       for freq in allFreqs :  
              if freq >= seuil10 and compt < len(allFreqs) and freq > 1:
                     sommeFreq10 +=  freq
                     comptRang += 1                                                                          
              compt += 1       
       freq10 = sommeFreq10/sum(allFreqs)*100          
       for i in range (0,comptRang) :
              #f_stats.write(str(i+1) + "\t" + str(freq_pattern[i][1]) + "\t" + str(freq_pattern[i][0]) + "\n")
              patternsTopX.append(freq_pattern[i][1])       
       return patternsTopX,dicoPOSnorm2Vars

def initInsertions(listePOS) :
       statsInsertions = {}
       for POS in listePOS :
              statsInsertions[POS] = 0
       return statsInsertions

def trainCupt2tableur(cupt,langue):
       # Adaptation pour IT et SL
       if langue in ["IT","SL"] :
              colonnePOS = 4
       else :
              colonnePOS = 3  
       #===================================================================
       # extract infos in train for annotated MWEs 
       #===================================================================	
       dico_NFinCorpus = {}						
       dico_allInfosNFinCorpus = {}					
       # Entete colonnes du Tableur
       nomTableur =  str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue) + "/train_MWEs.tsv"
       if not os.path.exists(str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue)):
              os.makedirs(str(repParent)  + "/OUTPUT/CANDIDATS/" + str(langue))          
       with open  (nomTableur,"a") as f_Tableur : 
              f_Tableur.write("idphrase_NUMmwe\tphrase\tFenêtre annotation\ttype MWE\tnb Elts Lex\tPosition Elts Lex\t" \
                              + "NF\tMWE : flexion\tMWE : lemmes\tMWE : POS\tinsertions : flexion\tinsertions : lemmes" \
                              +	"\tinsertions : POS\tNb insertions\tNb discontinuités\tSynthese insertions\t" \
                              + "Synthese insertions sans Doublons\tMorpho Elts Lex\tSynthèse dépendances Sans Doublons dans l'ordre d'apparition des elts lex\t" \
                              + "distance Syntaxiq V-N\tType Distance Syntaxique V-N\tdistance Syntaxiq 2elts\tType Distance Syntaxique 2elts\tRelDep entrante\n")        

              # Extract infos des MWE annotées du fichier cupt :        
              with open (cupt,"r") as f_cupt :
                     infosCupt =  f_cupt.readlines()	
                     corpusCupt = "".join(infosCupt[1:])					
                     phrases = corpusCupt.split("\n\n")[:-1]
                     # Init : Typologie des insertions que l'on souhaite comparer
                     # Extraction POS, relDep et morpho spécifiques à la langue :
                     dicoMorpho,listePOS,listeRelDep = cupt2POSdepMorpho(langue) 
                     # Recherche des patrons les + fréquents :
                     NFallPatterns = {}   
                     message = str(langue) + "-MWE Features"
                     pbar = tqdm(phrases,desc=message)     
                     for phrase in pbar:                        
                            #for numPhrase in progressbar.progressbar(range(len(phrases))):  
                            #phrase = phrases[numPhrase]
                            col3_fenetre = {}  
                            col4_typeMWE = {}
                            col6_positionEltsMwe = {}
                            col7_EltLexFlex = {}						
                            col8_EltLexLemme = {}
                            col9_EltLexPOS = {}
                            col11_insertFlex = {}
                            col12_insertLemme =  {}
                            col13_insertPOS =  {}
                            col14_nbInsert =  {}
                            col15_nbDiscont =  {}
                            col15_Insert = {}
                            col15_InsertSansDoublons = {}
                            col16_Morpho =  {}	
                            col16_listeMorpho = []
                            col16_AllMorpho = {}
                            col17_dep_eltsLex_flechi =  {}
                            col18_dep_eltsLex_lemme =  {}
                            col19_dep_eltsLex_pos =  {}		
                            col20_RelDepSansDoublon =  {}   
                            col25_RelDepEntrante = []
                            lignes = phrase.split("\n")
                            for ligne in lignes :
                                   if "# source_sent_id" in ligne :
                                          sourcePhrase = ligne.split("# source_sent_id = ")[1]
                                          idPhrase = sourcePhrase.split(" ")[2]     
                                   elif "# text = " in ligne :
                                          textPhrase =  ligne.split("# text = ")[1]  
                                          col2_phrase = textPhrase
                                   elif ligne[0] != "#" :              
                                          infos = ligne.split("\t")
                                          if len(infos) != 11 :
                                                 print("ERREUR : colonne manquante dans Phrase = ") + str(idPhrase)
                                          else :                                  
                                                 idMWEs = []   
                                                 typeMWEs = []
                                                 if infos[10] not in ["*","_"]  :
                                                        eltLex = True
                                                        #  Overlap => 1:VID;2:IAV   /  1;2      vs      pas d'overlap =>  1:VID /  1            
                                                        if ";" not in infos[10] and ":"  not in infos[10] :                                    
                                                               idMWEs.append(infos[10])
                                                        elif ";" not in infos[10] :
                                                               idMWEs.append(infos[10].split(":")[0])
                                                               typeMWEs.append(infos[10].split(":")[1])
                                                        elif ":" not in infos[10] :
                                                               idMWEs = infos[10].split(";")
                                                        else :                                    
                                                               temp_idMWEs = infos[10].split(";")
                                                               for temp in temp_idMWEs :
                                                                      idMWEs.append(temp.split(":")[0])
                                                                      typeMWEs.append(infos[10].split(":")[1])
                                                 else:
                                                        eltLex = False                                                       

                                                 # caractéristiques de l'Elt Lex   
                                                 if eltLex :
                                                        for idMWE in idMWEs :  
                                                               newId = "MWE" + str(idMWE)
                                                               if idMWE not in col6_positionEltsMwe :
                                                                      col6_positionEltsMwe[idMWE] = infos[0]
                                                                      col7_EltLexFlex[newId] = infos[1]
                                                                      col8_EltLexLemme[newId] = infos[2]
                                                                      col9_EltLexPOS[newId] = infos[colonnePOS]  
                                                               else :
                                                                      col6_positionEltsMwe[idMWE] = col6_positionEltsMwe[idMWE] + "|" + infos[0] 
                                                                      col7_EltLexFlex[newId] = col7_EltLexFlex[newId] + "|" +infos[1]
                                                                      col8_EltLexLemme[newId] =  col8_EltLexLemme[newId] + "|" +infos[2]
                                                                      col9_EltLexPOS[newId] = col9_EltLexPOS[newId] + "|" + infos[colonnePOS] 
                            for MWE in col8_EltLexLemme :  
                                   idMWE = "MWE" + str(MWE) 
                            # ====================================================================
                            #	Recherche des insertions / discontinuités
                            # ====================================================================                             
                            # ==== Pour chaque MWE de la phrase =======
                            #-------------- Extraction fenêtre annotation :----------
                            for MWE in col6_positionEltsMwe :           #{'1': '2|3', '2': '10|12'}
                                   #------------------- Initialisations -----------------------------
                                   dicostatsInsertions = initInsertions(listePOS)                     
                                   dicostatsInsertionsSansDoublons = initInsertions(listePOS)  
                                   #------------------------------------------------------------------
                                   idMWE = "MWE" + str(MWE)
                                   positionEltsMwe = col6_positionEltsMwe[MWE].split("|")  #['1', '2', '3', '4']
                                   positionDebut = positionEltsMwe[0]
                                   positionFin = positionEltsMwe[len(positionEltsMwe)-1]
                                   insertions_flechi = ""
                                   insertions_lemme = ""
                                   insertions_POS = ""	
                                   newFenetre = ""
                                   nbDiscont = 0
                                   nbInsertions = 0
                                   dicoStatsInsertions = {}   
                                   MorphoMWE = []
                                   relDepMWE = []                                          
                                   for ligne in lignes :                           
                                          infos = ligne.split("\t")   
                                          if len(infos) > 10 and "-" not in infos[0]:  # e.g. amalgames 7-8	du
                                                 if infos[0] == "_" :
                                                        print(infos)
                                                 if int(infos[0]) >= int(positionDebut) and int(infos[0]) <= int(positionFin) :
                                                        if newFenetre != "" :
                                                               newFenetre = newFenetre + " " + infos[1]
                                                        else :
                                                               newFenetre = infos[1]                              
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
                                                                             if int(elt) == int(positionDebut) +1 :
                                                                                    insertions_flechi = infos[1] 
                                                                                    insertions_lemme = infos[2]
                                                                                    insertions_POS = infos[colonnePOS]
                                                                             else:
                                                                                    insertions_flechi = insertions_flechi + "|" + infos[1] 
                                                                                    insertions_lemme = insertions_lemme + "|"  +  infos[2]
                                                                                    insertions_POS = insertions_POS + "|" +  infos[colonnePOS] 
                                                                      # Stats par type d'insertion 			
                                                                      dicostatsInsertions[elt_POS] += 1  
                                                                      if dicostatsInsertionsSansDoublons[elt_POS] == 0:
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
                                                                      col25_RelDepEntrante.append(infos[7])                                                                      
                                                                      #------------ Relations de Dépendance  SORTANTES 
                                                                      relDep = infos[8]
                                                                      #initialisation
                                                                      for propriete in listeRelDep :
                                                                             dicoRelDepElt[propriete] = 0      
                                                                      # --- éléments dépendant de l'elt lex  SANS DOUBLONS ---
                                                                      for l in lignes :
                                                                             infos2 = l.split("\t")
                                                                             if len(infos2) == 11 and "-" not in infos2[0] and infos2[6] not in ["_","-"] and infos2[7] not in ["_","-"] :

                                                                                    if int(infos2[6]) == int(elt) and dicoRelDepElt[infos2[7]] == 0:
                                                                                           dicoRelDepElt[infos2[7]] = 1
                                                                      #---- catégorie MWE ----  
                                                                      if ";" not in infos[10] and ":" in infos[10] :
                                                                             #1:VID
                                                                             idMWE2 = "MWE" + str(infos[10].split(":")[0])
                                                                             if idMWE2 not in col4_typeMWE :
                                                                                    col4_typeMWE[idMWE2] = infos[10].split(":")[1]
                                                                      elif  ";" in infos[10] :                                             
                                                                             #1:VID;2:LVC.full
                                                                             #1;2:LVC.full
                                                                             alltypes = infos[10].split(";")
                                                                             for typeMWE in alltypes :
                                                                                    if ":" in typeMWE :                                                
                                                                                           idMWE2 = "MWE" + str(typeMWE.split(":")[0])
                                                                                           col4_typeMWE[idMWE2] = typeMWE.split(":")[1]
                                                                                    else :   
                                                                                           idMWE2 = "MWE" + str(typeMWE)
                                                                      dicoRelDepElt = sorted(dicoRelDepElt.items(), key=lambda t: t[0])
                                                                      relDepMWE.append(dicoRelDepElt)   
                                   col3_fenetre[idMWE] = newFenetre  
                                   col11_insertFlex[idMWE] =  insertions_flechi
                                   col12_insertLemme[idMWE] =  insertions_lemme
                                   col13_insertPOS[idMWE] =  insertions_POS
                                   nbDiscont = len(insertions_flechi.split("|")) - 1 
                                   for inserts in insertions_flechi.split("|"):
                                          if inserts != [''] :
                                                 insert = inserts.split("--")
                                                 if insert != [''] :
                                                        nbInsertions += len(insert)
                                   col14_nbInsert[idMWE] = nbInsertions
                                   col15_nbDiscont[idMWE] = nbDiscont        
                                   # avec nombre exact d'insertions
                                   statsInsertions = sorted(dicostatsInsertions.items(), key=lambda t: t[0])                        
                                   col15_Insert[idMWE] = statsInsertions
                                   # Sans doublons d'insertions
                                   statsInsertionsSansDoublons = sorted(dicostatsInsertionsSansDoublons.items(), key=lambda t: t[0])      
                                   col15_InsertSansDoublons[idMWE] = statsInsertionsSansDoublons
                                   # Relations de dépendances sortantes :
                                   col20_RelDepSansDoublon[idMWE] = relDepMWE  
                                   # morpho : 
                                   col16_Morpho[idMWE] =  MorphoMWE   
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
                            #============================================================
                            #     distance syntaxique 
                            #============================================================                
                            col21_distSyn = {}
                            col22_typeDistSyn = {}
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
                            # calcul distSyn 
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
                            col21_distSyn = sorted(col21_distSyn.items(), key=lambda t: t[0])
                            col22_typeDistSyn = sorted(col22_typeDistSyn.items(), key=lambda t: t[0])
                            col23_distSyn2elts = sorted(col23_distSyn2elts.items(), key=lambda t: t[0])
                            col24_typeDistSyn2elts = sorted(col24_typeDistSyn2elts.items(), key=lambda t: t[0])
                            # =========================================
                            #	Fichier TSV <=> Tableur
                            # =========================================
                            for MWE in range(0,len(col3_fenetre)): 
                                   dicoPatronMWE = {}
                                   idMWE = col3_fenetre[MWE][0]                   
                                   col1 = str(idPhrase) + "_" + str(idMWE) #email-enronsent34_01-0024_MWE1  
                                   col2 = col2_phrase                      #A little birdie told me that ...                    
                                   col3 = col3_fenetre[MWE][1]             #A little birdie told                    
                                   col4 = col4_typeMWE[MWE][1]             #VID              
                                   col6 = col6_positionEltsMwe[MWE][1]     #1|2|3|4
                                   col7 = col7_EltLexFlex[MWE][1]          #A|little|birdie|told
                                   col8 = col8_EltLexLemme[MWE][1]         #a|little|birdie|tell
                                   col9 = col9_EltLexPOS[MWE][1]           #DET|ADJ|NOUN|VERB
                                   col18_ok = col16_Morpho[MWE][1]
                                   lemmesEltLex = col8.split("|")
                                   col5 = len(lemmesEltLex)                    
                                   #====== FR : Homogénéisation IRV avec pronoms lemmatisés différemment de SE ======
                                   if langue == "FR" :
                                          new_lemmesEltLex = []
                                          for l in range(0,len(lemmesEltLex)) :
                                                 if col4 == "IRV" and lemmesEltLex[l] in ["me","te","le","lui","nous","vous"]:
                                                        newl = "se"
                                                        new_lemmesEltLex.append(newl)                            
                                                 elif lemmesEltLex[l] in ["me","te","nous","vous"]:
                                                        newl = "se"
                                                        new_lemmesEltLex.append(newl)                            
                                                 else :
                                                        newl = lemmesEltLex[l].lower()
                                                        new_lemmesEltLex.append(newl)
                                          col10 = ";".join(sorted(new_lemmesEltLex))
                                   else :
                                          col8_lower = []
                                          col8_lower = map(lambda x:x.lower(),col8)
                                          col10 = ";".join(sorted(col8.split("|")))
                                   col11 = col11_insertFlex[MWE][1]        #
                                   col12 = col12_insertLemme[MWE][1]       #
                                   col13 = col13_insertPOS[MWE][1]         #DET--NOUN--ADP
                                   col14 = col14_nbInsert[MWE][1]          #3
                                   col15 = col15_nbDiscont[MWE][1]
                                   col16 = col15_Insert[MWE][1]            #[('ADJ', 0), ('ADP', 0),
                                   col17 = col15_InsertSansDoublons[MWE][1] #[('ADJ', 0), ('ADP', 0),   
                                   col21 = col21_distSyn[MWE][1]
                                   col22 = col22_typeDistSyn[MWE][1]
                                   col23 = col23_distSyn2elts[MWE][1]
                                   col24 = col24_typeDistSyn2elts[MWE][1]
                                   eltLexPOS = col9.split("|")
                                   dicoMorphoParElt = {}
                                   for POS in range(0,len(eltLexPOS)) :
                                          lemmeElt = col8.split("|")[POS]
                                          positionElt = col6.split("|")[POS]
                                          dicoMorphoParElt[str(eltLexPOS[POS]) + "_" + str(lemmeElt) + "_" + str(positionElt)] = col16_Morpho[MWE][1][POS]                    
                                   col18 = dicoMorphoParElt        #{'ADP': {'Gender': '_', 'Degree': '_'
                                   col19 = col20_RelDepSansDoublon[MWE][1]       
                                   f_Tableur.write(str(col1) + "\t" + str(col2) + "\t" + \
                                                   str(col3) + "\t" + str(col4) + "\t" + str(col5) + "\t" + \
                                                   str(col6) + "\t" + str(col10) + "\t" + str(col7) + "\t" + \
                                                   str(col8) + "\t" + str(col9) + "\t" + str(col11) + "\t" + \
                                                   str(col12) + "\t" + str(col13) + "\t" + str(col14) + "\t" + \
                                                   str(col15) + "\t" + str(col16) + "\t" + str(col17) + "\t" +\
                                                   str(col18_ok) + "\t" + str(col19) + "\t" + str(col21) + "\t" +\
                                                   str(col22) + "\t" + str(col23) + "\t" + str(col24) + "\t" +\
                                                   str(col25_RelDepEntrante) + "\n")


                                   # Synthese Patrons observés pour chaque NF :
                                   NF = col10 
                                   pattern = col9
                                   if NF not in NFallPatterns :
                                          NFallPatterns[NF] = {}
                                          NFallPatterns[NF][pattern] = 1
                                   else : 
                                          if pattern in NFallPatterns[NF] :
                                                 NFallPatterns[NF][pattern] += 1
                                          else :
                                                 NFallPatterns[NF][pattern] = 1 
       return NFallPatterns
       
