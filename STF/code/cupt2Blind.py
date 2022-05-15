#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os,sys
repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])

repParent = repParent + "/INPUT/DATA_LANG"
for langue in  ["BG","DE","EL","EN","ES","EU","FA","FR","HE","HI","HR","HU","IT","LT","PL","PT","RO","SL","TR"]: 

	fichierIN_Dev = str(repParent) + "/" + str(langue) + "/" + "dev.cupt"
	fichierOUT_Dev = str(repParent) + "/" +str(langue) + "/" + "new.dev.blind.cupt"
            
	with open(fichierIN_Dev,"r") as f_cupt:
		with open(fichierOUT_Dev,"a") as f_blind:
			lignes = f_cupt.readlines()		
			for ligne in lignes :				
				infos = ligne.split("\t") 
				if "#" in str(infos[0]) or infos == ['\n'] :
					f_blind.write(ligne)
				else :	
					newligne = ""
					for i in range(0,10) :			
						newligne += str(infos[i]) + "\t"
					newligne += "*\n"		
					f_blind.write(newligne)
				
	fichierIN_Test = str(repParent) + "/" +str(langue) + "/" + "test.blind.cupt"
	fichierOUT_Test = str(repParent) + "/" +str(langue) + "/" + "new.test.blind.cupt"
            
	with open(fichierIN_Test,"r") as f_cupt:
		with open(fichierOUT_Test,"a") as f_blind:
			lignes = f_cupt.readlines()		
			for ligne in lignes :				
				infos = ligne.split("\t") 
				if "#" in str(infos[0]) or infos == ['\n'] :
					f_blind.write(ligne)
				else :	
					newligne = ""
					for i in range(0,10) :			
						newligne += str(infos[i]) + "\t"
					newligne += "*\n"		
					f_blind.write(newligne)

