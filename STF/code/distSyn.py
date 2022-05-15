#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import dirname	# remonter arborescence fichiers
import os.path
repParent = "/".join(os.path.abspath(os.path.dirname(sys.argv[0])).split("/")[:-1])

#---------------- Distance syntaxique entre 2 elts --------------------
def parent(positionElt,phrase) :
       parentElt = -1          #init  
       lignes = phrase.split("\n")
       for ligne in lignes :        
              infos = ligne.split("\t")
              #"-" dans amalgames
              if len(infos) == 11 and "-" not in infos[0] :
                     if int(infos[0]) == int(positionElt):
                            parentElt = infos[6] 
                            if parentElt == "_" or parentElt == "-" :
                                   parentElt = -1 
       return parentElt

def grandParent(positionElt,phrase) :
       grandparentElt = -1     #init  
       lignes = phrase.split("\n")
       for ligne in lignes :        
              infos = ligne.split("\t")
              #"-" dans amalgames
              if len(infos) == 11 and "-" not in infos[0]  :
                     if int(infos[0]) == int(positionElt) :
                            if isinstance(infos[6], int) :
                                   grandparentElt = parent(infos[6],phrase)
                                   if grandparentElt == "_" or grandparentElt == "-":
                                          grandparentElt = -1
       return grandparentElt

def distanceSyntaxique(elt1,elt2,phrase) :
       typeDistSyn = ""   #init  
       distSyn = -1        #init  
       if  int(elt1) == int(parent(elt2,phrase)) or int(elt2) == int(parent(elt1,phrase))  :
              distSyn = 0
              #print("cas1")
              typeDistSyn = "direct"  
              return distSyn,typeDistSyn
              
       elif (int(parent(elt1,phrase)) == int(parent(elt2,phrase)) and int(parent(elt2,phrase)) != -1 ) or ( int(parent(elt2,phrase)) == int(parent(elt1,phrase))  and int(parent(elt2,phrase)) != -1 ):
              distSyn = 1
              #print("cas2")
              typeDistSyn = "parallel"  
              return distSyn,typeDistSyn
       elif int(grandParent(elt1,phrase)) == int(elt2) or int(grandParent(elt2,phrase)) == int(elt1) :
              distSyn = 1
              #print("cas3")
              typeDistSyn = "serie"
              return distSyn,typeDistSyn
       elif (int(grandParent(elt1,phrase)) == int(parent(elt2,phrase)) and int(grandParent(elt1,phrase)) != -1 )  or  (int(grandParent(elt2,phrase)) == int(parent(elt1,phrase)) and int(parent(elt1,phrase)) != -1 )  :
              distSyn = 2
              typeDistSyn = "parallel" 
              #print("cas4")
              return distSyn,typeDistSyn
       elif int(grandParent(parent(elt1,phrase),phrase)) == int(elt2) or  int(grandParent(parent(elt2,phrase),phrase)) == int(elt1) :
              distSyn = 2
              #print("cas5")
              typeDistSyn = "serie" 
              return distSyn,typeDistSyn

       return distSyn,typeDistSyn       
       


#-------------------------------------
