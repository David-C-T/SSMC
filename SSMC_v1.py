#DavidCT
import json,sys,logging
from owlready2 import *
import owlready2 
from math import log,inf
from statistics import mean
from itertools import product
import pandas as p
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
plt.rcParams.update({'figure.max_open_warning': 0})
sns.set()
logging.basicConfig(format='%(message)s',level = logging.INFO)

Settings_Path = sys.argv[1]

#===============================================Ontology Handling and basic Functions=======================================================

def Get_Nodes(Ontology):
        return list(Ontology.classes())

def Get_Ontology_Roots(Ontology,Singletons=True):
    Roots=list(Thing.subclasses())
    if Singletons==False:
        Actual_Roots=[]
        for x in Roots:
            b=list(x.subclasses())
            if len(b)> 0:
                Actual_Roots.append(x)
                Roots=Actual_Roots
    return Roots

def Get_Roots(Singletons=False,Obsoletes=False):
    #Removes Singletons and Obsoletes, also classes without a label
    Roots=list(Thing.subclasses())#Needs a more complete for MESH
    Roots=list([x for x in Roots if len(x.label)> 0])#REMOVE CLASSES WITHOUT LABELS
    message="Found "+str(len(Roots))+" roots"
    print(message)
    Actual_Roots=set(Roots)#Because python doennt know how to copy lists independently
    if Obsoletes==False:#Removes and counts obsoletes
        for x in Roots:
            if "obsolete" in x.label[0].lower():
                Actual_Roots.remove(x)
        message="   > Removed "+str(len(Roots)-len(Actual_Roots))+" obsolete classes."
        print(message)   
        Roots=list(Actual_Roots)#Get the list again for other filters
    if Singletons==False:#Remove and counts singletons
        for x in Roots:
            b=list(x.subclasses())
            if len(b)==0:
                Actual_Roots.remove(x)
        message="   > Removed "+str(len(Roots)-len(Actual_Roots))+" singletons."
        print(message)        
    return list(Actual_Roots)

def Find_Leaves(Concepts):
    #Find which concepts are leaves
    return [x for x in Concepts if len(list(x.subclasses()))==0]

def Get_Subclasses(Concept):
    return list(Concept.subclasses())

def Get_Ancestors(Concept):
    return Concept.ancestors()

def Get_Descendants(Concept):
    #Returns a list of concepts desceding from Concept including the Concept itself
    return list(Concept.descendants())

def Get_Parents(Concept):
    return Concept.is_a

def Is_Singleton(Concept):
    if len(Get_Subclasses(Concept))+len(Get_Parents)-1==0:
        return True
    else:
        return False

def Normalize(Max,Min,value):
    return (value-Min)/(Max-Min)

#===========================================Ontology Editing and Reasoning=======================================================

def subclass_inference(Concepts,namespace):
    #Converts euivalent classes into parent classes
    #There must be a namespace for the ontology
    #Cocnepts must be CONCEPTS, not IRI
    for node in Concepts:
        Equivalent=node.equivalent_to#Get quivalent classes
        if len(Equivalent)> 0:
            for x in str(Equivalent[0]).split(" & "):
                y=x.split(".")
                if len(y)==2:
                    ref=namespace+y[1]
                    z=IRIS[ref]
                    if Equivalent not in node.is_a:
                        node.is_a.append(z)

def ReRoot_Ontologies(Roots,Ontologies):
    #Make Root with iri namespace.Virtual_Root, attach to one of the ontologies
    class Virtual_Root(Thing):
        namespace = Ontologies[0]
        label="Virtual Root"
    for root in Roots:
        root.is_a.append(Virtual_Root)
    IRI=Ontologies[0].base_iri+"Virtual_Root"
    return IRI


#=====================================================Compute IC=================================================================

def IC_Sanchez2011(IC_Map,Concepts):
    #Max Leaves
    maxleaves=len(Find_Leaves(Concepts))
    for Concept in list(IC_Map):
        #Leaves
        leaves_c=[x for x in Concept.descendants() if len(list(x.subclasses()))==0].__len__()
        if leaves_c==0:
            leaves_c=1
        #Subsumers
        subsumers_c=len(Concept.ancestors())-1-1#Exclude Thing and itself
        #IC_Sanchez
        IC=-log(((abs(leaves_c))/(abs(subsumers_c)+1))/(maxleaves+1))
        IC_Map[Concept] = IC
    return IC_Map
    
def IC_Seco2004(IC_Map,Concepts):
    for Concept in list(IC_Map):
        #Get Hyponyms of a concept
        Hyponyms=len(Concept.descendants())-1#Remove Thing
        #IC_Seco
        IC=1-(log(Hyponyms+1)/log(len(Concepts)))
        IC_Map[Concept]= IC
    return IC_Map


#=====================================================PAIRWISE SS=================================================================

def Pairwise_Resnik1995(IC_Map,c1,c2,**args):
    #Get Common Ancestors
    CA=list(c1.ancestors().intersection(c2.ancestors()))
    CA.remove(owl.Thing)
    #Get and return MICA IC
    try:
        return max([IC_Map[x] for x in CA])
    except ValueError:#No common ancestor found
        return 0.0



#=====================================================GROUPWISE SS=================================================================


def Groupwise_SimGIC(IC_Map,ob1,ob2,**args):
    #Intersecting Concepts
    Intersect=list(set(ob1).intersection(set(ob2)))
    Intersect_sum=sum([IC_Map[x] for x in Intersect])  
    #All Concepts
    All=list(set().union(ob1,ob2))
    All_sum=sum([IC_Map[x] for x in All])
    return Intersect_sum/All_sum

def Groupwise_AVG(IC_Map,ob1,ob2,Pairwise,**args):
    #Get pairwise similarities from all combinations of concepts
    Pairwise_Sim=[eval("Pairwise_"+ Pairwise)(IC_Map=IC_Map,c1=x[0],c2=x[1]) for x in list(product(ob1,ob2))]#includes repetitions
    return mean(Pairwise_Sim)

def Groupwise_MAX(IC_Map,ob1,ob2,Pairwise,**args):
    #Get pairwise similarities from all combinations of concepts
    Pairwise_Sim=[eval("Pairwise_"+ Pairwise)(IC_Map=IC_Map,c1=x[0],c2=x[1]) for x in list(product(ob1,ob2))]#includes repetitions
    return max(Pairwise_Sim)

def Groupwise_BMA(IC_Map,ob1,ob2,Pairwise,**args):
    SIM_1=mean([Groupwise_MAX(IC_Map,[concept],ob2,Pairwise) for concept in ob1])
    SIM_2=mean([Groupwise_MAX(IC_Map,[concept],ob1,Pairwise) for concept in ob2])#Reciprocal
    return (SIM_1+SIM_2)/2


#=====================================================Utility and CLustering==================================================================

def Load_Objects(Path):
    try:
        with open(Settings["Object_Data"]) as data:
            Object_Data={}
            line=data.readline()
            while line:
                Content=line.rstrip().split("\t")
                try:
                    Annotations=[(IRIS[x]) for x in Content[1].split(';')]   
                except NameError as e:
                    message="The object: "+ str(Content[0])+" has no annotations. This file must contain only annotated objects."
                    print(message)
                    break    
                Object_Data[Content[0]]=Annotations
                line=data.readline()
            data.close()
    except KeyError as e:#No object data Pairwise comparison
        pass 
    except FileNotFoundError as e:
        print('File was not found')
    if None in Object_Data.values():
        print("Problem")
    else:
        return Object_Data   

def Get_SimMatrix(SimScores,Target_Score): 
    
    Pivot=SimScores.pivot(index='Object_A', columns='Object_B', values=Target_Score)
    index = Pivot.index.union(Pivot.columns)
    Matrix = Pivot.reindex(index=index, columns=index)

    Matrix=np.array(Matrix)
    Matrix=np.tril(Matrix)+np.triu(Matrix.T)
    np.fill_diagonal(Matrix,1)

    return Matrix

def Get_KMeans(Matrix,K,Path):
    #Only works for a complete matrix i.e., every object has been compared against every object
    NA=len(np.argwhere(np.isnan(Matrix)))
    if NA>0:
        logging.info("WARNING: Similarity Matrix has "+str(NA)+" NaN values")
        Matrix[np.isnan(Matrix)] = 0      
    Kmeans = KMeans(n_clusters=K)
    R=Kmeans.fit_predict(Matrix)
    Labels = Kmeans.labels_        

    #Cluster
    fig, ax1  = plt.subplots()
    ax1.scatter(Matrix[:, 0], Matrix[:, 1], c=R, s=25, cmap='Accent')
    centers = Kmeans.cluster_centers_
    ax1.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.suptitle(("Kmeans clustering, k = %d" % K),fontsize=14)  
    plt.savefig(Path, bbox_inches='tight')  
    return Labels
    

#=====================================================Actual Process==================================================================

def Sim_and_Cluster(Settings):
    #=====================================================Load Modules and Options==================================================================

    SSM_IDs=Settings["Similarity Settings"]
    Querries=Settings["Querries"]
    logging.info("   > Found "+str(len(SSM_IDs))+" SSMs")
    logging.info("   > Found "+str(len(Querries))+" querries")
    Normalize_values=True
    Reasoning=False

    #============================================================Ontology Loader==========================================================

    logging.info("Loading Ontologies")
    Ontology_Settings=Settings["Ontologies"]
    Ontologies=list(Ontology_Settings.keys())
    Ontology_Load={}#Loading in a dict to not lose references for custom namespaces, then converto actual list
    for onto in Ontologies:
        Ontology_Load[onto]=get_ontology(Ontology_Settings[onto]["Path"]).load()
        logging.info("   > loaded "+onto+ " as "+str(Ontology_Load[onto]))
        if "Prefix" in Ontology_Settings[onto]:
            Ontology_Load[onto].get_namespace(Ontology_Settings[onto]['Prefix'])
        else:
            Ontology_Load[onto].get_namespace(Ontology_Load[onto].base_iri)
    
        
    Onto=list(Ontology_Load.values())#A list of loaded Ontologies
    Onto_Namespace=[x.base_iri for x in Onto]#A list of ontology namespaces
    if Reasoning:
        for onto in Onto:
            logging.info("   > Applying Hermit reasoner to "+str(onto.name)+" ontology")
            with onto: sync_reasoner()
    else:
        for onto in Onto:
            if onto.name!='IOBC(edit).xrdf':
                logging.info("   > Infering Subclassess in "+ str(onto.name))
                subclass_inference(Get_Nodes(onto),Onto_Namespace[Onto.index(onto)])#Doenst work for IOBC

    #ReRooting
    Roots=Get_Roots()
    logging.info("   > Retrieved "+str(len(Roots))+" roots\n")
    if len(Onto)> 1:
        logging.info("Rerooting "+str(len(Onto))+" ontologies")
        Virtual_IRI=ReRoot_Ontologies(Roots,Onto)
        logging.info("   > Rerooted to a Virtual Root at: \""+Virtual_IRI+"\"\n")

    #Get Concepts
    Concepts=[Get_Nodes(x) for x in Onto]#error at IOBC (Cycles cause error, use edited version, test further)
    Concepts=[val for sublist in Concepts for val in sublist]
    logging.info("Found "+str(len(Concepts))+" concepts\n")#Includes Root

    #============================================================Compute IC==========================================================

    #Compute IC for all concepts
    logging.info("Computing IC:")
    IC_Map=dict.fromkeys(Concepts,{})

    IDs=["".join(Querries[x]["SSM_ID"]).split(",") for x in Querries.keys()]
    IDs=[x for l in IDs for x in l]
    IC_List=list(set([SSM_IDs[x]["IC"] for x in IDs]))


    Computed_ICs={}
    for IC in IC_List:
        Map=eval("IC_"+IC)(IC_Map,Concepts)
        if Normalize_values:
            Max=max(Map.values())
            Min=min(Map.values())
            if Max>1 or Min<0:
                Map = {x: Normalize(Max,Min,Map[x]) for x in Map}   
        logging.info("   > "+IC)
        Computed_ICs[IC]=Map#Dict of IC dicts   
    logging.info("\n")

    #============================================================Process Requests============================================================

    #Load Object Data
    Object_Data=Load_Objects(Settings["Object_Data"])
    logging.info("Loaded Data for "+str(len(Object_Data))+" objects.")

    #Process Querries
    for x in Querries:
        logging.info("Starting on querry "+x)
        IDs=Querries[x]["SSM_ID"].split(",")
        SSMs = [Settings["Similarity Settings"][ID] for ID in IDs]
        with open(Querries[x]["Path"]) as querry:
            a=[[num]+line.strip("\n").split() for num, line in enumerate(querry, 0)]
        b={row[0]: row[1:] for row in a}#A dictionary of all combinations of objects or concepts (easier to convert into dataframe)
        Header=["Object_A","Object_B"]#Dataframe header
        logging.info("   > Loaded query file "+Querries[x]["Path"])

        for ID in SSMs:#This adds all possible arguments for a SSM, making a universal call possible later on (spares if conditions)
            for arg in ['IC','Pairwise']:#Except groupwise since that is never an argument in SSM fucntions
                if arg not in ID:
                    ID[arg]=None#it will be ignored by the intended SSM function (**args)
        
            if "Groupwise" in ID:
                logging.info("   > Calculating Groupwise Scores using: "+ID["Groupwise"])
                Header.append(ID["Groupwise"]+"_"+ID["IC"])
                Err=[]
                for k,entry in b.items():
                    if k%(len(b.keys())/20)==0:
                        logging.info("      "+str(k)+"/"+str(len(b.keys())))
                    if entry[0] not in Object_Data or entry[1] not in Object_Data:
                        if entry[0] not in Object_Data:
                            Err.append(entry[0])
                        else:
                            Err.append(entry[1])
                        b[k].append(0)
                        continue
                    Score=eval("Groupwise_"+ID["Groupwise"])(ob1=Object_Data[entry[0]],ob2=Object_Data[entry[1]],Pairwise=ID["Pairwise"],IC_Map=eval("Computed_ICs[\""+str(ID["IC"])+"\"]"))#Universal call for groupwise
                    b[k].append(Score)
            else:
                logging.info("   > Calculating Pairwise Scores using: "+ID["Pairwise"])
                Header.append("Pairwise_"+ID["Pairwise"])
                Err=[]
                for k,entry in b.items():
                    Score=eval("Pairwise_"+ID["Pairwise"])(c1=IRIS[entry[0]],c2=IRIS[entry[1]],IC_Map=eval("Computed_ICs[\""+str(ID["IC"])+"\"]"))#Universal call for pairwise
                    b[k].append(Score)  
            if len(Err)>0:
                logging.info("WARNING: No object data for "+str(len(Err))+" comparisons")
        N=np.array(list(b.values()))
        SimScores=p.DataFrame(N,index=np.arange(1, N.shape[0]+1),columns=np.arange(1, N.shape[1]+1))
        SimScores.columns=Header
        for y in Header[2:]:
            SimScores[y]=SimScores[y].astype(float)

        if "Results" in Querries[x]:
            SimScores.to_csv(Querries[x]["Results"])
            logging.info("   > Saved Scores to "+Querries[x]["Results"])
        #Clustering
        if "Clusters" in Querries[x]:
            for ID in Querries[x]["Clusters"]["Score"].split(","): 
                #Get target header from the index in IDs, retrieve first value and make the name            
                Target_Score="_".join([x[1] for x in list(SSMs[IDs.index(ID)].items()) if x[1]!=None])     
                logging.info("   > Generating Similarity Matrix using: "+Target_Score)
                Matrix=Get_SimMatrix(SimScores,Target_Score)#Get Matrix from that score
                logging.info("   > Clustering")
                for k in Querries[x]["Clusters"]["K"].split(","):
                    Cluster_Path=Querries[x]["Clusters"]["Cluster Path"]+"\\"+Target_Score+"_k"+k+".png"
                    Labels=Get_KMeans(Matrix,int(k),Cluster_Path)   
    logging.info("\nFinished")



if __name__ == '__main__':
    
    with open(Settings_Path) as data:
        Settings=json.load(data)
        logging.info("\nGathering Data from directory: \""+Settings_Path+"\"")
    Sim_and_Cluster(Settings)
