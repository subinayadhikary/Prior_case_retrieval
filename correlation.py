from scipy import stats
from scipy.stats import spearmanr
import math
import rbo
pair_id=[]
gt_value=[]
sim_value=[]
pair_sim={}
pair_sim_train={}
doc1=[]
doc2=[]
with open("/home/subinay/Documents/data/pairwise_similarity/Fourth_50/tag_similarity_F_50.txt","r") as f1:
    for line in f1:
        line=line.strip()
        line1=line.split()
        str=line1[0]+line1[1]
        if(float(line1[2])!=0):
            pair_id.append(str)
            doc1.append(str)
            pair_sim[str]=float(line1[2])
            gt_value.append(float(line1[2]))
score_doc1=dict(sorted(pair_sim.items(), key=lambda item: item[1],reverse=True))
gt_doc= list(score_doc1.keys())
with open("/home/subinay/Documents/data/pairwise_similarity/supervised_non_prarmetric/sim_score.txt","r") as f2:
    for line in f2:
        line=line.strip() # remove null character
        line1=line.split() #divide row columnwise
        str=line1[0]+line1[1] #store pair id
        if str in pair_id: # to check pair in ground truth file
            sim_value.append(float(line1[2]))
            doc2.append(str)
            pair_sim_train[str]=float(line1[2])
score_doc2=dict(sorted(pair_sim_train.items(), key=lambda item: item[1],reverse=True))
train_doc= list(score_doc2.keys())
list1=[]
list2=[]
print(len(doc1))
print(len(doc2))

for i in range(len(doc1)):
    index1=gt_doc.index(doc1[i])
    list1.append(index1)
    index2=train_doc.index(doc2[i])
    list2.append(index2)

res = stats.kendalltau(list1, list2)
res2=stats.pearsonr(gt_value,sim_value)
corr, p_value = spearmanr(list1, list2)

#plt.scatter(list1,list2)
#plt.show()
# # Print the correlation coefficient and p-value
print("Kendall",res)
print("Spearman's correlation coefficient:", corr)
print("Pearson",res2)
print("Rank_biased_overlap",rbo.RankingSimilarity(list1, list2).rbo())
