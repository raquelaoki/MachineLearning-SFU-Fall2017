setwd('C:\\Users\\raque\\Documents\\SFU\\Machine Learning')
require(ggplot2)
require(reshape)

#Accuracies
ac_train = matrix(c(0.678,0.977,0.949,0.691,0.971,0.926,0.715, 0.982,
                    0.975,0.689,0.983,0.930,0.605,0.903,0.874),ncol=3,byrow = T)
ac_test = matrix(c(0.413,0.928,0.873,0.527,0.886,0.852,0.801,0.934,
                   0.919,0.746,0.875,0.861,0.402,0.809,0.852),ncol=3,byrow = T)
method = c('DA - Hidden Layer','DA - Top100', 'DFS - Top100','RF - Top100',
           'Top 4 genes')

ac_train2 = data.frame(method,ac_train)
names(ac_train2) = c('Method','Overall Survival','ER','HER2')
ac_train2 = melt(ac_train2,id = 'Method')

ac_test2 = data.frame(method,ac_test)
names(ac_test2) = c('Method','Overall Survival','ER','HER2')
ac_test2 = melt(ac_test2,id = 'Method')

#
p <-ggplot(ac_train2, aes(variable, value, fill=Method))
p +geom_bar(stat = "identity",position="dodge",alpha=0.93)+theme_minimal()+
  scale_fill_manual(values = c('#f16c20','#1395ba','#0d4563'), name='Method') + 
  ylab('Accuracy - TRAINING')+xlab('Clinical Information')+
  theme(axis.text = element_text(size = 15), 
        axis.title = element_text(size = 15))+
  theme(legend.text = element_text(size = 15),
        legend.title = element_text(size = 15),legend.position = 'top')

ac_test2$value = ac_test2$value*100
ac_test2 = ac_test2[order(ac_test2$variable,ac_test2$Method),]
label = paste(ac_test2$value,'%',sep='')
p <-ggplot(ac_test2, aes(variable, value, fill=Method))
p +geom_bar(stat = "identity",position="dodge",alpha=0.93)+theme_minimal()+
  scale_fill_manual(values = c('#1395ba','#0d4563','#8bc240','#f16c20','#a12059'), name='Method') + 
  ylab('Accuracy(%) - Testing Set')+xlab('Clinical Information')+ylim(0,1.1*max(ac_test2$value))+
  theme(axis.text = element_text(size = 12), axis.title = element_text(size = 12))+
  theme(legend.text = element_text(size = 12),
        legend.title = element_text(size = 12),legend.position = 'top')+
  guides(fill=guide_legend(ncol=3)) +
  geom_text(data=ac_test2,aes(label = paste(value,'%',sep='')),
            position = position_dodge(width=.9),vjust=-.5)

#feature importance
da = read.csv('features_importance_DA_all.csv',sep=',')

dfs1 = read.csv('features_importance_DFS_all_os.csv',sep=',')
dfs2 = read.csv('features_importance_DFS_all_er.csv',sep=',')
dfs3 = read.csv('features_importance_DFS_all_her.csv',sep=',',header=T)

rf1 = read.csv('overall_survival_all.csv',sep=',')
rf2 = read.csv('ER_status_all.csv',sep=',')
rf3 = read.csv('HER2_status_all.csv',sep=',')

#organize
names(da) = c('genes','weight_da')

names(dfs1) = c('genes','weight_dfs')
names(dfs2) = c('genes','weight_dfs')
names(dfs3) = c('genes','weight_dfs')

names(rf1) = c('genes','weight_rf')
names(rf2) = c('genes','weight_rf')
names(rf3) = c('genes','weight_rf')

dfs1$genes = as.character(dfs1$genes)
dfs2$genes = as.character(dfs2$genes)

dfs1$genes[dfs1$genes=='SEPT3']='03-Sep'
dfs1$genes[dfs1$genes=='SEP15']='15-Sep'
dfs1$genes[dfs1$genes=='MARCH6']='06-Mar'
dfs2$genes[dfs2$genes=='SEPT3']='03-Sep'
dfs2$genes[dfs2$genes=='SEP15']='15-Sep'
dfs2$genes[dfs2$genes=='MARCH6']='06-Mar'


ov = merge(da,dfs1,all=T,by.x='genes',by.y='genes')
ov = merge(ov,rf1,all=T,by.x='genes',by.y='genes')

er  = merge(da,dfs2,all=T,by.x='genes',by.y='genes')
er = merge(er,rf2,all=T,by.x='genes',by.y='genes')

her = merge(da,dfs3,all=T,by.x='genes',by.y='genes')
her = merge(her,rf3,all=T,by.x='genes',by.y='genes')

matrix_cor <- function(bd){
  m = c(cor(bd[,2],bd[,3]),cor(bd[,2],bd[,4]),cor(bd[,3],bd[,4]))
  return (m)
}

ovc = matrix_cor(ov)
erc = matrix_cor(er)
herc = matrix_cor(her)

correlations = c(ovc,erc,herc)
comparison = rep(c('DA-DFS','DA-RF','RF-DFS'),3)
clinical = c(rep('OS',3),rep('ER',3),rep('HER',3))

bd = data.frame(clinical, comparison,correlations)
bd = bd[order(bd$clinical,bd$comparison),]
label = round(bd$correlations,2)

p <-ggplot(bd, aes(clinical, correlations, fill=comparison))
p +geom_bar(stat = "identity",position="dodge",alpha=0.93)+theme_minimal()+
  scale_fill_brewer(palette="Greens", name='Comparison') + 
  ylab('Correlation')+xlab('Clinical Information')+ylim(0,1.1*max(bd$correlations))+
  theme(axis.text = element_text(size = 15), axis.title = element_text(size = 15))+
  theme(legend.text = element_text(size = 15),legend.title = element_text(size = 15),legend.position = 'top')+
  geom_text(position = position_dodge(width=.9),vjust=-.5,
            label=label[c(3,2,1,6,5,4,9,8,7)])















