library(datasets)
library(cluster)
data_original = state.x77

data_temp <- subset(data_original, select = -Income)
# data_frost <- subset(data_original, select = Frost)

data <- scale(data_temp)

# distance <- dist(as.matrix(data))
#clust <- hclust(distance)
#plot(clust)

#table = NULL
#for (i in 1:10) {
#  the_cluster <- kmeans(data, i)
#  table[i] <- the_cluster$tot.withinss
#}
#plot(table, type = 'o')

the_cluster <- kmeans(data, 4)
clusplot(data, the_cluster$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
