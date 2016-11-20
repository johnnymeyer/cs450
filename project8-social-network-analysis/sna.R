library(igraph)
library(sna)

# read the file
mommy_twitter <- read.csv("~/PycharmProjects/project8-social-network-analysis/Mommy_twitterMentions.csv", header = FALSE)
mommy_blog <- read.csv("~/PycharmProjects/project8-social-network-analysis/Mommy_blogLinks.csv", header = FALSE)
mommy_tiny <- read.csv("~/PycharmProjects/project8-social-network-analysis/Mommy_blogLinks_tiny.csv", header = FALSE)

#my_graph <- graph.data.frame(mommy_twitter)
#my_graph <- graph.data.frame(mommy_tiny, directed = TRUE)
my_graph <- graph.data.frame(mommy_blog)

# get the names of the nodes
nodes <- get.data.frame(my_graph, what="vertices")

#layout1 = layout.fruchterman.reingold(my_graph)
#layout2 = layout.auto(my_graph)
#plot(my_graph, layout=layout2)

graph.density(my_graph)

dc = centralization.degree(my_graph)
cc = centralization.closeness(my_graph)
bc = centralization.betweenness(my_graph)

index_dc <- which(dc$res == max(dc$res))
index_cc <- which(cc$res == max(cc$res))
index_bc <- which(bc$res == max(bc$res))

cat("Degree Centrality\n")
cat("Node:", nodes$name[index_dc], "Value:", dc$res[index_dc], "\n")

cat("Closeness Centrality\n")
cat("Node:", nodes$name[index_cc], "Value:", cc$res[index_cc], "\n")

cat("Betweenness Centrality\n")
cat("Node:", nodes$name[index_bc], "Value:", bc$res[index_bc])