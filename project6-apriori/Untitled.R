library(datasets)
library(arules)
library(arulesViz)

data(Groceries)

stuff <- apriori(Groceries, parameter = list(supp = 0.006, conf = 0.01))

inspect(head(sort(stuff, by = "support"), 50))
plot(stuff, measure = c("support", "confidence"), shadding = "lift")