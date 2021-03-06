library(e1071)

# the data set
data_set = read.csv("PycharmProjects/project5-svm/letters.csv", head=TRUE, sep=",")

all_rows = 1:nrow(data_set)
test_rows = sample(all_rows, trunc(length(all_rows)) * 0.3)

data_test = data_set[test_rows,]
data_train = data_set[-test_rows,]

best_accuracy <- 0
best_cost <- 0
best_gamma <- 0

run_svm <- function(g, c) {
  
  model = svm(letter~., data = data_train, kernel="radial", gamma = g, cost = c)
  
  prediction = predict(model, data_test[,-1])
  
  #confusion_matrix = table(pred = prediction, true = data_test$letter)
  
  agreement = prediction == data_test$letter
  accuracy = prop.table(table(agreement))
  
  #print(confusion_matrix)
  #print(accuracy)
  
  if (accuracy[["TRUE"]][1] > best_accuracy) {
    #best_accuracy <<- accuracy[["TRUE"]][1]
    assign("best_accuracy", accuracy[["TRUE"]][1], envir = .GlobalEnv)
    assign("best_gamma", g, envir = .GlobalEnv)
    assign("best_cost", c, envir = .GlobalEnv)
    
    print(paste("accuracy:", best_accuracy, "gamma:", best_gamma, "cost:", best_cost))
  }
  return (accuracy[["TRUE"]][1])
  
}

run <- function() {
  all_accuracy <- NULL
  for (g in seq(0.08, 0.080, 0.001)) {
    for (c in 20:30) {
      all_accuracy <- append(all_accuracy, run_svm(g, c))
    }
  }
  print("Best Reults are:")
  print(paste("Best Accuracy:", best_accuracy))
  print(paste("Best Gama:", best_gamma))
  print(paste("best Cost:", best_cost))
  
}

#run()