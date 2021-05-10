
## degreedata.xlsx

library(tidyverse)
library(readxl)
p2_degree = read_excel(path = "feature/p2-fro_degreedata.xlsx", col_names = TRUE)
p3_degree = read_excel(path = "feature/p3-fro_degreedata.xlsx", col_names = TRUE)
p4_degree = read_excel(path = "feature/p4-fro_degreedata.xlsx", col_names = TRUE)
p5_degree = read_excel(path = "feature/p5-fro_degreedata.xlsx", col_names = TRUE)
p6_degree = read_excel(path = "feature/p6-fro_degreedata.xlsx", col_names = TRUE)
p7_degree = read_excel(path = "feature/p7-fro_degreedata.xlsx", col_names = TRUE)

p2_all = read_excel(path = "feature/p2-fro_alldata.xlsx", col_names = TRUE)
p3_all = read_excel(path = "feature/p3-fro_alldata.xlsx", col_names = TRUE)
p4_all = read_excel(path = "feature/p4-fro_alldata.xlsx", col_names = TRUE)
p5_all = read_excel(path = "feature/p5-fro_alldata.xlsx", col_names = TRUE)
p6_all = read_excel(path = "feature/p6-fro_alldata.xlsx", col_names = TRUE)
p7_all = read_excel(path = "feature/p7-fro_alldata.xlsx", col_names = TRUE)


# Count of branchpoints
nrow(p2_degree)
nrow(p3_degree)
nrow(p4_degree)
nrow(p5_degree)
nrow(p6_degree)
nrow(p7_degree)


#Define Average Neighbor Degree column
p2_degree$avgneighbor <- 0
p3_degree$avgneighbor <- 0
p4_degree$avgneighbor <- 0
p5_degree$avgneighbor <- 0
p6_degree$avgneighbor <- 0
p7_degree$avgneighbor <- 0


for (i in 1:nrow(p2_degree)) {
  #List of neighbors
  neighbor.set <- p2_all[p2_all$node1 == p2_degree$nodes[i],"node2"]
  
  #Number of neighbors
  if (dim(neighbor.set)[1]==0) {
    p2_degree$avgneighbor[i] <- 0
  }
  else {
    num.neighbors <- dim(neighbor.set)[1]
    neighbor.degree <- numeric(num.neighbors)
    
    for (j in 1:num.neighbors) {
      #Degree of first neighbor
      neighbor.degree[j] <- p2_degree[p2_degree$nodes==neighbor.set$node2[j],"degree"]
    }
    
    p2_degree$avgneighbor[i] <-  mean(unlist(neighbor.degree))
  }
}

for (i in 1:nrow(p3_degree)) {
  #List of neighbors
  neighbor.set <- p3_all[p3_all$node1 == p3_degree$nodes[i],"node2"]
  
  #Number of neighbors
  if (dim(neighbor.set)[1]==0) {
    p3_degree$avgneighbor[i] <- 0
  }
  else {
    num.neighbors <- dim(neighbor.set)[1]
    neighbor.degree <- numeric(num.neighbors)
    
    for (j in 1:num.neighbors) {
      #Degree of first neighbor
      neighbor.degree[j] <- p3_degree[p3_degree$nodes==neighbor.set$node2[j],"degree"]
    }
    
    p3_degree$avgneighbor[i] <-  mean(unlist(neighbor.degree))
  }
}

for (i in 1:nrow(p4_degree)) {
  #List of neighbors
  neighbor.set <- p4_all[p4_all$node1 == p4_degree$nodes[i],"node2"]
  
  #Number of neighbors
  if (dim(neighbor.set)[1]==0) {
    p4_degree$avgneighbor[i] <- 0
  }
  else {
    num.neighbors <- dim(neighbor.set)[1]
    neighbor.degree <- numeric(num.neighbors)
    
    for (j in 1:num.neighbors) {
      #Degree of first neighbor
      neighbor.degree[j] <- p4_degree[p4_degree$nodes==neighbor.set$node2[j],"degree"]
    }
    
    p4_degree$avgneighbor[i] <-  mean(unlist(neighbor.degree))
  }
}

for (i in 1:nrow(p5_degree)) {
  #List of neighbors
  neighbor.set <- p5_all[p5_all$node1 == p5_degree$nodes[i],"node2"]
  
  #Number of neighbors
  if (dim(neighbor.set)[1]==0) {
    p5_degree$avgneighbor[i] <- 0
  }
  else {
    num.neighbors <- dim(neighbor.set)[1]
    neighbor.degree <- numeric(num.neighbors)
    
    for (j in 1:num.neighbors) {
      #Degree of first neighbor
      neighbor.degree[j] <- p5_degree[p5_degree$nodes==neighbor.set$node2[j],"degree"]
    }
    
    p5_degree$avgneighbor[i] <-  mean(unlist(neighbor.degree))
  }
}


for (i in 1:nrow(p6_degree)) {
  #List of neighbors
  neighbor.set <- p6_all[p6_all$node1 == p6_degree$nodes[i],"node2"]
  
  #Number of neighbors
  if (dim(neighbor.set)[1]==0) {
    p6_degree$avgneighbor[i] <- 0
  }
  else {
    num.neighbors <- dim(neighbor.set)[1]
    neighbor.degree <- numeric(num.neighbors)
    
    for (j in 1:num.neighbors) {
      #Degree of first neighbor
      neighbor.degree[j] <- p6_degree[p6_degree$nodes==neighbor.set$node2[j],"degree"]
    }
    
    p6_degree$avgneighbor[i] <-  mean(unlist(neighbor.degree))
  }
}


for (i in 1:nrow(p7_degree)) {
  #List of neighbors
  neighbor.set <- p7_all[p7_all$node1 == p7_degree$nodes[i],"node2"]
  
  #Number of neighbors
  if (dim(neighbor.set)[1]==0) {
    p7_degree$avgneighbor[i] <- 0
  }
  else {
    num.neighbors <- dim(neighbor.set)[1]
    neighbor.degree <- numeric(num.neighbors)
    
    for (j in 1:num.neighbors) {
      #Degree of first neighbor
      neighbor.degree[j] <- p7_degree[p7_degree$nodes==neighbor.set$node2[j],"degree"]
    }
    
    p7_degree$avgneighbor[i] <-  mean(unlist(neighbor.degree))
  }
}

#Plots
dev.off()
par(mfrow=c(2,3))
p2<-plot(p2_degree$degree, p2_degree$avgneighbor,
     col="gray", pch=21, bg="red", xlab=c("Node Degree"),
     xlim = c(1,10), ylim = c(1,10),
     ylab=c("Average Neighbor Degree"), main = "P2")

p3<-plot(p3_degree$degree, p3_degree$avgneighbor,
     col="gray", pch=21, bg="orange", xlab=c("Node Degree"),
     xlim = c(1,10), ylim = c(1,10),
     ylab=c("Average Neighbor Degree"), main = "P3")

p4<-plot(p4_degree$degree, p4_degree$avgneighbor,
     col="gray", pch=21, bg="magenta", xlab=c("Node Degree"),
     xlim = c(1,10), ylim = c(1,10),
     ylab=c("Average Neighbor Degree"), main = "P4")

p5<-plot(p5_degree$degree, p5_degree$avgneighbor,
     col="gray", pch=21, bg="green", xlab=c("Node Degree"),
     xlim = c(1,10), ylim = c(1,10),
     ylab=c("Average Neighbor Degree"), main = "P5")

p6<-plot(p6_degree$degree, p6_degree$avgneighbor,
     col="gray", pch=21, bg="blue", xlab=c("Node Degree"),
     xlim = c(1,10), ylim = c(1,10),
     ylab=c("Average Neighbor Degree"), main = "P6")

p7<-plot(p7_degree$degree, p7_degree$avgneighbor,
     col="gray", pch=21, bg="black", xlab=c("Node Degree"),
     xlim = c(1,10), ylim = c(1,10),
     ylab=c("Average Neighbor Degree"), main = "P7")

