
library(tidyverse)
library(readxl)

## degreedata.xlsx

p2_degree = read_excel(path = "feature/p2-fro_degreedata.xlsx", col_names = TRUE)
p3_degree = read_excel(path = "feature/p3-fro_degreedata.xlsx", col_names = TRUE)
p4_degree = read_excel(path = "feature/p4-fro_degreedata.xlsx", col_names = TRUE)
p5_degree = read_excel(path = "feature/p5-fro_degreedata.xlsx", col_names = TRUE)
p6_degree = read_excel(path = "feature/p6-fro_degreedata.xlsx", col_names = TRUE)
p7_degree = read_excel(path = "feature/p7-fro_degreedata.xlsx", col_names = TRUE)

## alldata.xlsx

p2_all = read_excel(path = "feature/p2-fro_alldata.xlsx", col_names = TRUE)
p3_all = read_excel(path = "feature/p3-fro_alldata.xlsx", col_names = TRUE)
p4_all = read_excel(path = "feature/p4-fro_alldata.xlsx", col_names = TRUE)
p5_all = read_excel(path = "feature/p5-fro_alldata.xlsx", col_names = TRUE)
p6_all = read_excel(path = "feature/p6-fro_alldata.xlsx", col_names = TRUE)
p7_all = read_excel(path = "feature/p7-fro_alldata.xlsx", col_names = TRUE)


#Variable names
colnames(p2_all)



#T-test for "line" variable
for (i in 2:6) {
  for (j in (i+1):7) {
    assign(paste0("line.test.",i,".",j), t.test(get(paste0("p",i,"_all"))$line, 
                                                get(paste0("p",j,"_all"))$line))
  }
}


#T-test for "length" variable
for (i in 2:6) {
  for (j in (i+1):7) {
    assign(paste0("length.test.",i,".",j), t.test(get(paste0("p",i,"_all"))$length, 
                                                get(paste0("p",j,"_all"))$length))
  }
}

#T-test for "width" variable
for (i in 2:6) {
  for (j in (i+1):7) {
    assign(paste0("width.test.",i,".",j), t.test(get(paste0("p",i,"_all"))$width, 
                                                  get(paste0("p",j,"_all"))$width))
  }
}


#T-test for "width_var" variable
for (i in 2:6) {
  for (j in (i+1):7) {
    assign(paste0("width.var.test.",i,".",j), t.test(get(paste0("p",i,"_all"))$width_var, 
                                                  get(paste0("p",j,"_all"))$width_var))
  }
}

#T-test for "tortuosity" variable
for (i in 2:6) {
  for (j in (i+1):7) {
    assign(paste0("tortuosity.test.",i,".",j), t.test(get(paste0("p",i,"_all"))$tortuosity, 
                                                  get(paste0("p",j,"_all"))$tortuosity))
  }
}


#Record test results

line.result <- matrix(nrow = 6, ncol = 6)
length.result <- matrix(nrow = 6, ncol =6)
width.result <- matrix(nrow = 6, ncol = 6)
width.var.result <- matrix(nrow = 6, ncol = 6)
tortuosity.result <- matrix(nrow = 6, ncol = 6)


for (i in 2:6) {
  for (j in (i+1):7) {
    line.result[i-1,j-1] <- ifelse(get(paste0("line.test.",i,".",j))$p.value<=0.05,"*","-")
    length.result[i-1,j-1] <- ifelse(get(paste0("length.test.",i,".",j))$p.value<=0.05,"*","-")
    width.result[i-1,j-1] <- ifelse(get(paste0("width.test.",i,".",j))$p.value<=0.05,"*","-")
    width.var.result[i-1,j-1] <- ifelse(get(paste0("width.var.test.",i,".",j))$p.value<=0.05,"*","-")
    tortuosity.result[i-1,j-1] <- ifelse(get(paste0("tortuosity.test.",i,".",j))$p.value<=0.05,"*","-")
    }
}

#Clean-up test result summary tables
line.result[is.na(line.result)] <- " "
length.result[is.na(length.result)] <- " "
width.result[is.na(width.result)] <- " "
width.var.result[is.na(width.var.result)] <- " "
tortuosity.result[is.na(tortuosity.result)] <- " "

rownames(line.result) <- c("P2","P3","P4","P5","P6","P7")
colnames(line.result) <- c("P2","P3","P4","P5","P6","P7")
rownames(length.result) <- c("P2","P3","P4","P5","P6","P7")
colnames(length.result) <- c("P2","P3","P4","P5","P6","P7")
rownames(width.result) <- c("P2","P3","P4","P5","P6","P7")
colnames(width.result) <- c("P2","P3","P4","P5","P6","P7")
rownames(width.var.result) <- c("P2","P3","P4","P5","P6","P7")
colnames(width.var.result) <- c("P2","P3","P4","P5","P6","P7")
rownames(tortuosity.result) <- c("P2","P3","P4","P5","P6","P7")
colnames(tortuosity.result) <- c("P2","P3","P4","P5","P6","P7")

#Final Result Tables
knitr::kable(as.data.frame(line.result), align="c")
knitr::kable(as.data.frame(length.result), align="c")
knitr::kable(as.data.frame(width.result), align="c")
knitr::kable(as.data.frame(width.var.result), align="c")
knitr::kable(as.data.frame(tortuosity.result), align="c")

