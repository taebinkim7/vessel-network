library(tidyverse)
library(readxl)

setwd("D:/STOR893-Zhang/vessel-network/feature_extraction/feature")

all.dat.p2 = read_excel("p2-fro_alldata.xlsx")
all.dat.p3 = read_excel("p3-fro_alldata.xlsx")
all.dat.p4 = read_excel("p4-fro_alldata.xlsx")
all.dat.p5 = read_excel("p5-fro_alldata.xlsx")
all.dat.p6 = read_excel("p6-fro_alldata.xlsx")
all.dat.p7 = read_excel("p7-fro_alldata.xlsx")
all.dat.p7.x0 = read_excel("p7-x 0_alldata.xlsx")
all.dat.N.129 = read_excel("N_129__alldata.xlsx")


p1to7_all <- rbind(all.dat.p2, 
                   all.dat.p3, 
                   all.dat.p4, 
                   all.dat.p5, 
                   all.dat.p6, 
                   all.dat.p7)

p1to7_all$Mice_age <- c(rep("P2", nrow(all.dat.p2)), 
                     rep("P3", nrow(all.dat.p3)), 
                     rep("P4", nrow(all.dat.p4)), 
                     rep("P5", nrow(all.dat.p5)), 
                     rep("P6", nrow(all.dat.p6)), 
                     rep("P7", nrow(all.dat.p7)))

colnames(p1to7_all) <- c("nodespair", "node1", "node2", "Vessel line", 
                         "Vessel length", "Vessel width", "wide_var", "Vessel tortuosity", 
                         "curve", "Mice_age")

p1to7_all_long <- gather(p1to7_all, 
                         key = "feature", 
                         value = "value",
                         "Vessel line", 
                         "Vessel length", 
                         "Vessel width", 
                         "Vessel tortuosity")


G.all = ggplot(p1to7_all_long, mapping = aes(x=value, color=Mice_age)) + 
  geom_line(stat = "Density",cex=0.8) +
  scale_x_continuous(name="Log10-scaled values",trans='log10') + 
  ylab("Density") + 
  theme(legend.position = "right") +
  scale_color_manual(name="Mice age",values=c("red","orange","magenta","green","blue","black")) +
  facet_wrap(.~feature, nrow = 2, scales = "free")

G.all


G.box = ggplot(p1to7_all_long, aes(x=Mice_age, y=value, fill=Mice_age,color=Mice_age)) + 
          facet_wrap(.~feature, nrow = 2, scales = "free") +   
          geom_boxplot(width = 0.8, outlier.size=0.4) +
          theme(legend.position = "right") +  
          scale_y_continuous(name="Log10-scaled values", trans='log10') + 
          scale_fill_manual(name="Mice age",values = c("red","orange","magenta","green","blue","black")) +
          scale_color_manual(name="Mice age",values = c("red","orange","magenta","green","blue","black")) 

G.box




