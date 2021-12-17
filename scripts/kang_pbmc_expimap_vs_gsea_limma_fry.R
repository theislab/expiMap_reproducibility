library(ggplot2)

res_fry <- read.csv("07122021_kang_pbmc_go_enrich_filteredGO.csv")
# res_expimap <- read.csv("09122021_expimap_bayes_factors.csv")
# alternatively, this:
res_expimap <- read.csv("expimap_bayes_factors.csv")



head(res_fry)
head(res_expimap)

all(res_fry$X == res_expimap$X)

colnames(res_fry) <- gsub("stimvsctrl", "stimulated",
                          colnames(res_fry))

colnames(res_fry) <- gsub("(stimVsCtrl)_(.*)", "\\2_stimulated",
                          colnames(res_fry))



ggdat_expimap <- reshape2::melt(res_expimap, id = c("X","go_size"))
colnames(ggdat_expimap) <- gsub("value", "expimap", 
                                colnames(ggdat_expimap))


table(abs(ggdat_expimap$expimap) > 3)


ggdat_fry <- reshape2::melt(res_fry, id = c("X"))
colnames(ggdat_fry) <- gsub("value", "gsea", 
                                colnames(ggdat_fry))


ggdat_fry$gsea <- -log10(ggdat_fry$gsea)
ggdat_expimap$expimap <- abs(ggdat_expimap$expimap)



# all contrasts -----
groups <- intersect(as.character(ggdat_expimap$variable),
                    as.character(ggdat_fry$variable))
                    


ggdat_expimap$X <- gsub("_"," ", ggdat_expimap$X)
ggdat_expimap$X <- gsub("REACTOME","", ggdat_expimap$X)



library(gridExtra)
library(ggrepel)
library(stringr)

wrap_text <- function(string, n) {
  spaces <- str_locate_all(string, " ")[[1]][,1]
  chars  <- nchar(string)
  for(i in 1:floor(chars/n)) {
    s <- spaces[which.min(abs(spaces - n*i))]
    substring(string, s, s) <- "\n "
  }
  return(string)
}



ggdat_expimap$X <- sapply(ggdat_expimap$X, wrap_text, n= 35)

pos <- position_jitter(width = 0.2, seed = 2, height = 0.15)

for(group in  groups){
  ggdat_stimulated <- cbind(ggdat_expimap[ggdat_expimap$variable %in% group ,], 
                            "gsea" = ggdat_fry[ggdat_fry$variable %in% group, "gsea"])
  
  
  pdf(file = paste0("kang_pbmc_gsea_figures_limmafry/expimap_vs_gsea_kang_pbmc_",group,"_limma_fry.pdf"), 
      width = 7, height = 7)
  p <- ggplot(ggdat_stimulated, aes(x= gsea, y = expimap, size = go_size)) +
    geom_point() + 
    labs(title = gsub("\\.|_", " ", group), size = "Gene set size") +
    
    geom_vline(xintercept = 1.3, 
               color = "grey50", 
               linetype="dashed",
               size = 1.2)+
    geom_hline(yintercept = 2.3, 
               color = "grey50", 
               linetype="dashed",
               size = 1.2)+
    
    geom_text_repel(data = subset(ggdat_stimulated, gsea > 1.3 | expimap > 2.3),
                    aes(label = X), size = 2.8, 
                    #force_pull = 2,
                    #force = 10,
                    position = pos,
                    # nudge_x = 0.35, 
                    # nudge_y =0.3,
                    check_overlap = TRUE,
                    max.overlaps = 5,
                    #box.padding = unit(0.35, "lines"),
                    #point.padding = unit(0.3, "lines")
    ) +
    
    labs(x = expression(GSEA~-log[10](FDR)),
         y = expression("|log"[e]~Bayes~"Factor|")) + 
    xlim(0,NA) +
    theme_bw() + theme(
      legend.key.size = unit(0.05, 'cm'),
      #legend.position = c(0.28, 1),
      #legend.position = "None",
      panel.border = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(), 
      axis.line = element_line(colour = "black"),
      axis.text =  element_text(colour = "black", size = 12)) + 
    scale_x_continuous(sec.axis=sec_axis(~., labels = NULL,breaks = NULL)) +
    scale_y_continuous(sec.axis=sec_axis(~., labels = NULL,breaks = NULL)) 
  
  
  #i = i + 1
  
  print(p)
  dev.off()
  
}


### make plots individually -------


# group <- "B"
# title <- "B cells"


# group <- "CD16..Monocytes"
# title <- "CD16+ Monocytes"

group <- "CD14..Monocytes"
title <- "CD14+ Monocytes"


# group <- "stimulated"
# title <- "IFN stimulated"


# group <- "CD14..Monocytes_stimulated"
# title <- "CD14+ Monocytes_stimulated"




ggdat_stimulated <- cbind(ggdat_expimap[ggdat_expimap$variable %in% group ,],
                          "gsea" = ggdat_fry[ggdat_fry$variable %in% group, "gsea"])



png(file = paste0("./13122021_expimap_vs_gsea_kang_pbmc_",group,"_limma_fry.png"), 
    width=5,height=5,units="in",res=1200)
p <- ggplot(ggdat_stimulated, aes(x= gsea, y = expimap, size = go_size)) +
  geom_point() + 
  labs(title = gsub("\\.|_", " ", title), size = "Gene set size") +
  
  geom_vline(xintercept = 1.3, 
             color = "grey50", 
             linetype="dashed",
             size = 1.2)+
  geom_hline(yintercept = 2.3, 
             color = "grey50", 
             linetype="dashed",
             size = 1.2)+
  
  geom_text_repel(data = subset(ggdat_stimulated, gsea > 1.3 | expimap > 2.3),
                  aes(label = X), size = 2.3, 
                  #force_pull = 2,
                  #force = 10,
                  #position = pos,
                  # nudge_x = 0.35, 
                  # nudge_y =0.3,
                  #check_overlap = TRUE,
                  max.overlaps = 10,
                  #box.padding = unit(0.35, "lines"),
                  #point.padding = unit(0.3, "lines")
  ) +
  
  labs(x = expression(GSEA~-log[10](FDR)),
       y = expression("|log"[e]~Bayes~"Factor|")) + 
  xlim(0,NA) +
  theme_bw() + theme(
    legend.key.size = unit(0.05, 'cm'),
    legend.position = c(0.85, 0.2),
    #legend.position = "None",
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(), 
    axis.line = element_line(colour = "black"),
    axis.text =  element_text(colour = "black", size = 14),
    plot.title = element_text(colour = "black", size = 14),
    axis.title = element_text(colour = "black", size = 14)) + 
  scale_x_continuous(sec.axis=sec_axis(~., labels = NULL,breaks = NULL)) +
  scale_y_continuous(sec.axis=sec_axis(~., labels = NULL,breaks = NULL)) 


#i = i + 1

print(p)
dev.off()
