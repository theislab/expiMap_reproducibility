library(ggplot2)

res_fry <- read.csv("kang_covid_gsea_limma_fry.csv")
res_expimap <- read.csv("expiMap_reproducibility-/pbmc_covid/09122021_kang_covid_bayes_factors.csv")
res_expimap$X <- NULL


head(res_fry)
head(res_expimap)


# clean up column (contrast) names in fry  and expimap results ----
res_expimap <- tidyr::spread(res_expimap, key = 'contrast', value = 'bf')
colnames(res_expimap) <- gsub(" \\(query\\)", "", colnames(res_expimap))

colnames(res_fry) <- gsub("\\.\\.","\\+ ", colnames(res_fry))
colnames(res_fry) <- gsub("\\."," ", colnames(res_fry))
colnames(res_fry) <- gsub("INF Beta","INF-Beta", colnames(res_fry))


# identify common contrasts between the two sets of gsea analysis -----
all(colnames(res_expimap)[-1] %in% colnames(res_fry)[-1])

expimap_contrasts <- colnames(res_expimap)[-1]
fry_contrasts <- colnames(res_fry)[-1]
expimap_contrasts[!expimap_contrasts %in% fry_contrasts]


all(res_fry$X == res_expimap$terms)

# match row names (go terms) between expimap and fry --------
res_expimap <- res_expimap[match(res_fry$X, res_expimap$terms),]
all(res_fry$X == res_expimap$terms)





# need to compute go_size ------
library(zellkonverter)
library(SingleCellExperiment)

adata <- readH5AD("kang_covid_pbmc.h5ad")

# gmt2list
reactome <- readLines("reactome.gmt")
reactome <- strsplit(reactome, split = "\t")
names(reactome) <-  lapply(reactome, FUN=function(x) x[1])
reactome <- lapply(reactome, FUN=function(x) x[-c(1:2)])
reactome <- reactome[metadata(adata)$terms]
length(reactome)

gos <- limma::ids2indices(reactome, rownames(adata))
go_size <- unlist(lapply(gos, length))



res_expimap$go_size <- go_size


# prepare data for plotting by ggplot2 --------
ggdat_expimap <- reshape2::melt(res_expimap, id = c("terms","go_size"))
colnames(ggdat_expimap) <- gsub("value", "expimap", 
                                colnames(ggdat_expimap))




ggdat_fry <- reshape2::melt(res_fry, id = c("X"))
colnames(ggdat_fry) <- gsub("value", "gsea", 
                            colnames(ggdat_fry))


ggdat_fry$gsea <- -log10(ggdat_fry$gsea)
ggdat_expimap$expimap <- abs(ggdat_expimap$expimap)









groups <- intersect(expimap_contrasts, fry_contrasts)



ggdat_expimap$terms <- gsub("_"," ", ggdat_expimap$terms)
ggdat_expimap$terms <- gsub("REACTOME","", ggdat_expimap$terms)



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



ggdat_expimap$terms <- sapply(ggdat_expimap$terms, wrap_text, n= 40)

pos <- position_jitter(width = 0.2, seed = 2, height = 0.15)

for(group in  groups){
  ggdat_stimulated <- cbind(ggdat_expimap[ggdat_expimap$variable %in% group ,], 
                            "gsea" = ggdat_fry[ggdat_fry$variable %in% group, "gsea"])
  
  
  pdf(file = paste0("kang_covid_gsea_figures_limmafry/expimap_vs_gsea_kang_pbmc_",group,"_limma_fry.pdf"), 
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
                    aes(label = terms), size = 2.8, 
                    #force_pull = 2,
                    #force = 10,
                    position = pos,
                    # nudge_x = 0.35, 
                    # nudge_y =0.3,
                    check_overlap = TRUE,
                    max.overlaps = 15,
                    box.padding = unit(0.35, "lines"),
                    point.padding = unit(0.3, "lines")
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
  
  

  print(p)
  dev.off()
  
}

