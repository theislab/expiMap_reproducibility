library(zellkonverter)
library(limma)
library(edgeR)
library(SingleCellExperiment)
library(parallel)
library(foreach)
library(doParallel)

adata <- readH5AD("kang_covid_pbmc.h5ad")

# gmt2list
reactome <- readLines("reactome.gmt")
reactome <- strsplit(reactome, split = "\t")
names(reactome) <-  lapply(reactome, FUN=function(x) x[1])
reactome <- lapply(reactome, FUN=function(x) x[-c(1:2)])
length(reactome)
reactome <- reactome[metadata(adata)$terms]
length(reactome)


table(adata$source2, adata$cell_type_joint)
table(adata$source, adata$cell_type_joint)
table(adata$source, adata$source2)

table(adata$source)
table(adata$cell_type_joint)
table(is.na(adata$cell_type_joint))


celltypes <- adata$cell_type_joint
treatment <- factor(gsub(" \\(query\\)", "" ,adata$source2))
patient <- gsub("(P[12])-day.*","\\1", adata$source)
patient <- gsub("INF-Beta|control","reference", patient)


# drop Unknown cells

j <- grep("Unknown", celltypes, invert = TRUE)

celltypes <- celltypes[j]
celltypes <- droplevels(celltypes)

treatment <- treatment[j]
patient <- patient[j]




y <- DGEList(counts = assay(adata)[,j])
y <- calcNormFactors(y, method = "TMMwsp")

idx <- ids2indices(reactome, rownames(y))


# stim vs ctrl (over all celltypes) -----
design <- model.matrix(~ treatment)
v <- voom(y, design, plot = FALSE)


registerDoParallel(3)

go_enrich_treatment <- foreach(i=2:ncol(design),
        .combine = 'cbind',
        .export = c("design", "v","idx"),
        .packages = c("limma"),
        .final = function(x) {
          colnames(x) <- colnames(design)[2:ncol(design)]
          return(x)
          }) %dopar% {
            cat("processing", i, "/", ncol(design),"\n")
          fry(v, index = idx,
              design = design,
              contrast = i,
              sort = "none"
              )$FDR.Mixed
          }


stopImplicitCluster()
rownames(go_enrich_treatment) <- names(idx)





# one celltype vs everything else ---------
# this removes/adjusts for baseline differences due to treatment/condition (e.g. severe, remission)

registerDoParallel(20)
celltypes_unique <- levels(celltypes)
ncelltypes <- length(celltypes_unique)
freq_tbl <- table(celltypes, treatment)
not_full_rank <- rowSums(freq_tbl != 0) == 1

go_enrich_oneVsALL <- foreach(i=seq_len(ncelltypes),
                              .combine = 'cbind',
                              .export = c("y","idx", "treatment", "celltypes"),
                              .packages = c("limma"),
                              .final = function(x) {
                                colnames(x) <- celltypes_unique
                                return(x)
                              }) %dopar% {
                                cat("processing celltype", celltypes_unique[i],"\n")
                                cell <- factor(ifelse(celltypes == celltypes_unique[i],
                                                      celltypes_unique[i],
                                                      "other"), 
                                               levels = c("other", celltypes_unique[i]))
                                
                                if(not_full_rank[celltypes_unique[i]]){
                                  
                                  design <- model.matrix(~ cell)
                                  
                                }else{
                                  
                                  design <- model.matrix(~ treatment + cell)
                                }
                               
                                #print(head(design))
                                v <- voom(y, design, plot = FALSE)
                                fry(v,
                                    index = idx, 
                                    design = design,
                                    contrast = ncol(design),
                                    sort = "none"
                                )$FDR.Mixed
                              }


rownames(go_enrich_oneVsALL) <- names(idx)
stopImplicitCluster()



# stim vs ctrl per celltype ------
# celltype should exist in both stim and ctrl
# celltype-specific enrichment in remission and sever

# this encodes a combination of what makes the model for a cell type not full rank
# and have sufficient number of cells
not_full_rank <- rowSums(freq_tbl > 20) == 1








# Parallelize
registerDoParallel(10)


## correct model is ~ patient + condition, subsetting only on query data
## expimap results however compare to control, which itself comes from multiple studies
## remission refers to  Day5 (D5) or Day7 (D7)
## severe is Day1 (D1)
## control cells span multiple studies, but study is not modeled in expimap results

condition_name <- "Severe"
celltypes_unique <- levels(celltypes)[!not_full_rank & rowSums(freq_tbl > 3) > 1 & freq_tbl[, condition_name] != 0]
go_enrich_severe <- foreach(i=seq_along(celltypes_unique),
                     .combine = 'cbind',
                     .export = c("y","idx","celltypes","treatment", 
                                 "condition_name","celltypes_unique",
                                 "freq_tbl","patient"),
                     .packages = c("limma"),
                     .final = function(x) {
                       colnames(x) <- paste(celltypes_unique, condition_name, sep = "_")
                       return(x)}
                     ) %dopar% {
                       cat("processing celltype", celltypes_unique[i],"\n")
                       y2 <- y[, celltypes == celltypes_unique[i]]
                       condition <- treatment[celltypes == celltypes_unique[i]]
                       condition <- droplevels(condition)
                       p <- patient[celltypes == celltypes_unique[i]]
                       p <- factor(p, levels = c('reference', 'P1', 'P2'))
                       
                       
                       d2 <- model.matrix(~ p + condition) 
                       # corr <- duplicateCorrelation(edgeR::cpm(y2, log=TRUE,
                       #                                         prior.count = 5),
                       #                              design = d2,
                       #                              block = p)
                       d2 <- d2[,grep("Remission", colnames(d2), invert=TRUE)]
                       v2 <- voom(y2, d2, 
                                  #block = p, correlation = corr$cor,
                                  plot = FALSE)
                       
                       fry(v2,
                           index = idx, 
                           design = d2,
                           contrast = grep(condition_name, colnames(d2)),
                           #block = p, correlation = corr$cor,
                           sort = "none"
                       )$FDR.Mixed
                       
                      
                     }


rownames(go_enrich_severe) <- names(idx)
stopImplicitCluster()



## remission
registerDoParallel(10)

condition_name <- "Remission"
celltypes_unique <- levels(celltypes)[!not_full_rank & rowSums(freq_tbl > 3) > 1 & freq_tbl[, condition_name] != 0]
go_enrich_remission <- foreach(i=seq_along(celltypes_unique),
                            .combine = 'cbind',
                            .export = c("y","idx","celltypes","treatment", 
                                        "condition_name","celltypes_unique",
                                        "freq_tbl","patient"),
                            .packages = c("limma"),
                            .final = function(x) {
                              colnames(x) <- paste(celltypes_unique, condition_name, sep = "_")
                              return(x)}
) %dopar% {
  cat("processing celltype", celltypes_unique[i],"\n")
  y2 <- y[, celltypes == celltypes_unique[i]]
  condition <- treatment[celltypes == celltypes_unique[i]]
  condition <- droplevels(condition)
  p <- patient[celltypes == celltypes_unique[i]]
  p <- factor(p, levels = c('reference', 'P1', 'P2'))
  
  
  d2 <- model.matrix(~ p + condition) 
  # corr <- duplicateCorrelation(edgeR::cpm(y2, log=TRUE,
  #                                         prior.count = 5),
  #                              design = d2,
  #                              block = p)
  d2 <- d2[,grep("Severe", colnames(d2), invert=TRUE)]
  v2 <- voom(y2, d2, 
             #block = p, correlation = corr$cor,
             plot = FALSE)
  
  fry(v2,
      index = idx, 
      design = d2,
      contrast = grep(condition_name, colnames(d2)),
      #block = p, correlation = corr$cor,
      sort = "none"
  )$FDR.Mixed
  
  
}


rownames(go_enrich_remission) <- names(idx)
stopImplicitCluster()



# IFN ---


registerDoParallel(10)

condition_name <- "INF-Beta"
celltypes_unique <- levels(celltypes)[!not_full_rank & rowSums(freq_tbl > 3) > 1 & freq_tbl[, condition_name] != 0]
go_enrich_ifn <- foreach(i=seq_along(celltypes_unique),
                               .combine = 'cbind',
                               .export = c("y","idx","celltypes","treatment", 
                                           "condition_name","celltypes_unique",
                                           "freq_tbl","patient"),
                               .packages = c("limma"),
                               .final = function(x) {
                                 colnames(x) <- paste(celltypes_unique, condition_name, sep = "_")
                                 return(x)}
) %dopar% {
  cat("processing celltype", celltypes_unique[i],"\n")
  y2 <- y[, celltypes == celltypes_unique[i]]
  condition <- treatment[celltypes == celltypes_unique[i]]
  condition <- droplevels(condition)
  p <- patient[celltypes == celltypes_unique[i]]
  p <- factor(p, levels = c('reference', 'P1', 'P2'))
  
  
  d2 <- model.matrix(~ condition) 
  # corr <- duplicateCorrelation(edgeR::cpm(y2, log=TRUE,
  #                                         prior.count = 5),
  #                              design = d2,
  #                              block = p)
  #d2 <- d2[,grep("Severe", colnames(d2), invert=TRUE)]
  v2 <- voom(y2, d2, 
             #block = p, correlation = corr$cor,
             plot = FALSE)
  
  fry(v2,
      index = idx, 
      design = d2,
      contrast = grep(condition_name, colnames(d2)),
      #block = p, correlation = corr$cor,
      sort = "none"
  )$FDR.Mixed
  
  
}


rownames(go_enrich_ifn) <- names(idx)
stopImplicitCluster()


res <- do.call(cbind, list(go_enrich_severe, 
                           go_enrich_remission,
                           go_enrich_ifn))

# write.csv(res, file = "kang_covid_gsea_limma_fry.csv",
#           row.names = TRUE, quote = FALSE)
