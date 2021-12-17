library(zellkonverter)
library(limma)
library(edgeR)
library(SingleCellExperiment)
library(parallel)
library(foreach)
library(doParallel)

adata <- readH5AD("kang_pbmc_merged_061221.h5ad")


# gmt2list
reactome <- readLines("reactome.gmt")
reactome <- strsplit(reactome, split = "\t")
names(reactome) <-  lapply(reactome, FUN=function(x) x[1])
reactome <- lapply(reactome, FUN=function(x) x[-c(1:2)])
length(reactome)
reactome <- reactome[metadata(adata)$terms]
length(reactome)

condition <- adata$condition_joint
celltype <- adata$cell_type_joint
study <- adata$study
y <- DGEList(counts = assay(adata))
y <- calcNormFactors(y, method = "TMMwsp")


######################
# fixed effect models
####################


# stim vs ctrl (over all celltypes) -----
design <- model.matrix(~ study + condition)
v <- voom(y, design, plot = FALSE)

idx <- ids2indices(reactome, rownames(v))

go_enrich_stimvsctrl <- fry(v, index = idx, 
                            design = design,
                            sort = "none",
                            contrast = ncol(design)
)$FDR.Mixed


names(go_enrich_stimvsctrl) <- names(idx)

# stim vs ctrl per celltype ------
# celltype should exist in both stim and ctrl

table(celltype, condition)



keep <- (celltype %in% levels(celltype)[rowSums(table(celltype, condition) != 0) > 1])
celltype2 <- celltype[keep]
celltype2 <- droplevels(celltype2)

study2 <- study[keep]
study2 <- droplevels(study2)

condition2 <- condition[keep]
condition2 <- droplevels(condition2)



design <- model.matrix(~ study2 + condition2)
v <- voom(y[,keep], design, plot = FALSE)

idx <- ids2indices(reactome, rownames(v))


# Parallelize
registerDoParallel(20)
celltypes_unique <- levels(celltype2)[-c(6,8)]
ncelltypes <- length(celltypes_unique)
go_enrich <- foreach(i=seq_len(ncelltypes),
                     .combine = 'cbind',
                     .export = c("design", "v","idx"),
                     .packages = c("limma"),
                     .final = function(x) {
                       colnames(x) <- paste0("stimVsCtrl_", celltypes_unique)
                       return(x)
                     }) %dopar% {
                       cat("processing celltype", celltypes_unique[i],"\n")
                       fry(v[,celltype2 == celltypes_unique[i]],
                           index = idx, 
                           design = design[celltype2 == celltypes_unique[i],],
                           contrast = ncol(design),
                           sort = "none"
                       )$FDR.Mixed
                     }


rownames(go_enrich) <- names(idx)
stopImplicitCluster()


# one celltype vs everything else ---------
# again, this can only be done for celltypes which give full rank matrix

registerDoParallel(20)
celltypes_unique <- levels(celltype)
ncelltypes <- length(celltypes_unique)
go_enrich_oneVsALL <- foreach(i=seq_len(ncelltypes),
                              .combine = 'cbind',
                              .export = c("design", "v","idx"),
                              .packages = c("limma"),
                              .final = function(x) {
                                colnames(x) <- celltypes_unique
                                return(x)
                              }) %dopar% {
                                cat("processing celltype", celltypes_unique[i],"\n")
                                cell <- factor(ifelse(celltype == celltypes_unique[i],
                                                      celltypes_unique[i],
                                                      "other"), 
                                               levels = c("other", celltypes_unique[i]))
                                design <- model.matrix(~ study + cell)
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



res <- do.call(cbind, list('stimvsctrl' = go_enrich_stimvsctrl,go_enrich,
                           go_enrich_oneVsALL))








#################
# Random effect model
#################
y <- DGEList(counts = assay(adata))

samples <- data.frame(cbind("condition" = as.character(adata$condition_joint),
                            "celltype" = as.character(adata$cell_type_joint),
                            "study" = as.character(adata$study)))



# subset on two studies only ----
y2 <- y[, samples$study %in% c("Freytag","Kang")]
samples <- samples[samples$study %in% c("Freytag","Kang"), ]


# downsample (within the subsetted data) -----
set.seed(2021)
dwn_idx <- sample(seq_len(ncol(y2)), 5000, replace = FALSE)

y2 <- y2[,dwn_idx]
samples <- samples[dwn_idx,]


y2 <- calcNormFactors(y2, method = "TMMwsp")
design <- model.matrix(~ condition + celltype, data = samples)

table(samples$condition, samples$study)

corr <- duplicateCorrelation(edgeR::cpm(y2, log=TRUE, prior.counts = 2),
                             design, block = samples$study)

corr$cor

v <- voom(y2, design, plot=FALSE,
          correlation = corr$cor,
          block = samples$study)




idx <- ids2indices(reactome, rownames(v))

registerDoParallel(20)

go_enrich <- foreach(i=2:ncol(design),
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
                           sort = "none",
                           correlation = corr$cor,
                           block = samples$study)$FDR.Mixed
                     }


rownames(go_enrich) <- names(idx)

stopImplicitCluster()
