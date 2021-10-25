### Load required modules

library(NMF)
library(dplyr)
library(igraph)
library(Matrix)
library(ggplot2)
library(CellChat) 
library(patchwork)
library(ggalluvial)
library(reticulate)
options(stringsAsFactors = FALSE)

### Read in data

ad <- import("anndata", convert = FALSE)
pd <- import("pandas", convert = FALSE)
ad_object <- ad$read_h5ad("/Volumes/TIGERII/nobackup/CTRL_anotated.h5ad")

### Access expression matrix

data.input <- t(py_to_r(ad_object$X))

### Add metadata

rownames(data.input) <- rownames(py_to_r(ad_object$var))
colnames(data.input) <- rownames(py_to_r(ad_object$obs))

meta.data <- py_to_r(ad_object$obs)
meta <- meta.data

### Normalise data 

data.input <- normalizeData(data.input, scale.factor = 10000, do.log = TRUE)

### Create `cellchat` object

cellchat <- createCellChat(object = data.input, meta = meta, group.by = "celltype")

### Set up ligand-receptor interaction database for `cellchat`

CellChatDB <- CellChatDB.human
showDatabaseCategory(CellChatDB)
dplyr::glimpse(CellChatDB$interaction)
CellChatDB.use <- CellChatDB
cellchat@DB <- CellChatDB.use

### Process expression data

#future::plan("multiprocess", workers = 3)
#options(future.globals.maxSize = 8912896000)

cellchat <- subsetData(cellchat)

cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)
cellchat <- projectData(cellchat, PPI.human)

cellchat <- computeCommunProb(cellchat, raw.use = TRUE)
cellchat <- filterCommunication(cellchat, min.cells = 5)

### Export results as dataframe

df.net <- subsetCommunication(cellchat)
head(df.net)
#write.table(df.net, sep = ',', row.names = FALSE, 'IFNB_cellchat_net.csv')

### Infer cell-cell communication

cellchat <- computeCommunProbPathway(cellchat)

### Calculate aggregated cell-cell communication

cellchat <- aggregateNet(cellchat)

groupSize <- as.numeric(table(cellchat@idents))
par(mfrow = c(1,3), xpd = TRUE)
options(repr.plot.width = 40, repr.plot.height = 40)
netVisual_circle(cellchat@net$count, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Number of interactions")
netVisual_circle(cellchat@net$weight, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Interaction weights/strength")

mat <- cellchat@net$weight
par(mfrow = c(3,3), xpd=TRUE)
for (i in 1:nrow(mat)) {
  mat2 <- matrix(0, nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
  mat2[i, ] <- mat[i, ]
  netVisual_circle(mat2, vertex.weight = groupSize, weight.scale = T, edge.weight.max = max(mat), title.name = rownames(mat)[i])
}

unique(df.net$pathway_name)

options(repr.plot.width = 10, repr.plot.height = 15)
pathways.show <- c("ANNEXIN") 
netAnalysis_contribution(cellchat, signaling = pathways.show)
vertex.receiver = seq(1,4) # a numeric vector. 
netVisual_aggregate(cellchat, signaling = pathways.show,  vertex.receiver = vertex.receiver)

par(mfrow=c(1,1))
netVisual_aggregate(cellchat, signaling = pathways.show, layout = "circle")

netVisual_aggregate(cellchat, signaling = pathways.show, layout = "chord")

cellchat <- netAnalysis_computeCentrality(cellchat, slot.name = "netP")

options(repr.plot.width = 10, repr.plot.height = 10)
gg1 <- netAnalysis_signalingRole_scatter(cellchat)
gg2 <- netAnalysis_signalingRole_scatter(cellchat, signaling = c("MHC-I", "MHC-II"))
gg1 + gg2

options(repr.plot.width = 5, repr.plot.height = 5)
ht1 <- netAnalysis_signalingRole_heatmap(cellchat, pattern = "outgoing", width = 17, height = 19, color.heatmap = "YlGnBu")
ht2 <- netAnalysis_signalingRole_heatmap(cellchat, pattern = "incoming", width = 17, height = 19, color.heatmap = "YlGnBu")
ht1 + ht2

### Identify global communication patterns

selectK(cellchat, pattern = "outgoing")

options(repr.plot.width = 15, repr.plot.height = 15)
nPatterns = 5
cellchat <- identifyCommunicationPatterns(cellchat, pattern = "outgoing", k = nPatterns,  width = 10, height = 10)

### Sankey plot

options(repr.plot.width = 40, repr.plot.height = 22.5)
netAnalysis_river(cellchat, pattern = "outgoing", font.size = 2.5, font.size.title = 20)

netAnalysis_dot(cellchat, pattern = "outgoing")

pairLR.pathway <- extractEnrichedLR(cellchat, signaling = pathways.show, geneLR.return = FALSE)
LR.show <- pairLR.pathway[1,] # show one ligand-receptor pair

# Hierarchy plot

vertex.receiver = seq(1,4) 
netVisual_individual(cellchat, signaling = pathways.show, pairLR.use = LR.show, layout = "chord")

plotGeneExpression(cellchat, signaling = "CCL")


