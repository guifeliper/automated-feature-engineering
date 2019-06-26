library(parallel)
library(mfe)

# Definition of datasets
datasets <-  list.files('./data/base/',full.names = FALSE)

  
# Parallel only for Linux and OS X
mclapply(datasets, function(ds){
  metadata <- data.frame(read.csv(paste0("./data/filter_selection/", ds), header=TRUE, sep=","))
  rownames(metadata) <- make.names(metadata$feature, unique = TRUE)
  metadata <- metadata[ , -which(names(metadata) %in% c("X"))]
  
  X <- read.csv(paste0("./data/base/", ds), header=TRUE, sep=",")
  X_derived <- read.csv(paste0("./data/filter_extended/", ds), header=TRUE, sep=",")
  
  # Removing the base features
  dropList <- colnames(X)
  d.features <- X_derived[, !colnames(X_derived) %in% dropList]
  
  #Selecting class
  d.labels <- X[,'class']
  j = 1 
  numberOfFeatures = ncol(d.features)
  columns <- make.names(colnames(d.features))
  for (column in c(columns)){
    print(paste(j, 'of', numberOfFeatures, 'on', ds))
    d.extX <- X_derived[, !colnames(X_derived) %in% column]
    d.info <- tryCatch(
      metafeatures(d.extX, d.labels, groups=c("general","statistical", "model.based" )),
      error=function(e) {print(paste(e));NaN}
    )
    
    i = 1 
    for (info in d.info)
    {
      name = paste0(tolower(names(d.info)[i]))
      metadata[c(column), c(name)] <- info
      i = i + 1
    }
    j = j + 1
  }
  # Saving
  write.csv(metadata, file = paste0("./data/mfe_ext_x/", ds))
  
}, mc.cores=8L)
